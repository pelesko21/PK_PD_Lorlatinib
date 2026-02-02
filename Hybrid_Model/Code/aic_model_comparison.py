"""
AIC Comparison of Three Pharmacokinetic Models for Lorlatinib

Models compared:
1. Original Gosia Model - 2-compartment with c_in modifying elimination rate
2. Chen et al. Model - 2-compartment with time-varying clearance
3. Hybrid Model - Chen structure + c_in personalization

Author: Generated for MetabolismProject_BasantaLab_Rotation_2025
Date: November 2025
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Clinical Data (from Bauer et al. 2021)
# =============================================================================
CLINICAL_DATA = {
    'single_dose': {
        'dose_mg': 100,
        'Cmax_mean': 695.2,  # ng/mL
        'Cmax_CV': 0.40,
        'Tmax_mean': 1.15,   # hours
        'AUC_mean': 9088,    # ng·h/mL
        'AUC_CV': 0.35,
        'CL_F_mean': 11.01,  # L/h
        'n': 19
    },
    'steady_state_day15': {
        'dose_mg': 100,
        'Cmax_mean': 576.5,  # ng/mL
        'AUCtau_mean': 5650, # ng·h/mL
        'CL_F_mean': 17.70,  # L/h
        'n': 22
    }
}


# =============================================================================
# Model 1: Original Gosia Model
# =============================================================================
class GosiaModel:
    """
    Original 2-compartment model with c_in modifying elimination rate.

    Compartments: Depot (LD) → Free (LF) ⇄ Bound (LB)
    ODEs:
        dLD/dt = -ka * LD
        dLF/dt = ka*LD - (kel*c_in + kFB)*LF + kBF*LB
        dLB/dt = kFB*LF - kBF*LB
    """

    def __init__(self, c_in: float = 1.0):
        # Original fitted parameters from July 2025
        self.ka = 8.96       # h^-1, absorption rate
        self.kel = 0.1005    # h^-1, elimination rate constant
        self.kFB = 1.9       # h^-1, free to bound transfer
        self.kBF = 1.0       # h^-1, bound to free transfer
        self.L0 = 36252.68   # ng, standard dose amount

        # Estimated volume of distribution (not explicitly in original model)
        # Derived to match clinical Cmax ~695 ng/mL for 100mg dose
        self.V_d = 121.0     # L (using Chen et al. V2 for consistency)

        self.c_in = c_in
        self.n_params = 6  # ka, kel, kFB, kBF, L0, V_d (fixed) + c_in if estimated

    def ode_system(self, y: np.ndarray, t: float) -> List[float]:
        LD, LF, LB = y

        dLDdt = -self.ka * LD
        dLFdt = self.ka * LD - (self.kel * self.c_in + self.kFB) * LF + self.kBF * LB
        dLBdt = self.kFB * LF - self.kBF * LB

        return [dLDdt, dLFdt, dLBdt]

    def simulate_single_dose(self, dose_mg: float = 100.0, duration_hours: float = 168.0) -> Dict:
        """Simulate single dose PK profile."""
        # Convert dose to ng (assumes complete bioavailability if not specified)
        dose_ng = dose_mg * 1e6  # mg to ng

        # Initial conditions: all drug in depot
        y0 = [dose_ng, 0.0, 0.0]

        # Time points
        t = np.linspace(0, duration_hours, 1000)

        # Solve ODEs
        solution = odeint(self.ode_system, y0, t)

        # Convert amounts to concentrations
        C_free = solution[:, 1] / self.V_d  # ng/L = ng/mL (approximately)

        return {
            'time': t,
            'C_central': C_free / 1000,  # Convert to ng/mL (ng/L / 1000)
            'LD': solution[:, 0],
            'LF': solution[:, 1],
            'LB': solution[:, 2]
        }


# =============================================================================
# Model 2: Chen et al. Model (No c_in)
# =============================================================================
class ChenModel:
    """
    Chen et al. (2021) 2-compartment model with time-varying clearance.

    NO personalization parameter (c_in = 1 fixed).
    """

    def __init__(self):
        # Chen et al. (2021) validated parameters
        self.ka = 3.11           # h^-1, absorption rate constant
        self.CL_initial = 9.04   # L/h, initial clearance
        self.CL_max = 14.5       # L/h, steady-state clearance
        self.tau = 34.8          # h, induction time constant
        self.V2 = 121.0          # L, central volume
        self.V3 = 155.0          # L, peripheral volume
        self.Q = 22.0            # L/h, inter-compartmental clearance
        self.F = 0.759           # bioavailability

        self.n_params = 8  # ka, CL_init, CL_max, tau, V2, V3, Q, F

    def clearance(self, t: float) -> float:
        """Time-varying clearance (no c_in)."""
        induction_factor = 1 - np.exp(-t / self.tau)
        return self.CL_initial + (self.CL_max - self.CL_initial) * induction_factor

    def ode_system(self, y: np.ndarray, t: float) -> List[float]:
        A_depot, A_central, A_peripheral = y
        CL_t = self.clearance(t)

        dA_depot = -self.ka * A_depot
        dA_central = (self.ka * A_depot
                     - (CL_t / self.V2) * A_central
                     - (self.Q / self.V2) * A_central
                     + (self.Q / self.V3) * A_peripheral)
        dA_peripheral = (self.Q / self.V2) * A_central - (self.Q / self.V3) * A_peripheral

        return [dA_depot, dA_central, dA_peripheral]

    def simulate_single_dose(self, dose_mg: float = 100.0, duration_hours: float = 168.0) -> Dict:
        """Simulate single dose PK profile."""
        dose_ug = dose_mg * 1000 * self.F  # mg to μg with bioavailability

        y0 = [dose_ug, 0.0, 0.0]
        t = np.linspace(0, duration_hours, 1000)

        solution = odeint(self.ode_system, y0, t)

        A_central = solution[:, 1]
        C_central = (A_central / self.V2) # ng/mL

        return {
            'time': t,
            'C_central': C_central,
            'A_depot': solution[:, 0],
            'A_central': solution[:, 1],
            'A_peripheral': solution[:, 2]
        }


# =============================================================================
# Model 3: Hybrid Model (Chen + c_in)
# =============================================================================
class HybridModel:
    """
    Hybrid model: Chen et al. structure + c_in personalization.

    CL(t, c_in) = c_in × [CL_initial + (CL_max - CL_initial) × (1 - exp(-t/τ))]
    """

    def __init__(self, c_in: float = 1.0):
        # Chen et al. (2021) validated parameters
        self.ka = 3.11
        self.CL_initial = 9.04
        self.CL_max = 14.5
        self.tau = 34.8
        self.V2 = 121.0
        self.V3 = 155.0
        self.Q = 22.0
        self.F = 0.759

        self.c_in = c_in
        self.n_params = 9  # 8 Chen params + c_in

    def clearance(self, t: float) -> float:
        """Time-varying clearance personalized by c_in."""
        induction_factor = 1 - np.exp(-t / self.tau)
        CL_t = self.CL_initial + (self.CL_max - self.CL_initial) * induction_factor
        return self.c_in * CL_t

    def ode_system(self, y: np.ndarray, t: float) -> List[float]:
        A_depot, A_central, A_peripheral = y
        CL_t = self.clearance(t)

        dA_depot = -self.ka * A_depot
        dA_central = (self.ka * A_depot
                     - (CL_t / self.V2) * A_central
                     - (self.Q / self.V2) * A_central
                     + (self.Q / self.V3) * A_peripheral)
        dA_peripheral = (self.Q / self.V2) * A_central - (self.Q / self.V3) * A_peripheral

        return [dA_depot, dA_central, dA_peripheral]

    def simulate_single_dose(self, dose_mg: float = 100.0, duration_hours: float = 168.0) -> Dict:
        """Simulate single dose PK profile."""
        dose_ug = dose_mg * 1000 * self.F

        y0 = [dose_ug, 0.0, 0.0]
        t = np.linspace(0, duration_hours, 1000)

        solution = odeint(self.ode_system, y0, t)

        A_central = solution[:, 1]
        C_central = (A_central / self.V2) 

        return {
            'time': t,
            'C_central': C_central,
            'A_depot': solution[:, 0],
            'A_central': solution[:, 1],
            'A_peripheral': solution[:, 2]
        }


# =============================================================================
# AIC Calculation Functions
# =============================================================================
def calculate_pk_metrics(simulation: Dict) -> Dict:
    """Calculate PK metrics from simulation."""
    t = simulation['time']
    C = simulation['C_central']

    Cmax = np.max(C)
    Tmax_idx = np.argmax(C)
    Tmax = t[Tmax_idx]
    AUC = np.trapz(C, t)

    # Clearance: CL/F = Dose / AUC
    # For 100 mg dose with F=0.759: CL = Dose / AUC
    CL_F = 100 * 1000 / AUC  # μg / (ng·h/mL) = L/h

    return {
        'Cmax': Cmax,
        'Tmax': Tmax,
        'AUC': AUC,
        'CL_F': CL_F
    }


def calculate_aic(n_obs: int, n_params: int, rss: float) -> float:
    """
    Calculate Akaike Information Criterion.

    AIC = n * ln(RSS/n) + 2k

    where:
        n = number of observations
        k = number of parameters
        RSS = residual sum of squares
    """
    if rss <= 0:
        return np.inf
    return n_obs * np.log(rss / n_obs) + 2 * n_params


def calculate_bic(n_obs: int, n_params: int, rss: float) -> float:
    """
    Calculate Bayesian Information Criterion.

    BIC = n * ln(RSS/n) + k * ln(n)
    """
    if rss <= 0:
        return np.inf
    return n_obs * np.log(rss / n_obs) + n_params * np.log(n_obs)


def generate_synthetic_clinical_data(n_patients: int = 50, seed: int = 42) -> Dict:
    """
    Generate synthetic clinical observations based on clinical statistics.

    Using log-normal distributions to match reported CV values.
    """
    np.random.seed(seed)

    # Generate individual patient PK parameters
    # Cmax ~ LogNormal(μ, σ) with mean=695.2, CV=0.40
    Cmax_mean = CLINICAL_DATA['single_dose']['Cmax_mean']
    Cmax_CV = CLINICAL_DATA['single_dose']['Cmax_CV']
    sigma_Cmax = np.sqrt(np.log(1 + Cmax_CV**2))
    mu_Cmax = np.log(Cmax_mean) - 0.5 * sigma_Cmax**2
    Cmax_obs = np.random.lognormal(mu_Cmax, sigma_Cmax, n_patients)

    # AUC ~ LogNormal with mean=9088, CV=0.35
    AUC_mean = CLINICAL_DATA['single_dose']['AUC_mean']
    AUC_CV = CLINICAL_DATA['single_dose']['AUC_CV']
    sigma_AUC = np.sqrt(np.log(1 + AUC_CV**2))
    mu_AUC = np.log(AUC_mean) - 0.5 * sigma_AUC**2
    AUC_obs = np.random.lognormal(mu_AUC, sigma_AUC, n_patients)

    # CL/F = Dose / AUC (derived, not independent)
    CL_F_obs = 100 * 1000 / AUC_obs  # L/h

    return {
        'n': n_patients,
        'Cmax': Cmax_obs,
        'AUC': AUC_obs,
        'CL_F': CL_F_obs
    }


def fit_model_to_population(model_class, clinical_data: Dict, fit_cin: bool = False) -> Tuple[object, float]:
    """
    Fit model to population data by finding optimal c_in (if applicable).

    Returns fitted model and RSS.
    """

    def objective(params):
        if fit_cin:
            c_in = params[0]
            model = model_class(c_in=c_in)
        else:
            model = model_class()

        # Simulate for average patient
        sim = model.simulate_single_dose(dose_mg=100, duration_hours=168)
        metrics = calculate_pk_metrics(sim)

        # Calculate RSS against population means
        rss_Cmax = (metrics['Cmax'] - np.mean(clinical_data['Cmax']))**2
        rss_AUC = (metrics['AUC'] - np.mean(clinical_data['AUC']))**2

        # Normalize by scale
        rss = rss_Cmax / np.mean(clinical_data['Cmax'])**2 + rss_AUC / np.mean(clinical_data['AUC'])**2
        return rss

    if fit_cin:
        result = minimize(objective, [1.0], bounds=[(0.1, 5.0)], method='L-BFGS-B')
        optimal_cin = result.x[0]
        if model_class == GosiaModel:
            model = model_class(c_in=optimal_cin)
        else:
            model = model_class(c_in=optimal_cin)
    else:
        model = model_class()

    # Calculate final RSS
    sim = model.simulate_single_dose(dose_mg=100, duration_hours=168)
    metrics = calculate_pk_metrics(sim)

    # RSS for each patient
    total_rss = 0
    for i in range(clinical_data['n']):
        rss_Cmax = (metrics['Cmax'] - clinical_data['Cmax'][i])**2
        rss_AUC = (metrics['AUC'] - clinical_data['AUC'][i])**2
        total_rss += rss_Cmax + rss_AUC

    return model, total_rss


def run_aic_comparison():
    """
    Run comprehensive AIC comparison of all three models.
    """
    print("=" * 70)
    print("AIC MODEL COMPARISON: Lorlatinib PK Models")
    print("=" * 70)
    print()

    # Generate synthetic clinical data
    print("Generating synthetic clinical population (n=50)...")
    clinical_data = generate_synthetic_clinical_data(n_patients=50, seed=42)
    print(f"  Cmax: {np.mean(clinical_data['Cmax']):.1f} ± {np.std(clinical_data['Cmax']):.1f} ng/mL")
    print(f"  AUC:  {np.mean(clinical_data['AUC']):.1f} ± {np.std(clinical_data['AUC']):.1f} ng·h/mL")
    print()

    # Number of observations (2 metrics per patient)
    n_obs = 2 * clinical_data['n']

    results = {}

    # =========================================================================
    # Model 1: Gosia Model
    # =========================================================================
    print("-" * 70)
    print("MODEL 1: Original Gosia Model")
    print("-" * 70)

    # Without c_in fitting (c_in = 1.0)
    goisha_fixed = GosiaModel(c_in=1.0)
    sim_goisha = goisha_fixed.simulate_single_dose(dose_mg=100, duration_hours=168)
    metrics_goisha = calculate_pk_metrics(sim_goisha)

    print(f"  c_in = 1.0 (fixed)")
    print(f"  Cmax: {metrics_goisha['Cmax']:.1f} ng/mL (target: {np.mean(clinical_data['Cmax']):.1f})")
    print(f"  AUC:  {metrics_goisha['AUC']:.1f} ng·h/mL (target: {np.mean(clinical_data['AUC']):.1f})")

    # Calculate RSS
    rss_goisha = 0
    for i in range(clinical_data['n']):
        rss_goisha += (metrics_goisha['Cmax'] - clinical_data['Cmax'][i])**2
        rss_goisha += (metrics_goisha['AUC'] - clinical_data['AUC'][i])**2

    # AIC: Gosia has fewer structural parameters but may fit worse
    # Parameters: ka, kel, kFB, kBF (4 rate constants, no time-varying)
    n_params_goisha = 4
    aic_goisha = calculate_aic(n_obs, n_params_goisha, rss_goisha)
    bic_goisha = calculate_bic(n_obs, n_params_goisha, rss_goisha)

    print(f"  RSS:  {rss_goisha:.2e}")
    print(f"  AIC:  {aic_goisha:.2f}")
    print(f"  BIC:  {bic_goisha:.2f}")
    print(f"  Parameters: {n_params_goisha}")
    print()

    results['Gosia'] = {
        'model': goisha_fixed,
        'metrics': metrics_goisha,
        'rss': rss_goisha,
        'aic': aic_goisha,
        'bic': bic_goisha,
        'n_params': n_params_goisha
    }

    # =========================================================================
    # Model 2: Chen et al. Model
    # =========================================================================
    print("-" * 70)
    print("MODEL 2: Chen et al. Model (No c_in)")
    print("-" * 70)

    chen_model = ChenModel()
    sim_chen = chen_model.simulate_single_dose(dose_mg=100, duration_hours=168)
    metrics_chen = calculate_pk_metrics(sim_chen)

    print(f"  c_in = N/A (no personalization)")
    print(f"  Cmax: {metrics_chen['Cmax']:.1f} ng/mL (target: {np.mean(clinical_data['Cmax']):.1f})")
    print(f"  AUC:  {metrics_chen['AUC']:.1f} ng·h/mL (target: {np.mean(clinical_data['AUC']):.1f})")

    # Calculate RSS
    rss_chen = 0
    for i in range(clinical_data['n']):
        rss_chen += (metrics_chen['Cmax'] - clinical_data['Cmax'][i])**2
        rss_chen += (metrics_chen['AUC'] - clinical_data['AUC'][i])**2

    # Parameters: ka, CL_init, CL_max, tau, V2, V3, Q, F (8 parameters)
    # But for fair comparison, count structural parameters that affect dynamics
    n_params_chen = 8
    aic_chen = calculate_aic(n_obs, n_params_chen, rss_chen)
    bic_chen = calculate_bic(n_obs, n_params_chen, rss_chen)

    print(f"  RSS:  {rss_chen:.2e}")
    print(f"  AIC:  {aic_chen:.2f}")
    print(f"  BIC:  {bic_chen:.2f}")
    print(f"  Parameters: {n_params_chen}")
    print()

    results['Chen'] = {
        'model': chen_model,
        'metrics': metrics_chen,
        'rss': rss_chen,
        'aic': aic_chen,
        'bic': bic_chen,
        'n_params': n_params_chen
    }

    # =========================================================================
    # Model 3: Hybrid Model
    # =========================================================================
    print("-" * 70)
    print("MODEL 3: Hybrid Model (Chen + c_in)")
    print("-" * 70)

    hybrid_model = HybridModel(c_in=1.0)
    sim_hybrid = hybrid_model.simulate_single_dose(dose_mg=100, duration_hours=168)
    metrics_hybrid = calculate_pk_metrics(sim_hybrid)

    print(f"  c_in = 1.0 (average metabolizer)")
    print(f"  Cmax: {metrics_hybrid['Cmax']:.1f} ng/mL (target: {np.mean(clinical_data['Cmax']):.1f})")
    print(f"  AUC:  {metrics_hybrid['AUC']:.1f} ng·h/mL (target: {np.mean(clinical_data['AUC']):.1f})")

    # Calculate RSS
    rss_hybrid = 0
    for i in range(clinical_data['n']):
        rss_hybrid += (metrics_hybrid['Cmax'] - clinical_data['Cmax'][i])**2
        rss_hybrid += (metrics_hybrid['AUC'] - clinical_data['AUC'][i])**2

    # Parameters: 8 Chen params + c_in (9 total)
    n_params_hybrid = 9
    aic_hybrid = calculate_aic(n_obs, n_params_hybrid, rss_hybrid)
    bic_hybrid = calculate_bic(n_obs, n_params_hybrid, rss_hybrid)

    print(f"  RSS:  {rss_hybrid:.2e}")
    print(f"  AIC:  {aic_hybrid:.2f}")
    print(f"  BIC:  {bic_hybrid:.2f}")
    print(f"  Parameters: {n_params_hybrid}")
    print()

    results['Hybrid'] = {
        'model': hybrid_model,
        'metrics': metrics_hybrid,
        'rss': rss_hybrid,
        'aic': aic_hybrid,
        'bic': bic_hybrid,
        'n_params': n_params_hybrid
    }

    # =========================================================================
    # Summary Comparison
    # =========================================================================
    print("=" * 70)
    print("SUMMARY: AIC COMPARISON")
    print("=" * 70)
    print()

    # Find best model (lowest AIC)
    best_model = min(results.keys(), key=lambda x: results[x]['aic'])

    print(f"{'Model':<20} {'AIC':<12} {'BIC':<12} {'ΔAIC':<12} {'ΔBIC':<12} {'RSS':<15} {'Params':<10}")
    print("-" * 95)

    base_aic = results[best_model]['aic']
    base_bic = results[best_model]['bic']

    for model_name in ['Gosia', 'Chen', 'Hybrid']:
        r = results[model_name]
        delta_aic = r['aic'] - base_aic
        delta_bic = r['bic'] - base_bic
        marker = " ← BEST" if model_name == best_model else ""
        print(f"{model_name:<20} {r['aic']:<12.2f} {r['bic']:<12.2f} {delta_aic:<12.2f} {delta_bic:<12.2f} {r['rss']:<15.2e} {r['n_params']:<10}{marker}")

    print()
    print("Interpretation:")
    print(f"  ✓ Best model by AIC: {best_model}")

    # Check ΔAIC > 2 for statistical significance
    for model_name in results:
        if model_name != best_model:
            delta_aic = results[model_name]['aic'] - base_aic
            if delta_aic > 10:
                print(f"  ✓ {best_model} is STRONGLY better than {model_name} (ΔAIC = {delta_aic:.1f} > 10)")
            elif delta_aic > 2:
                print(f"  ✓ {best_model} is significantly better than {model_name} (ΔAIC = {delta_aic:.1f} > 2)")
            else:
                print(f"  ⚠ {best_model} is similar to {model_name} (ΔAIC = {delta_aic:.1f} < 2)")

    print()

    return results


def plot_model_comparison(results: Dict):
    """Create visualization of model comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: AIC Comparison
    ax = axes[0, 0]
    models = list(results.keys())
    aic_values = [results[m]['aic'] for m in models]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax.bar(models, aic_values, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('AIC Score', fontsize=12, fontweight='bold')
    ax.set_title('Akaike Information Criterion (Lower is Better)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, aic_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Plot 2: BIC Comparison
    ax = axes[0, 1]
    bic_values = [results[m]['bic'] for m in models]
    bars = ax.bar(models, bic_values, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('BIC Score', fontsize=12, fontweight='bold')
    ax.set_title('Bayesian Information Criterion (Lower is Better)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, bic_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Plot 3: Cmax Predictions
    ax = axes[1, 0]
    cmax_values = [results[m]['metrics']['Cmax'] for m in models]
    clinical_cmax = 695.2
    bars = ax.bar(models, cmax_values, color=colors, edgecolor='black', linewidth=1.5, label='Model Prediction')
    ax.axhline(clinical_cmax, color='blue', linestyle='--', linewidth=2, label=f'Clinical Target ({clinical_cmax} ng/mL)')
    ax.set_ylabel('Cmax (ng/mL)', fontsize=12, fontweight='bold')
    ax.set_title('Cmax Predictions vs Clinical Target', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, cmax_values):
        pct_err = (val - clinical_cmax) / clinical_cmax * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val:.0f}\n({pct_err:+.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 4: AUC Predictions
    ax = axes[1, 1]
    auc_values = [results[m]['metrics']['AUC'] for m in models]
    clinical_auc = 9088
    bars = ax.bar(models, auc_values, color=colors, edgecolor='black', linewidth=1.5, label='Model Prediction')
    ax.axhline(clinical_auc, color='blue', linestyle='--', linewidth=2, label=f'Clinical Target ({clinical_auc} ng·h/mL)')
    ax.set_ylabel('AUC (ng·h/mL)', fontsize=12, fontweight='bold')
    ax.set_title('AUC Predictions vs Clinical Target', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, auc_values):
        pct_err = (val - clinical_auc) / clinical_auc * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{val:.0f}\n({pct_err:+.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('AIC Model Comparison: Gosia vs Chen vs Hybrid', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


if __name__ == "__main__":
    # Run AIC comparison
    results = run_aic_comparison()

    # Create visualization
    fig = plot_model_comparison(results)
    plt.savefig('AIC_Model_Comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: AIC_Model_Comparison.png")
    plt.close()  # Close figure to avoid blocking
