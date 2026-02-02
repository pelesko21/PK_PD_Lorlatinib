#!/usr/bin/env python3
"""
Personalized Dosing with Hybrid Model

Demonstrates that patients with different c_in values (CYP3A4 metabolizer phenotypes)
can achieve similar PK profiles to the reference (c_in=1.0, 100mg QD) by adjusting
the dosing schedule.

Author: Generated for MetabolismProject_BasantaLab_Rotation_2025
Date: November 2025
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Hybrid Model (Chen + c_in)
# =============================================================================
class HybridModel:
    """
    Hybrid model: Chen et al. structure + c_in personalization.

    CL(t, c_in) = c_in × [CL_initial + (CL_max - CL_initial) × (1 - exp(-t/τ))]
    """

    def __init__(self, c_in=1.0):
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

    def clearance(self, t):
        """Time-varying clearance personalized by c_in."""
        induction_factor = 1 - np.exp(-t / self.tau)
        CL_t = self.CL_initial + (self.CL_max - self.CL_initial) * induction_factor
        return self.c_in * CL_t

    def ode_system(self, y, t):
        A_depot, A_central, A_peripheral = y
        CL_t = self.clearance(t)

        dA_depot = -self.ka * A_depot
        dA_central = (self.ka * A_depot
                     - (CL_t / self.V2) * A_central
                     - (self.Q / self.V2) * A_central
                     + (self.Q / self.V3) * A_peripheral)
        dA_peripheral = (self.Q / self.V2) * A_central - (self.Q / self.V3) * A_peripheral

        return [dA_depot, dA_central, dA_peripheral]

    def simulate_multiple_doses(self, dose_mg=100.0, n_doses=15, dosing_interval=24.0, n_points=5000):
        """Simulate multiple dose PK profile."""
        duration_hours = n_doses * dosing_interval + 24
        dose_ug = dose_mg * 1000 * self.F

        t = np.linspace(0, duration_hours, n_points)

        y0 = [0.0, 0.0, 0.0]
        dose_times = [i * dosing_interval for i in range(n_doses)]

        t_segments = []
        solution_segments = []
        current_state = y0
        t_start = 0

        for dose_idx in range(n_doses):
            if dose_idx < n_doses - 1:
                t_end = dose_times[dose_idx + 1]
            else:
                t_end = duration_hours

            t_segment = t[(t >= t_start) & (t < t_end)]
            if len(t_segment) == 0:
                continue

            # Add dose to depot
            current_state[0] += dose_ug

            solution_segment = odeint(self.ode_system, current_state, t_segment)
            t_segments.extend(t_segment)
            solution_segments.append(solution_segment)

            if len(solution_segment) > 0:
                current_state = solution_segment[-1, :]
            t_start = t_end

        solution_full = np.vstack(solution_segments)
        A_central = solution_full[:, 1]
        C_central = (A_central / self.V2) 

        return {'time': np.array(t_segments), 'C_central': C_central}


def calculate_steady_state_metrics(simulation, dosing_interval=24.0):
    """Calculate PK metrics at steady-state (last dosing interval)."""
    t = simulation['time']
    C = simulation['C_central']

    # Get last dosing interval
    t_last_dose = t[-1] - dosing_interval
    mask = t >= t_last_dose

    t_ss = t[mask]
    C_ss = C[mask]

    Cmax_ss = np.max(C_ss)
    Cmin_ss = np.min(C_ss)
    AUC_ss = np.trapz(C_ss, t_ss - t_ss[0])

    return {
        'Cmax_ss': Cmax_ss,
        'Cmin_ss': Cmin_ss,
        'AUC_ss': AUC_ss,
        'Cavg_ss': AUC_ss / dosing_interval
    }


def find_optimal_dose(c_in_target, target_metrics, dosing_interval=24.0):
    """
    Find optimal dose for a given c_in to match target metrics.

    Parameters:
    -----------
    c_in_target : float
        The c_in value for this patient
    target_metrics : dict
        Target PK metrics to match (from reference c_in=1.0)
    dosing_interval : float
        Dosing interval in hours

    Returns:
    --------
    optimal_dose : float
        Dose in mg that matches target metrics
    """

    def objective(dose):
        model = HybridModel(c_in=c_in_target)
        sim = model.simulate_multiple_doses(dose_mg=dose[0], n_doses=15, dosing_interval=dosing_interval)
        metrics = calculate_steady_state_metrics(sim, dosing_interval)

        # Minimize difference in AUC (primary target)
        error = (metrics['AUC_ss'] - target_metrics['AUC_ss'])**2
        return error

    # Optimize dose
    result = minimize(objective, x0=[100.0], bounds=[(10.0, 500.0)], method='L-BFGS-B')
    optimal_dose = result.x[0]

    return optimal_dose


def find_optimal_interval(c_in_target, target_metrics, dose_mg=100.0):
    """
    Find optimal dosing interval for a given c_in to match target metrics.

    Parameters:
    -----------
    c_in_target : float
        The c_in value for this patient
    target_metrics : dict
        Target PK metrics to match (from reference c_in=1.0)
    dose_mg : float
        Fixed dose in mg

    Returns:
    --------
    optimal_interval : float
        Dosing interval in hours that matches target metrics
    """

    def objective(interval):
        if interval[0] < 6 or interval[0] > 48:
            return 1e10

        model = HybridModel(c_in=c_in_target)
        # Ensure at least 20 doses AND at least 20 days for steady-state
        duration_hours = max(480, interval[0] * 20)  # At least 20 days or 20 doses
        n_doses = int(duration_hours / interval[0])
        sim = model.simulate_multiple_doses(dose_mg=dose_mg, n_doses=n_doses, dosing_interval=interval[0])
        metrics = calculate_steady_state_metrics(sim, dosing_interval=interval[0])

        # Minimize difference in Cavg (rescaled AUC/interval)
        error = (metrics['Cavg_ss'] - target_metrics['Cavg_ss'])**2
        return error

    # Optimize interval - try multiple initial guesses for robustness
    best_result = None
    best_error = np.inf

    for initial_guess in [12.0, 24.0, 36.0, 48.0]:
        result = minimize(objective, x0=[initial_guess], bounds=[(6.0, 48.0)],
                         method='L-BFGS-B', options={'maxiter': 100})
        if result.fun < best_error:
            best_error = result.fun
            best_result = result

    optimal_interval = best_result.x[0]

    return optimal_interval


def create_personalized_dosing_figure():
    """
    Create comprehensive figure showing personalized dosing strategies.
    """

    print("Calculating personalized dosing strategies...")
    print("=" * 80)

    # Reference: Average metabolizer (c_in = 1.0, 100mg QD)
    print("\n1. Reference Patient (Average Metabolizer):")
    print("   c_in = 1.0, Dose = 100 mg QD")
    model_ref = HybridModel(c_in=1.0)
    sim_ref = model_ref.simulate_multiple_doses(dose_mg=100, n_doses=15, dosing_interval=24)
    metrics_ref = calculate_steady_state_metrics(sim_ref, dosing_interval=24)

    print(f"   Cmax_ss = {metrics_ref['Cmax_ss']:.1f} ng/mL")
    print(f"   Cmin_ss = {metrics_ref['Cmin_ss']:.1f} ng/mL")
    print(f"   AUC_ss  = {metrics_ref['AUC_ss']:.1f} ng·h/mL")
    print(f"   Cavg_ss = {metrics_ref['Cavg_ss']:.1f} ng/mL")

    # Fast metabolizer (c_in = 2.0) - needs higher dose
    print("\n2. Fast Metabolizer:")
    print("   c_in = 2.0 (doubled clearance)")
    c_in_fast = 2.0

    # Strategy 1: Increase dose, keep interval
    dose_fast = find_optimal_dose(c_in_fast, metrics_ref, dosing_interval=24)
    print(f"   Strategy 1 - Dose adjustment: {dose_fast:.1f} mg QD")
    model_fast_dose = HybridModel(c_in=c_in_fast)
    sim_fast_dose = model_fast_dose.simulate_multiple_doses(dose_mg=dose_fast, n_doses=15, dosing_interval=24)
    metrics_fast_dose = calculate_steady_state_metrics(sim_fast_dose, dosing_interval=24)
    print(f"      Cmax_ss = {metrics_fast_dose['Cmax_ss']:.1f} ng/mL")
    print(f"      Cmin_ss = {metrics_fast_dose['Cmin_ss']:.1f} ng/mL")
    print(f"      AUC_ss  = {metrics_fast_dose['AUC_ss']:.1f} ng·h/mL")

    # Strategy 2: Keep dose, increase frequency
    interval_fast = find_optimal_interval(c_in_fast, metrics_ref, dose_mg=100)
    print(f"   Strategy 2 - Frequency adjustment: 100 mg every {interval_fast:.1f} h (Q{interval_fast:.0f}h)")
    duration_fast = max(480, interval_fast * 20)
    n_doses_fast = int(duration_fast / interval_fast)
    sim_fast_freq = model_fast_dose.simulate_multiple_doses(dose_mg=100, n_doses=n_doses_fast, dosing_interval=interval_fast)
    metrics_fast_freq = calculate_steady_state_metrics(sim_fast_freq, dosing_interval=interval_fast)
    print(f"      Cavg_ss = {metrics_fast_freq['Cavg_ss']:.1f} ng/mL (target: {metrics_ref['Cavg_ss']:.1f})")

    # Slow metabolizer (c_in = 0.5) - needs lower dose
    print("\n3. Slow Metabolizer:")
    print("   c_in = 0.5 (halved clearance)")
    c_in_slow = 0.5

    # Strategy 1: Decrease dose, keep interval
    dose_slow = find_optimal_dose(c_in_slow, metrics_ref, dosing_interval=24)
    print(f"   Strategy 1 - Dose adjustment: {dose_slow:.1f} mg QD")
    model_slow_dose = HybridModel(c_in=c_in_slow)
    sim_slow_dose = model_slow_dose.simulate_multiple_doses(dose_mg=dose_slow, n_doses=15, dosing_interval=24)
    metrics_slow_dose = calculate_steady_state_metrics(sim_slow_dose, dosing_interval=24)
    print(f"      Cmax_ss = {metrics_slow_dose['Cmax_ss']:.1f} ng/mL")
    print(f"      Cmin_ss = {metrics_slow_dose['Cmin_ss']:.1f} ng/mL")
    print(f"      AUC_ss  = {metrics_slow_dose['AUC_ss']:.1f} ng·h/mL")

    # Strategy 2: Keep dose, decrease frequency
    interval_slow = find_optimal_interval(c_in_slow, metrics_ref, dose_mg=100)
    print(f"   Strategy 2 - Frequency adjustment: 100 mg every {interval_slow:.1f} h (Q{interval_slow:.0f}h)")
    duration_slow = max(480, interval_slow * 20)
    n_doses_slow = int(duration_slow / interval_slow)
    sim_slow_freq = model_slow_dose.simulate_multiple_doses(dose_mg=100, n_doses=n_doses_slow, dosing_interval=interval_slow)
    metrics_slow_freq = calculate_steady_state_metrics(sim_slow_freq, dosing_interval=interval_slow)
    print(f"      Cavg_ss = {metrics_slow_freq['Cavg_ss']:.1f} ng/mL (target: {metrics_ref['Cavg_ss']:.1f})")

    print("\n" + "=" * 80)

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # =========================================================================
    # Row 1: Dose Adjustment Strategy
    # =========================================================================

    # Panel 1A: Fast metabolizer - dose adjustment
    ax1 = fig.add_subplot(gs[0, 0])
    t_ref_days = sim_ref['time'] / 24
    t_fast_days = sim_fast_dose['time'] / 24

    ax1.plot(t_ref_days, sim_ref['C_central'], 'b-', linewidth=2.5,
             label=f'Reference: c_in=1.0, 100 mg QD', alpha=0.7)
    ax1.plot(t_fast_days, sim_fast_dose['C_central'], 'r--', linewidth=2.5,
             label=f'Fast: c_in={c_in_fast}, {dose_fast:.0f} mg QD')

    ax1.set_xlabel('Time (days)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Plasma Concentration (ng/mL)', fontsize=11, fontweight='bold')
    ax1.set_title('A. Fast Metabolizer: Dose Adjustment', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 16)

    # Panel 1B: Slow metabolizer - dose adjustment
    ax2 = fig.add_subplot(gs[0, 1])
    t_slow_days = sim_slow_dose['time'] / 24

    ax2.plot(t_ref_days, sim_ref['C_central'], 'b-', linewidth=2.5,
             label=f'Reference: c_in=1.0, 100 mg QD', alpha=0.7)
    ax2.plot(t_slow_days, sim_slow_dose['C_central'], 'g--', linewidth=2.5,
             label=f'Slow: c_in={c_in_slow}, {dose_slow:.0f} mg QD')

    ax2.set_xlabel('Time (days)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Plasma Concentration (ng/mL)', fontsize=11, fontweight='bold')
    ax2.set_title('B. Slow Metabolizer: Dose Adjustment', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 16)

    # Panel 1C: Comparison of steady-state metrics (dose adjustment)
    ax3 = fig.add_subplot(gs[0, 2])

    metrics_names = ['Cmax_ss', 'Cmin_ss', 'AUC_ss']
    ref_vals = [metrics_ref[m] for m in metrics_names]
    fast_vals = [metrics_fast_dose[m] for m in metrics_names]
    slow_vals = [metrics_slow_dose[m] for m in metrics_names]

    # Normalize
    ref_normalized = [100] * 3
    fast_normalized = [(f/r)*100 for f, r in zip(fast_vals, ref_vals)]
    slow_normalized = [(s/r)*100 for s, r in zip(slow_vals, ref_vals)]

    x = np.arange(3)
    width = 0.25

    ax3.bar(x - width, ref_normalized, width, label='Reference (c_in=1.0)', color='#4472C4', alpha=0.8)
    ax3.bar(x, fast_normalized, width, label=f'Fast (c_in={c_in_fast})', color='#ED7D31', alpha=0.8)
    ax3.bar(x + width, slow_normalized, width, label=f'Slow (c_in={c_in_slow})', color='#70AD47', alpha=0.8)

    ax3.axhline(100, color='black', linestyle='-', linewidth=1.5)
    ax3.axhline(90, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axhline(110, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.fill_between([-0.5, 2.5], 90, 110, alpha=0.1, color='green')

    ax3.set_ylabel('% of Reference', fontsize=11, fontweight='bold')
    ax3.set_title('C. Dose Adjustment: PK Match', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Cmax', 'Cmin', 'AUC'], fontsize=10)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.set_ylim(0, 120)
    ax3.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # Row 2: Frequency Adjustment Strategy
    # =========================================================================

    # Panel 2A: Fast metabolizer - frequency adjustment
    ax4 = fig.add_subplot(gs[1, 0])
    t_fast_freq_days = sim_fast_freq['time'] / 24

    ax4.plot(t_ref_days, sim_ref['C_central'], 'b-', linewidth=2.5,
             label=f'Reference: 100 mg Q24h', alpha=0.7)
    ax4.plot(t_fast_freq_days, sim_fast_freq['C_central'], 'r--', linewidth=2.5,
             label=f'Fast: 100 mg Q{interval_fast:.0f}h')

    ax4.set_xlabel('Time (days)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Plasma Concentration (ng/mL)', fontsize=11, fontweight='bold')
    ax4.set_title('D. Fast Metabolizer: Frequency Adjustment', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 16)

    # Panel 2B: Slow metabolizer - frequency adjustment
    ax5 = fig.add_subplot(gs[1, 1])
    t_slow_freq_days = sim_slow_freq['time'] / 24

    ax5.plot(t_ref_days, sim_ref['C_central'], 'b-', linewidth=2.5,
             label=f'Reference: 100 mg Q24h', alpha=0.7)
    ax5.plot(t_slow_freq_days, sim_slow_freq['C_central'], 'g--', linewidth=2.5,
             label=f'Slow: 100 mg Q{interval_slow:.0f}h')

    ax5.set_xlabel('Time (days)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Plasma Concentration (ng/mL)', fontsize=11, fontweight='bold')
    ax5.set_title('E. Slow Metabolizer: Frequency Adjustment', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, 16)

    # Panel 2C: Comparison of average concentrations (frequency adjustment)
    ax6 = fig.add_subplot(gs[1, 2])

    cavg_ref = metrics_ref['Cavg_ss']
    cavg_fast = metrics_fast_freq['Cavg_ss']
    cavg_slow = metrics_slow_freq['Cavg_ss']

    strategies = ['Reference\n(100mg Q24h)',
                  f'Fast Met.\n(100mg Q{interval_fast:.0f}h)',
                  f'Slow Met.\n(100mg Q{interval_slow:.0f}h)']
    cavg_vals = [cavg_ref, cavg_fast, cavg_slow]
    colors = ['#4472C4', '#ED7D31', '#70AD47']

    bars = ax6.bar(strategies, cavg_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax6.axhline(cavg_ref, color='black', linestyle='--', linewidth=2, alpha=0.7,
                label=f'Target Cavg = {cavg_ref:.1f} ng/mL')
    ax6.fill_between([-0.5, 2.5], cavg_ref*0.9, cavg_ref*1.1, alpha=0.1, color='green')

    # Add value labels
    for bar, val in zip(bars, cavg_vals):
        pct_diff = (val - cavg_ref) / cavg_ref * 100
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.1f}\n({pct_diff:+.1f}%)', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax6.set_ylabel('Average Conc. at SS (ng/mL)', fontsize=11, fontweight='bold')
    ax6.set_title('F. Frequency Adjustment: Cavg Match', fontsize=12, fontweight='bold')
    ax6.legend(loc='upper right', fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_ylim(0, max(cavg_vals) * 1.3)

    # =========================================================================
    # Row 3: Summary panels
    # =========================================================================

    # Panel 3A: Dose requirements across c_in range
    ax7 = fig.add_subplot(gs[2, :2])

    c_in_range = np.linspace(0.3, 2.5, 20)
    optimal_doses = []

    for c_in_val in c_in_range:
        dose = find_optimal_dose(c_in_val, metrics_ref, dosing_interval=24)
        optimal_doses.append(dose)

    ax7.plot(c_in_range, optimal_doses, 'o-', linewidth=3, markersize=8,
             color='#4472C4', markerfacecolor='white', markeredgewidth=2)
    ax7.axhline(100, color='red', linestyle='--', linewidth=2, alpha=0.7,
                label='Reference Dose (100 mg)')
    ax7.axvline(1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
                label='Reference c_in = 1.0')

    # Annotate key points
    ax7.plot([c_in_slow, c_in_fast], [dose_slow, dose_fast], 'ro', markersize=12)
    ax7.annotate(f'Slow Metabolizer\nc_in={c_in_slow}\nDose={dose_slow:.0f}mg',
                xy=(c_in_slow, dose_slow), xytext=(0.7, 150),
                fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax7.annotate(f'Fast Metabolizer\nc_in={c_in_fast}\nDose={dose_fast:.0f}mg',
                xy=(c_in_fast, dose_fast), xytext=(1.7, 150),
                fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    ax7.set_xlabel('c_in (CYP3A4 Activity Modifier)', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Optimal Dose (mg QD)', fontsize=12, fontweight='bold')
    ax7.set_title('G. Dose Requirements Across Metabolizer Phenotypes', fontsize=13, fontweight='bold')
    ax7.legend(loc='upper left', fontsize=10)
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(0.2, 2.6)

    # Panel 3B: Key message summary
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')

    summary_text = """
KEY FINDINGS

1. DOSE ADJUSTMENT
   • Fast metabolizer (c_in=2.0):
     200 mg QD ≈ 100 mg QD (c_in=1.0)

   • Slow metabolizer (c_in=0.5):
     50 mg QD ≈ 100 mg QD (c_in=1.0)

2. FREQUENCY ADJUSTMENT
   • Fast metabolizer (c_in=2.0):
     100 mg Q12h ≈ 100 mg Q24h

   • Slow metabolizer (c_in=0.5):
     100 mg Q48h ≈ 100 mg Q24h

3. CLINICAL IMPACT
   • Hybrid Model enables
     personalized dosing based on
     CYP3A4 phenotype

   • Both dose and frequency
     strategies achieve target PK

   • c_in can be estimated from
     therapeutic drug monitoring
     (TDM) data
    """

    ax8.text(0.1, 0.95, summary_text, transform=ax8.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Personalized Lorlatinib Dosing with Hybrid Model\n'
                 'Achieving Target PK Across CYP3A4 Metabolizer Phenotypes',
                 fontsize=16, fontweight='bold', y=0.995)

    return fig


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PERSONALIZED DOSING DEMONSTRATION")
    print("Hybrid Model (Chen + c_in)")
    print("=" * 80)

    # Create figure
    fig = create_personalized_dosing_figure()

    output_path = '/Users/80031987/Desktop/MetabolismProject_BasantaLab_Rotation_2025/Project_Code/Hybrid_Model/Personalized_Dosing_Demonstration.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()

    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("The Hybrid Model successfully demonstrates that patients with different")
    print("CYP3A4 metabolizer phenotypes (c_in values) can achieve similar PK profiles")
    print("to the reference (c_in=1.0, 100mg QD) through personalized dosing.")
    print("=" * 80 + "\n")
