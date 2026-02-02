# Population Variability Enhancement Guide
## Improving Hybrid Model Population Validation

**Document Purpose:** Reference guide for enhancing population-level variability in the Hybrid PK Model to match clinical observations

**Date Created:** November 2025
**Location:** `/Users/80031987/Desktop/MetabolismProject_BasantaLab_Rotation_2025/Project_Code/Hybrid_Model/`

---

## 1. Current Limitation

### Problem Statement

The current population validation (Section 8.4 of `hybrid_pk_model.ipynb`) shows:

| Metric | Clinical CV | Simulated CV | Status |
|--------|-------------|--------------|--------|
| Cmax   | 40.0%       | 0.7%         | ✗ FAIL |
| AUC    | 35.0%       | 15.4%        | ✗ FAIL |
| CL/F   | 35.0%       | 15.5%        | ✓ PASS |

**Root Cause:** Only clearance variability (via c_in, CV=17.2%) is modeled. All other PK parameters (ka, V2, V3, Q, F) are fixed across the population.

### Why This Matters

1. **Unrealistic population predictions:** Model cannot capture the full range of observed patient responses
2. **Poor Visual Predictive Check (VPC) performance:** Prediction intervals too narrow
3. **Limited clinical utility:** Cannot properly assess variability in drug exposure
4. **Invalid dose optimization:** Personalized dosing recommendations may be overly confident

---

## 2. Solution Approaches

### Overview Table

| Solution | Complexity | Rigor | Data Required | Implementation Time |
|----------|------------|-------|---------------|---------------------|
| 1. Multi-parameter IIV | Moderate | High | IIV estimates | 2-3 hours |
| 2. Full Chen IIV structure | High | Very High | Chen omega matrix | 4-6 hours |
| 3. Residual error model | Low | Moderate | RUV estimates | 1 hour |
| 4. Empirical c_in scaling | Low | Low | None | 30 minutes |

---

## 3. SOLUTION 1: Multi-Parameter IIV (RECOMMENDED)

### Rationale

Add inter-individual variability to key PK parameters that directly influence Cmax and AUC:
- **ka** (absorption rate): affects Cmax and Tmax
- **V2** (central volume): directly affects Cmax (Cmax ∝ Dose/V2)
- **F** (bioavailability): affects both Cmax and AUC
- **c_in** (CL modifier): already implemented

### Required Data

IIV estimates needed from Chen et al. (2021) or typical values for oral oncology drugs:

| Parameter | Typical CV | Source/Rationale |
|-----------|------------|------------------|
| ka        | 25-40%     | High variability in oral absorption |
| V2        | 20-30%     | Body size, tissue binding differences |
| V3        | 30-40%     | Peripheral distribution variability |
| Q         | 30-50%     | Inter-compartmental transfer |
| F         | 15-25%     | First-pass metabolism, food effects |
| CL        | 17.2%      | Chen et al. 2021 (already used) |

### Implementation Code

```python
# =========================================================================
# Enhanced Population Simulation with Multi-Parameter IIV
# =========================================================================

import numpy as np
from scipy.integrate import odeint

# IIV definitions (CV%)
IIV_PARAMS = {
    'ka': 0.30,    # 30% CV - absorption variability
    'V2': 0.25,    # 25% CV - central volume variability
    'V3': 0.35,    # 35% CV - peripheral volume variability
    'Q': 0.40,     # 40% CV - inter-compartmental clearance
    'F': 0.20,     # 20% CV - bioavailability variability
    'CL': 0.172    # 17.2% CV - clearance (from Chen et al.)
}

# Reference (population mean) values
REF_PARAMS = {
    'ka': 3.11,
    'V2': 121.0,
    'V3': 155.0,
    'Q': 22.0,
    'F': 0.759,
    'CL_initial': 9.04,
    'CL_max': 14.5,
    'tau': 34.8
}

def generate_population_parameters(n_patients, seed=123, correlations=None):
    """
    Generate individual PK parameters with lognormal IIV.

    Parameters:
    -----------
    n_patients : int
        Number of virtual patients
    seed : int
        Random seed for reproducibility
    correlations : dict, optional
        Parameter correlations, e.g., {'CL-V2': 0.5}

    Returns:
    --------
    params_dict : dict
        Dictionary with arrays of individual parameters
    """
    np.random.seed(seed)

    params_dict = {}

    # Generate independent lognormal variability for each parameter
    for param, cv in IIV_PARAMS.items():
        sigma_log = np.sqrt(np.log(1 + cv**2))
        mu_log = -0.5 * sigma_log**2  # Ensures geometric mean = 1.0

        # Draw from lognormal distribution
        eta = np.random.lognormal(mu_log, sigma_log, n_patients)

        # Scale by reference value (except CL which is handled via c_in)
        if param == 'CL':
            params_dict['c_in'] = eta  # CL modifier
        else:
            params_dict[param] = REF_PARAMS[param] * eta

    # Optional: Add correlations (if known from Chen et al.)
    if correlations is not None:
        # Example: Positive CL-V2 correlation (typical in PK)
        if 'CL-V2' in correlations:
            rho = correlations['CL-V2']
            # Adjust V2 based on c_in
            V2_adjustment = 1 + rho * (params_dict['c_in'] - 1)
            params_dict['V2'] *= V2_adjustment

    return params_dict


def simulate_population_pk(params_dict, dose_mg=100, duration_hours=168):
    """
    Simulate PK for entire population with individual parameters.

    Parameters:
    -----------
    params_dict : dict
        Individual parameters from generate_population_parameters()
    dose_mg : float
        Dose in mg
    duration_hours : float
        Simulation duration

    Returns:
    --------
    results : dict
        PK metrics for each patient
    """
    n_patients = len(params_dict['c_in'])

    results = {
        'Cmax': np.zeros(n_patients),
        'Tmax': np.zeros(n_patients),
        'AUC': np.zeros(n_patients),
        'CL_F': np.zeros(n_patients)
    }

    for i in range(n_patients):
        # Create individual model
        model = HybridPKModel(
            c_in=params_dict['c_in'][i],
            ka=params_dict['ka'][i],
            V2=params_dict['V2'][i],
            V3=params_dict['V3'][i],
            Q=params_dict['Q'][i],
            F=params_dict['F'][i],
            calibration_factor=1.35
        )

        # Simulate
        sim = model.simulate_single_dose(dose_mg=dose_mg, duration_hours=duration_hours)

        # Calculate metrics
        results['Cmax'][i] = np.max(sim['C_central'])
        results['Tmax'][i] = sim['time'][np.argmax(sim['C_central'])]
        results['AUC'][i] = np.trapezoid(sim['C_central'], sim['time'])
        results['CL_F'][i] = dose_mg * 1000 / results['AUC'][i]

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{n_patients} patients")

    return results


# =========================================================================
# USAGE EXAMPLE
# =========================================================================

# Generate 329 patients (matching Bauer et al. cohort size)
params = generate_population_parameters(n_patients=329, seed=123)

# Print population statistics
print("Population Parameter Distributions:")
print("=" * 70)
for param in ['c_in', 'ka', 'V2', 'V3', 'Q', 'F']:
    values = params[param]
    cv = np.std(values) / np.mean(values) * 100
    print(f"{param:8s}: Mean={np.mean(values):7.3f}, CV={cv:5.1f}%, "
          f"Range=[{np.min(values):.3f}, {np.max(values):.3f}]")

# Simulate population PK
print("\nSimulating population PK...")
results = simulate_population_pk(params, dose_mg=100, duration_hours=168)

# Validate against clinical data
print("\nValidation Against Clinical Data:")
print("=" * 70)
clinical_targets = {
    'Cmax': {'mean': 695.2, 'cv': 40.0},
    'AUC': {'mean': 9088.0, 'cv': 35.0},
    'CL_F': {'mean': 11.01, 'cv': 35.0}
}

for metric, target in clinical_targets.items():
    sim_mean = np.mean(results[metric])
    sim_cv = np.std(results[metric]) / sim_mean * 100

    mean_error = (sim_mean - target['mean']) / target['mean'] * 100
    cv_error = sim_cv - target['cv']

    print(f"{metric}:")
    print(f"  Clinical: {target['mean']:.1f} (CV={target['cv']:.1f}%)")
    print(f"  Simulated: {sim_mean:.1f} (CV={sim_cv:.1f}%)")
    print(f"  Mean error: {mean_error:+.1f}%")
    print(f"  CV difference: {cv_error:+.1f}%")
    print()
```

### Expected Outcomes

With typical IIV values:
- **Cmax CV:** Should increase from 0.7% → 30-35% (closer to clinical 40%)
- **AUC CV:** Should increase from 15.4% → 30-35% (matches clinical 35%)
- **Better VPC performance:** Wider prediction intervals
- **More realistic population:** Captures inter-patient heterogeneity

### Pros/Cons

**Pros:**
- Mechanistically sound
- Can be calibrated to clinical data
- Maintains interpretability of c_in as CL modifier
- Enables parameter-specific personalization

**Cons:**
- Requires IIV estimates for all parameters
- May need correlation structure (more complex)
- Slightly longer computation time

---

## 4. SOLUTION 2: Full Chen et al. IIV Structure with Correlations

### Rationale

Use the complete variance-covariance (omega) matrix from Chen et al. (2021) population PK analysis. This captures:
- Individual parameter variabilities
- **Correlations between parameters** (e.g., CL-V2 correlation is common)
- Exactly matches the validated Chen model

### Required Data

From Chen et al. (2021) Table/Supplementary Materials:
1. Omega (Ω) matrix: variance-covariance of random effects
2. Or at minimum: Individual %CV and correlation coefficients

**Example structure (hypothetical):**
```
       CL     V2     V3      Q     ka
CL   [0.030  0.015  0.005   0     0   ]
V2   [0.015  0.062  0.010   0     0   ]
V3   [0.005  0.010  0.120   0     0   ]
Q    [0      0      0      0.16   0   ]
ka   [0      0      0       0    0.09 ]
```

### Implementation Code

```python
import numpy as np
from scipy.stats import multivariate_normal

def generate_correlated_population(n_patients, omega_matrix, seed=123):
    """
    Generate population with correlated PK parameters.

    Parameters:
    -----------
    n_patients : int
        Number of virtual patients
    omega_matrix : np.ndarray
        Variance-covariance matrix of log-transformed parameters
    seed : int
        Random seed

    Returns:
    --------
    params_dict : dict
        Individual PK parameters
    """
    np.random.seed(seed)

    # Parameter order (must match omega_matrix)
    param_names = ['CL', 'V2', 'V3', 'Q', 'ka']
    ref_values = [1.0, 121.0, 155.0, 22.0, 3.11]  # CL as c_in=1.0

    # Sample from multivariate normal in log-space
    # Assumes log(param) ~ MVN(0, omega_matrix)
    eta = multivariate_normal.rvs(
        mean=np.zeros(len(param_names)),
        cov=omega_matrix,
        size=n_patients
    )

    # Transform to parameters
    params_dict = {}
    for i, (name, ref) in enumerate(zip(param_names, ref_values)):
        if name == 'CL':
            params_dict['c_in'] = np.exp(eta[:, i])
        else:
            params_dict[name.lower()] = ref * np.exp(eta[:, i])

    return params_dict


# =========================================================================
# EXAMPLE USAGE (with hypothetical omega matrix)
# =========================================================================

# Example omega matrix (variance-covariance on log scale)
# These are EXAMPLE values - replace with actual Chen et al. estimates
omega = np.array([
    [0.0296, 0.0150, 0.0050, 0.0000, 0.0000],  # CL
    [0.0150, 0.0625, 0.0100, 0.0000, 0.0000],  # V2
    [0.0050, 0.0100, 0.1225, 0.0000, 0.0000],  # V3
    [0.0000, 0.0000, 0.0000, 0.1600, 0.0000],  # Q
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0900]   # ka
])

# Generate population
params = generate_correlated_population(n_patients=329, omega_matrix=omega, seed=123)

# Check correlation
cl_log = np.log(params['c_in'])
v2_log = np.log(params['v2'] / 121.0)
correlation = np.corrcoef(cl_log, v2_log)[0, 1]
print(f"CL-V2 correlation: {correlation:.3f}")

# Simulate as before
results = simulate_population_pk(params, dose_mg=100, duration_hours=168)
```

### Data Sources

**Where to find Chen et al. omega matrix:**
1. Main paper Table 3 or 4 (parameter estimates)
2. Supplementary Materials
3. Contact authors if not published
4. Use NONMEM/Monolix output files if available

**Alternative if omega unavailable:**
- Use literature values for similar drugs (lorlatinib class)
- Tyrosine kinase inhibitors (TKIs) PK variability database
- FDA Clinical Pharmacology reviews

### Pros/Cons

**Pros:**
- Highest rigor and scientific validity
- Captures all parameter correlations
- Exactly matches validated Chen model structure
- Publishable quality

**Cons:**
- Requires access to full Chen et al. data
- More complex implementation
- Harder to debug if issues arise
- May need specialized knowledge of PopPK

---

## 5. SOLUTION 3: Residual Unexplained Variability (RUV)

### Rationale

Even with all IIV modeled, clinical observations have additional variability from:
- Measurement/assay error
- Model misspecification
- Unmodeled patient factors (food, adherence, drug-drug interactions)

Standard PopPK models include both IIV and RUV.

### Implementation Code

```python
def add_residual_error(pk_metric, proportional_cv=0.15, additive_sd=10):
    """
    Add residual unexplained variability to PK metric.

    Parameters:
    -----------
    pk_metric : np.ndarray
        True simulated PK values (e.g., Cmax)
    proportional_cv : float
        Proportional error as CV (default 15%)
    additive_sd : float
        Additive error standard deviation in same units as pk_metric

    Returns:
    --------
    observed_values : np.ndarray
        PK values with residual error
    """
    n = len(pk_metric)

    # Draw random error
    epsilon_prop = np.random.normal(0, 1, n)
    epsilon_add = np.random.normal(0, 1, n)

    # Combined error model: Y_obs = Y_true * (1 + CV*ε₁) + SD*ε₂
    observed_values = (
        pk_metric * (1 + proportional_cv * epsilon_prop) +
        additive_sd * epsilon_add
    )

    # Ensure no negative concentrations
    observed_values = np.maximum(observed_values, 0)

    return observed_values


# =========================================================================
# USAGE: Add RUV to simulation results
# =========================================================================

# After simulating population (from Solution 1 or 2)
results_true = simulate_population_pk(params, dose_mg=100)

# Add residual error
results_observed = {
    'Cmax': add_residual_error(results_true['Cmax'],
                                proportional_cv=0.15, additive_sd=10),
    'AUC': add_residual_error(results_true['AUC'],
                               proportional_cv=0.10, additive_sd=100),
    'CL_F': results_true['CL_F']  # Derived, not directly observed
}

# Compare CVs
print("Effect of Residual Error:")
print("=" * 70)
for metric in ['Cmax', 'AUC']:
    cv_true = np.std(results_true[metric]) / np.mean(results_true[metric]) * 100
    cv_obs = np.std(results_observed[metric]) / np.mean(results_observed[metric]) * 100
    print(f"{metric}:")
    print(f"  True (IIV only): CV = {cv_true:.1f}%")
    print(f"  Observed (IIV + RUV): CV = {cv_obs:.1f}%")
    print(f"  Increase: +{cv_obs - cv_true:.1f}%")
    print()
```

### Typical RUV Values

From PopPK literature:
- **Proportional error:** 10-20% CV (for plasma concentrations)
- **Additive error:** ~5-20 ng/mL for Cmax-scale measurements
- **Model:** Usually combined proportional + additive

### When to Use RUV

- **Always** in combination with IIV (Solutions 1 or 2)
- When validating against individual patient data points
- For Visual Predictive Checks (VPCs)
- When simulating TDM scenarios

### Pros/Cons

**Pros:**
- Captures realistic measurement uncertainty
- Standard practice in PopPK
- Small code addition
- Improves realism

**Cons:**
- Requires RUV estimates from literature or Chen model
- Adds "noise" that may obscure patterns
- Not interpretable as patient variability

---

## 6. SOLUTION 4: Empirical c_in Scaling (Quick Fix)

### Rationale

**Pragmatic shortcut:** Scale c_in distribution to empirically match clinical CV, treating c_in as a "total variability factor" rather than just CL variability.

**WARNING:** Less mechanistically meaningful. Use only for preliminary work or when other solutions not feasible.

### Implementation Code

```python
# =========================================================================
# Empirical c_in Scaling
# =========================================================================

def generate_empirical_cin(n_patients, target_auc_cv=0.35, seed=123):
    """
    Generate c_in with inflated CV to match clinical variability.

    Parameters:
    -----------
    n_patients : int
        Number of patients
    target_auc_cv : float
        Target AUC CV (e.g., 0.35 for 35%)
    seed : int
        Random seed

    Returns:
    --------
    cin_population : np.ndarray
        c_in values with inflated variability
    """
    np.random.seed(seed)

    # For AUC ∝ 1/CL ∝ 1/c_in, we need CV(c_in) ≈ CV(AUC)
    # Use target AUC CV as c_in CV
    sigma_log = np.sqrt(np.log(1 + target_auc_cv**2))
    mu_log = -0.5 * sigma_log**2

    cin_population = np.random.lognormal(mu_log, sigma_log, n_patients)

    return cin_population


# Generate population
cin_empirical = generate_empirical_cin(n_patients=329, target_auc_cv=0.35, seed=123)

print(f"Empirical c_in distribution:")
print(f"  Mean: {np.mean(cin_empirical):.3f}")
print(f"  CV: {np.std(cin_empirical)/np.mean(cin_empirical)*100:.1f}%")
print(f"  Range: [{np.min(cin_empirical):.3f}, {np.max(cin_empirical):.3f}]")

# Use in standard simulation
results = []
for cin in cin_empirical:
    model = HybridPKModel(c_in=cin, calibration_factor=1.35)
    sim = model.simulate_single_dose(dose_mg=100)
    results.append({
        'Cmax': np.max(sim['C_central']),
        'AUC': np.trapz(sim['C_central'], sim['time'])
    })

# Convert to arrays
Cmax = np.array([r['Cmax'] for r in results])
AUC = np.array([r['AUC'] for r in results])

print(f"\nSimulated Cmax CV: {np.std(Cmax)/np.mean(Cmax)*100:.1f}%")
print(f"Simulated AUC CV: {np.std(AUC)/np.mean(AUC)*100:.1f}%")
```

### Expected Outcomes

- AUC CV will approximately match target (35%)
- Cmax CV will remain low (still only ~CV dependent on CL)
- **Does NOT solve Cmax variability problem**

### Pros/Cons

**Pros:**
- Extremely simple (5-line code change)
- No additional data required
- Quick validation for AUC metrics

**Cons:**
- Not mechanistically sound
- c_in loses interpretability as CYP3A4 activity
- Cannot explain Cmax variability
- Not suitable for publication
- Misleading for personalization applications

---

## 7. Implementation Recommendations

### Recommended Path Forward

**Phase 1: Quick Assessment (1-2 hours)**
1. Implement Solution 3 (RUV) with typical values
2. Check if this alone brings CV into acceptable range
3. Document gap remaining

**Phase 2: Multi-Parameter IIV (2-4 hours)**
1. Implement Solution 1 with typical IIV values from literature
2. Validate against clinical data
3. Perform sensitivity analysis on IIV assumptions

**Phase 3: Full Validation (if needed, 4-8 hours)**
1. Obtain Chen et al. full omega matrix
2. Implement Solution 2 with correlated parameters
3. Generate publication-quality VPCs
4. Compare all three approaches (current, Solution 1, Solution 2)

### Code Integration Strategy

**Step 1:** Modify `HybridPKModel` class to accept individual parameters
```python
class HybridPKModel:
    def __init__(self, c_in=1.0, ka=None, V2=None, V3=None, Q=None, F=None,
                 calibration_factor=1.35):
        # Use defaults if not provided
        self.ka = ka if ka is not None else 3.11
        self.V2 = V2 if V2 is not None else 121.0
        self.V3 = V3 if V3 is not None else 155.0
        # ... etc
```

**Step 2:** Update population validation section (8.4)
```python
# Replace lines 438-447 with multi-parameter generation
params = generate_population_parameters(n_patients=329, seed=123)

# Replace lines 466-468 with individual parameter simulation
for i in range(n_patients):
    model = HybridPKModel(
        c_in=params['c_in'][i],
        ka=params['ka'][i],
        V2=params['V2'][i],
        # ... etc
    )
```

**Step 3:** Add RUV before calculating statistics
```python
# After line 480, before calculating CVs
results_observed = {
    'Cmax': add_residual_error(results['Cmax'], proportional_cv=0.15),
    'AUC': add_residual_error(results['AUC'], proportional_cv=0.10)
}
```

---

## 8. Validation Criteria

### Acceptance Criteria for Enhanced Model

| Metric | Clinical | Target Range | Pass/Fail |
|--------|----------|--------------|-----------|
| Cmax Mean | 695.2 ng/mL | 625-765 ng/mL (±10%) | Mean within ±10% |
| Cmax CV | 40% | 35-45% | CV within ±5% |
| AUC Mean | 9088 ng·h/mL | 8170-9996 (±10%) | Mean within ±10% |
| AUC CV | 35% | 30-40% | CV within ±5% |
| CL/F Mean | 11.01 L/h | 9.9-12.1 (±10%) | Mean within ±10% |
| CL/F CV | 35% | 30-40% | CV within ±5% |

### Visual Checks

1. **Histogram overlap:** Simulated vs clinical distributions
2. **Q-Q plots:** Check lognormality assumption
3. **VPC plots:** 5th, 50th, 95th percentiles match clinical ranges
4. **Scatter plots:** Individual Cmax vs AUC correlation

---

## 9. Literature References

### Key Papers for IIV Values

1. **Chen et al. (2021)** - Lorlatinib PopPK analysis (primary source)
2. **Bauer et al. (2021)** - Clinical PK data and variability estimates
3. **FDA Clinical Pharmacology Review** - Lorlatinib NDA 210868
4. **Typical TKI variability:**
   - Widmer et al. (2014) - Imatinib PopPK: CV(CL)=32%, CV(V)=54%
   - Gotta et al. (2013) - Erlotinib: CV(CL)=42%

### Standard PopPK References

- Mould & Upton (2013) "Basic concepts in population modeling, simulation, and model-based drug development"
- Savic & Karlsson (2009) "Importance of shrinkage in empirical Bayes estimates for diagnostics"
- Bergstrand et al. (2011) "Prediction-corrected VPC"

---

## 10. Next Steps Checklist

When ready to implement:

- [ ] Decide on solution approach (1, 2, 3, or combination)
- [ ] Gather required IIV estimates from literature
- [ ] Back up current `hybrid_pk_model.ipynb`
- [ ] Modify `HybridPKModel` class to accept individual parameters
- [ ] Implement population parameter generation function
- [ ] Update section 8.4 simulation loop
- [ ] Add residual error model (if using Solution 3)
- [ ] Re-run validation and check acceptance criteria
- [ ] Generate updated figures (histograms, VPCs)
- [ ] Document changes and rationale
- [ ] Compare old vs new validation results side-by-side

---

## 11. Code Snippets for Quick Reference

### Complete Working Example (Solution 1)

See `/Users/80031987/Desktop/MetabolismProject_BasantaLab_Rotation_2025/Project_Code/Hybrid_Model/enhanced_population_validation.py` (to be created when implementing)

### Key Functions

```python
# Core functions to implement
generate_population_parameters(n_patients, seed)  # Solution 1
generate_correlated_population(n_patients, omega_matrix, seed)  # Solution 2
add_residual_error(pk_metric, proportional_cv, additive_sd)  # Solution 3
simulate_population_pk(params_dict, dose_mg, duration_hours)  # Common
```

---

## 12. Contact & Support

**Questions or Issues:**
- Review Section 8.4 of `hybrid_pk_model.ipynb` for current implementation
- Check Chen et al. (2021) paper for IIV estimates
- Consult FDA Lorlatinib Clinical Pharmacology Review

**Future Enhancements:**
- Bayesian c_in estimation with IIV
- Patient-specific parameter estimation from sparse TDM data
- Covariate analysis (age, weight, renal function on PK)

---

**End of Document**

*This guide provides a comprehensive roadmap for enhancing population variability in the Hybrid PK Model. Choose the solution that best balances rigor, data availability, and implementation timeline for your specific needs.*
