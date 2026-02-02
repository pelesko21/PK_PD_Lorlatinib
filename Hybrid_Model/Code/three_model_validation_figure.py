#!/usr/bin/env python3
"""
Three-Model Comparison: Clinical vs Chen vs Hybrid

Creates Panel D from Chen validation figure with added Hybrid model bar.

Author: Generated for MetabolismProject_BasantaLab_Rotation_2025
Date: November 2025
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Clinical Data from Bauer et al. (2021)
# =============================================================================
CLINICAL_DATA = {
    'single_dose': {
        'Cmax_mean': 695.2,    # ng/mL
        'Tmax_mean': 1.15,     # hours
        'AUC_mean': 9088,      # ng·h/mL
        'CL_F_mean': 11.01,    # L/h
    }
}


# =============================================================================
# Chen et al. Model
# =============================================================================
class ChenModel:
    """Chen et al. (2021) 2-compartment model with time-varying clearance."""

    def __init__(self):
        self.ka = 3.11
        self.CL_initial = 9.04
        self.CL_max = 14.5
        self.tau = 34.8
        self.V2 = 121.0
        self.V3 = 155.0
        self.Q = 22.0
        self.F = 0.759

    def clearance(self, t):
        induction_factor = 1 - np.exp(-t / self.tau)
        return self.CL_initial + (self.CL_max - self.CL_initial) * induction_factor

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

    def simulate_single_dose(self, dose_mg=100.0, duration_hours=168.0, n_points=1000):
        dose_ug = dose_mg * 1000 * self.F
        y0 = [dose_ug, 0.0, 0.0]
        t = np.linspace(0, duration_hours, n_points)
        solution = odeint(self.ode_system, y0, t)
        A_central = solution[:, 1]
        C_central = (A_central / self.V2) 
        return {'time': t, 'C_central': C_central}


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

    def simulate_single_dose(self, dose_mg=100.0, duration_hours=168.0, n_points=1000):
        """Simulate single dose PK profile."""
        dose_ug = dose_mg * 1000 * self.F

        y0 = [dose_ug, 0.0, 0.0]
        t = np.linspace(0, duration_hours, n_points)

        solution = odeint(self.ode_system, y0, t)

        A_central = solution[:, 1]
        C_central = (A_central / self.V2) 

        return {'time': t, 'C_central': C_central}


def calculate_pk_metrics(simulation):
    """Calculate PK metrics from simulation."""
    t = simulation['time']
    C = simulation['C_central']

    Cmax = np.max(C)
    Tmax_idx = np.argmax(C)
    Tmax = t[Tmax_idx]
    AUC = np.trapz(C, t)
    CL_F = 100 * 1000 / AUC

    return {
        'Cmax': Cmax,
        'Tmax': Tmax,
        'AUC': AUC,
        'CL_F': CL_F
    }


def create_three_model_comparison_figure():
    """
    Reproduce Panel D with three models: Clinical, Chen, and Hybrid.
    """

    # Initialize models
    chen_model = ChenModel()
    hybrid_model = HybridModel(c_in=1.0)  # Average metabolizer

    # Simulate single dose
    sim_chen = chen_model.simulate_single_dose(dose_mg=100, duration_hours=168)
    sim_hybrid = hybrid_model.simulate_single_dose(dose_mg=100, duration_hours=168)

    # Calculate metrics
    metrics_chen = calculate_pk_metrics(sim_chen)
    metrics_hybrid = calculate_pk_metrics(sim_hybrid)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Single-dose comparison (key metrics)
    metrics_names = ['Cmax\n(ng/mL)', 'AUC\n(ng·h/mL)', 'CL/F\n(L/h)', 'Tmax\n(h)']
    clinical_vals = [
        CLINICAL_DATA['single_dose']['Cmax_mean'],
        CLINICAL_DATA['single_dose']['AUC_mean'],
        CLINICAL_DATA['single_dose']['CL_F_mean'],
        CLINICAL_DATA['single_dose']['Tmax_mean']
    ]
    chen_vals = [
        metrics_chen['Cmax'],
        metrics_chen['AUC'],
        metrics_chen['CL_F'],
        metrics_chen['Tmax']
    ]
    hybrid_vals = [
        metrics_hybrid['Cmax'],
        metrics_hybrid['AUC'],
        metrics_hybrid['CL_F'],
        metrics_hybrid['Tmax']
    ]

    # Normalize to clinical values (% of clinical)
    pct_chen = [(m / c) * 100 for m, c in zip(chen_vals, clinical_vals)]
    pct_hybrid = [(m / c) * 100 for m, c in zip(hybrid_vals, clinical_vals)]

    x = np.arange(len(metrics_names))
    width = 0.25

    # Create three bars for each metric
    bars1 = ax.bar(x - width, [100]*4, width, label='Clinical (Bauer et al.)',
                   color='#FF6B6B', edgecolor='black', linewidth=1.5, alpha=0.8)
    bars2 = ax.bar(x, pct_chen, width, label='Chen et al. Model',
                   color='#4ECDC4', edgecolor='black', linewidth=1.5, alpha=0.8)
    bars3 = ax.bar(x + width, pct_hybrid, width, label='Hybrid Model (c_in=1.0)',
                   color='#95E1D3', edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add percentage labels for Chen model
    for i, (bar, pct) in enumerate(zip(bars2, pct_chen)):
        diff = pct - 100
        color = 'green' if abs(diff) < 20 else 'red'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{pct:.1f}%\n({diff:+.1f}%)', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=color)

    # Add percentage labels for Hybrid model
    for i, (bar, pct) in enumerate(zip(bars3, pct_hybrid)):
        diff = pct - 100
        color = 'green' if abs(diff) < 20 else 'red'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{pct:.1f}%\n({diff:+.1f}%)', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=color)

    # Reference lines
    ax.axhline(100, color='black', linestyle='-', linewidth=1.5, zorder=0)
    ax.axhline(80, color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=0)
    ax.axhline(120, color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=0)
    ax.fill_between([-0.5, 3.5], 80, 120, alpha=0.1, color='green', zorder=0)

    ax.set_ylabel('% of Clinical Value', fontsize=13, fontweight='bold')
    ax.set_title('D. Single-Dose PK Metrics: Model vs. Clinical', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.set_ylim(0, 140)
    ax.grid(True, alpha=0.3, axis='y', zorder=0)

    # Add ±20% acceptable range annotation
    ax.text(3.3, 100, '±20%\nacceptable', fontsize=10, ha='left', va='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # Overall title
    plt.suptitle('Chen et al. (2021) PK Model Validation Against Clinical Data\n'
                 'Lorlatinib 100 mg Oral Dose',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()

    return fig


if __name__ == "__main__":
    print("Creating Three-Model Comparison Figure (Panel D)...")
    print("=" * 70)

    # Create figure
    fig = create_three_model_comparison_figure()
    output_path = '/Users/80031987/Desktop/MetabolismProject_BasantaLab_Rotation_2025/Project_Code/Hybrid_Model/Panel_D_Three_Models.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

    # Print summary
    print()
    print("Model Performance Summary:")
    print("-" * 70)

    chen_model = ChenModel()
    hybrid_model = HybridModel(c_in=1.0)

    sim_chen = chen_model.simulate_single_dose(dose_mg=100, duration_hours=168)
    sim_hybrid = hybrid_model.simulate_single_dose(dose_mg=100, duration_hours=168)

    metrics_chen = calculate_pk_metrics(sim_chen)
    metrics_hybrid = calculate_pk_metrics(sim_hybrid)

    print(f"Single-Dose Cmax:")
    print(f"  Clinical:       {CLINICAL_DATA['single_dose']['Cmax_mean']:.1f} ng/mL")
    print(f"  Chen Model:     {metrics_chen['Cmax']:.1f} ng/mL ({(metrics_chen['Cmax']/CLINICAL_DATA['single_dose']['Cmax_mean']*100):.1f}%)")
    print(f"  Hybrid Model:   {metrics_hybrid['Cmax']:.1f} ng/mL ({(metrics_hybrid['Cmax']/CLINICAL_DATA['single_dose']['Cmax_mean']*100):.1f}%)")
    print()

    print(f"Single-Dose AUC:")
    print(f"  Clinical:       {CLINICAL_DATA['single_dose']['AUC_mean']:.1f} ng·h/mL")
    print(f"  Chen Model:     {metrics_chen['AUC']:.1f} ng·h/mL ({(metrics_chen['AUC']/CLINICAL_DATA['single_dose']['AUC_mean']*100):.1f}%)")
    print(f"  Hybrid Model:   {metrics_hybrid['AUC']:.1f} ng·h/mL ({(metrics_hybrid['AUC']/CLINICAL_DATA['single_dose']['AUC_mean']*100):.1f}%)")
    print()

    print(f"Single-Dose CL/F:")
    print(f"  Clinical:       {CLINICAL_DATA['single_dose']['CL_F_mean']:.2f} L/h")
    print(f"  Chen Model:     {metrics_chen['CL_F']:.2f} L/h ({(metrics_chen['CL_F']/CLINICAL_DATA['single_dose']['CL_F_mean']*100):.1f}%)")
    print(f"  Hybrid Model:   {metrics_hybrid['CL_F']:.2f} L/h ({(metrics_hybrid['CL_F']/CLINICAL_DATA['single_dose']['CL_F_mean']*100):.1f}%)")
    print()

    print(f"Single-Dose Tmax:")
    print(f"  Clinical:       {CLINICAL_DATA['single_dose']['Tmax_mean']:.2f} h")
    print(f"  Chen Model:     {metrics_chen['Tmax']:.2f} h ({(metrics_chen['Tmax']/CLINICAL_DATA['single_dose']['Tmax_mean']*100):.1f}%)")
    print(f"  Hybrid Model:   {metrics_hybrid['Tmax']:.2f} h ({(metrics_hybrid['Tmax']/CLINICAL_DATA['single_dose']['Tmax_mean']*100):.1f}%)")
    print()

    print("=" * 70)
    print("Note: Hybrid Model with c_in=1.0 (average metabolizer) performs")
    print("      identically to Chen model. The c_in parameter enables")
    print("      personalization for individual patient variability.")
    print("=" * 70)
