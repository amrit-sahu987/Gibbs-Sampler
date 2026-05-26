import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'figure.autolayout': True
})

def load_and_clean_data(filename="research_data.csv"):
    df = pd.read_csv(filename)
    # Filter out the negative values caused by the curve_fit regression bug at loose tolerances
    df_clean = df[df['Tau_Mix'] > 0].copy()

    # filtering out the huge outliers that are likely due to numerical instability
    df_clean = df_clean[df_clean['Tau_Mix'] <= 60]
    return df, df_clean

def calculate_r_squared(y_true, y_pred):
    """Calculates the R^2 value to determine goodness of fit."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0: return 0
    return 1 - (ss_res / ss_tot)

def find_best_fit(x, y, independent_var_name='x'):
    models = {
        'Linear': {
            'func': lambda x, a, b: a * x + b,
            'p0': [1.0, 1.0],
            'label': "$\\tau \\propto {:.2f}" + independent_var_name + " + {:.2f}$"
        },
        'Power (Polynomial)': {
            'func': lambda x, a, b: a * (x ** b),
            'p0': [1.0, 1.0],
            'label': "$\\tau \\propto {:.2f}" + independent_var_name + "^{{{:.2f}}}$"
        },
        'Exponential': {
            'func': lambda x, a, b: a * np.exp(b * x),
            'p0': [1.0, 1.0],
            'label': "$\\tau \\propto {:.2f} e^{{{:.2f}" + independent_var_name + "}}$"
        },
        'Logarithmic': {
            'func': lambda x, a, b: a * np.log(x) + b,
            'p0': [1.0, 1.0],
            'label': "$\\tau \\propto {:.2f} \\ln(" + independent_var_name + ") + {:.2f}$"
        }
    }

    best_r2 = -float('inf')
    best_math_label = None
    best_fit_label = None
    best_fit_y = None
    best_model_name = None

    x_fit = np.linspace(min(x), max(x), 100)

    for name, model in models.items():
        try:
            popt, _ = curve_fit(model['func'], x, y, p0=model['p0'], maxfev=10000)
            y_pred = model['func'](x, *popt)
            r2 = calculate_r_squared(y, y_pred)

            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name
                best_fit_y = model['func'](x_fit, *popt)
                
                # Separate the pure math from the legend metadata
                best_math_label = model['label'].format(*popt)
                best_fit_label = f"{name} ($R^2={r2:.3f}$)"
        except Exception:
            continue 

    return x_fit, best_fit_y, best_fit_label, best_math_label, best_model_name

# def plot_finite_size_scaling(df):
#     """Plot 1: Tau vs N to prove Rapid Mixing."""
#     # Filter for low temperature and strict tolerance
#     data = df[(df['Beta'] == 1.0) & (df['Tol'] == 1e-05)].sort_values('N')
    
#     if data.empty:
#         print("Not enough data for Finite-Size Scaling plot.")
#         return

#     N_vals = data['N'].values
#     tau_vals = data['Tau_Mix'].values

#     # Fit to a polynomial scaling law: tau = c * N^alpha
#     def poly_law(N, c, alpha):
#         return c * (N ** alpha)
    
#     try:
#         popt, _ = curve_fit(poly_law, N_vals, tau_vals, p0=[1.0, 1.0])
#         c, alpha = popt
#         fit_N = np.linspace(min(N_vals), max(N_vals), 100)
#         fit_tau = poly_law(fit_N, c, alpha)
#         fit_label = f"Fit: $\\tau \\propto N^{{{alpha:.2f}}}$"
#     except:
#         fit_N, fit_tau, fit_label = [], [], None

#     plt.figure(figsize=(7, 5))
#     plt.plot(N_vals, tau_vals, 'bo', label='TDVP Data', zorder=3)
#     if fit_label:
#         plt.plot(fit_N, fit_tau, 'r--', label=fit_label, zorder=2)

#     plt.title("Finite-Size Scaling of the Gibbs Sampler")
#     plt.xlabel("System Size ($N$)")
#     plt.ylabel("Mixing Time $\\tau_{\\text{mix}}$")
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend()
#     plt.savefig("Plot1_Finite_Size_Scaling.pdf", dpi=300)
#     plt.close()
#     print("Saved Plot 1: Plot1_Finite_Size_Scaling.pdf")

# def plot_critical_slowing_down(df):
#     """Plot 2: Tau vs Beta to prove Phase Transition / Freezing."""
#     # Filter for the largest reliable system size and strict tolerance
#     data = df[(df['N'] == 6) & (df['Tol'] == 1e-05)].sort_values('Beta')
    
#     if data.empty:
#         print("Not enough data for Critical Slowing Down plot.")
#         return

#     beta_vals = data['Beta'].values
#     tau_vals = data['Tau_Mix'].values

#     # Fit to an exponential divergence law: tau = A * e^(c * beta)
#     def exp_law(beta, A, c):
#         return A * np.exp(c * beta)
    
#     try:
#         popt, _ = curve_fit(exp_law, beta_vals, tau_vals, p0=[0.1, 1.0])
#         A, c = popt
#         fit_beta = np.linspace(min(beta_vals), max(beta_vals), 100)
#         fit_tau = exp_law(fit_beta, A, c)
#         fit_label = f"Fit: $\\tau \\propto e^{{{c:.2f} \\beta}}$"
#     except:
#         fit_beta, fit_tau, fit_label = [], [], None

#     plt.figure(figsize=(7, 5))
#     # Using semi-log scale to turn the exponential into a straight line!
#     plt.semilogy(beta_vals, tau_vals, 'go', label='TDVP Data ($N=6$)', zorder=3)
#     if fit_label:
#         plt.semilogy(fit_beta, fit_tau, 'r--', label=fit_label, zorder=2)

#     plt.title("Critical Slowing Down (Temperature Dependence)")
#     plt.xlabel("Inverse Temperature ($\\beta$)")
#     plt.ylabel("Mixing Time $\\tau_{\\text{mix}}$ (Log Scale)")
#     plt.grid(True, which="both", linestyle='--', alpha=0.7)
#     plt.legend()
#     plt.savefig("Plot2_Critical_Slowing_Down.pdf", dpi=300)
#     plt.close()
#     print("Saved Plot 2: Plot2_Critical_Slowing_Down.pdf")

# def plot_algorithmic_stability(df):
#     """Plot 3: Tau vs Tolerance to prove numerical convergence."""
#     # Filter for a specific, difficult regime
#     data = df[(df['N'] == 5) & (df['Beta'] == 1.0)].sort_values('Tol', ascending=False)
    
#     if data.empty:
#         print("Not enough data for Algorithmic Stability plot.")
#         return

#     tols = data['Tol'].astype(str).values
#     tau_vals = data['Tau_Mix'].values

#     plt.figure(figsize=(7, 5))
#     colors = ['#ff9999', '#66b3ff', '#99ff99']
    
#     bars = plt.bar(tols, tau_vals, color=colors, edgecolor='black', zorder=3)
    
#     plt.title("Algorithmic Stability vs. Termination Tolerance")
#     plt.xlabel("Derivative Tolerance ($\epsilon$)")
#     plt.ylabel("Extracted Mixing Time $\\tau_{\\text{mix}}$")
#     plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
#     # Add the value labels on top of the bars
#     for bar in bars:
#         yval = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f}', ha='center', va='bottom')

#     plt.savefig("Plot3_Algorithmic_Stability.pdf", dpi=300)
#     plt.close()
#     print("Saved Plot 3: Plot3_Algorithmic_Stability.pdf")

# def plot_finite_size_scaling(df):
#     """Plot 1: Tau vs N with automated best-fit selection."""
#     data = df[(df['Beta'] == 1.0) & (df['Tol'] <= 1e-05)].sort_values('N')
#     if data.empty: return

#     N_vals = data['N'].values
#     tau_vals = data['Tau_Mix'].values

#     x_fit, y_fit, fit_label, model_name = find_best_fit(N_vals, tau_vals, independent_var_name='N')

#     plt.figure(figsize=(7, 5))
#     plt.plot(N_vals, tau_vals, 'bo', label='TDVP Data', zorder=3)
    
#     if y_fit is not None:
#         plt.plot(x_fit, y_fit, 'r--', label=fit_label, zorder=2)

#     plt.title(f"Finite-Size Scaling (Best fit: {model_name})")
#     plt.xlabel("System Size ($N$)")
#     plt.ylabel("Mixing Time $\\tau_{\\text{mix}}$")
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend()
#     plt.savefig("Plot1_Finite_Size_Scaling_AutoFit.pdf", dpi=300)
#     plt.close()
#     print("Saved: Plot1_Finite_Size_Scaling_AutoFit.pdf")

# def plot_critical_slowing_down(df):
#     """Plot 2: Tau vs Beta with automated best-fit selection."""
#     data = df[(df['N'] == 6) & (df['Tol'] <= 1e-05)].sort_values('Beta')
#     if data.empty: return

#     beta_vals = data['Beta'].values
#     tau_vals = data['Tau_Mix'].values

#     x_fit, y_fit, fit_label, model_name = find_best_fit(beta_vals, tau_vals, independent_var_name='\\beta')

#     plt.figure(figsize=(7, 5))
    
#     # We use a standard plot now, since we don't assume exponential (semilogy) anymore
#     plt.plot(beta_vals, tau_vals, 'go', label='TDVP Data ($N=6$)', zorder=3)
    
#     if y_fit is not None:
#         plt.plot(x_fit, y_fit, 'r--', label=fit_label, zorder=2)

#     plt.title(f"Temperature Dependence (Best fit: {model_name})")
#     plt.xlabel("Inverse Temperature ($\\beta$)")
#     plt.ylabel("Mixing Time $\\tau_{\\text{mix}}$")
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend()
#     plt.savefig("Plot2_Critical_Slowing_Down_AutoFit.pdf", dpi=300)
#     plt.close()
#     print("Saved: Plot2_Critical_Slowing_Down_AutoFit.pdf")

def plot_finite_size_scaling(df):
    df_filtered = df[df['Tol'] <= 1e-05]
    if df_filtered.empty: return
    
    betas = sorted(df_filtered['Beta'].unique())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(betas)))
    max_x_val = 0

    for beta, color in zip(betas, colors):
        data = df_filtered[df_filtered['Beta'] == beta].sort_values('N')
        if len(data) < 3: continue

        N_vals = data['N'].values
        tau_vals = data['Tau_Mix'].values
        max_x_val = max(max_x_val, max(N_vals))

        # Unpack the new math label
        x_fit, y_fit, fit_label, math_label, model_name = find_best_fit(N_vals, tau_vals, independent_var_name='N')

        ax.plot(N_vals, tau_vals, 'o', color=color, markersize=7)
        
        if y_fit is not None:
            ax.plot(x_fit, y_fit, '--', color=color, linewidth=1.5, label=f"$\\beta={beta}$ | {math_label}")
            
            # Annotate the exact equation at the end of the fitted line
            # ax.annotate(math_label,
            #             xy=(x_fit[-1], y_fit[-1]), 
            #             xytext=(8, 0), # Offset 8 points to the right
            #             textcoords='offset points',
            #             color=color,
            #             fontsize=10,
            #             va='center')
        else:
            ax.plot(N_vals, tau_vals, '--', color=color, linewidth=1.5, label=f"$\\beta={beta}$ (No Fit)")

    # Expand the x-axis limit by 25% to give the inline equations room to breathe
    ax.set_xlim(min(df_filtered['N']), max_x_val * 1.25)

    ax.set_title("Finite-Size Scaling Across Temperatures")
    ax.set_xlabel("System Size ($N$)")
    ax.set_ylabel("Mixing Time $\\tau_{\\text{mix}}$")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    fig.savefig("Plot1_Finite_Size_Scaling_Multi.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: Plot1_Finite_Size_Scaling_Multi.pdf")


def plot_critical_slowing_down(df):
    df_filtered = df[df['Tol'] <= 1e-05]
    if df_filtered.empty: return
    
    n_values = sorted(df_filtered['N'].unique())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(n_values)))
    max_x_val = 0

    for N, color in zip(n_values, colors):
        data = df_filtered[df_filtered['N'] == N].sort_values('Beta')
        if len(data) < 3: continue

        beta_vals = data['Beta'].values
        tau_vals = data['Tau_Mix'].values
        max_x_val = max(max_x_val, max(beta_vals))

        # Unpack the new math label
        x_fit, y_fit, fit_label, math_label, model_name = find_best_fit(beta_vals, tau_vals, independent_var_name='\\beta')

        ax.plot(beta_vals, tau_vals, 's', color=color, markersize=7)
        
        if y_fit is not None:
            ax.plot(x_fit, y_fit, '-', color=color, linewidth=1.5, alpha=0.8, label=f"$N={N}$ | {math_label}")
            
            # Annotate the exact equation at the end of the fitted line
            # ax.annotate(math_label,
            #             xy=(x_fit[-1], y_fit[-1]), 
            #             xytext=(8, 0), 
            #             textcoords='offset points',
            #             color=color,
            #             fontsize=10,
            #             va='center')
        else:
            ax.plot(beta_vals, tau_vals, '-', color=color, linewidth=1.5, alpha=0.8, label=f"$N={N}$ (No Fit)")

    # Expand the x-axis limit by 25% to make room for the text
    ax.set_xlim(min(df_filtered['Beta']), max_x_val * 1.25)

    ax.set_title("Temperature Dependence Across System Sizes")
    ax.set_xlabel("Inverse Temperature ($\\beta$)")
    ax.set_ylabel("Mixing Time $\\tau_{\\text{mix}}$")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    fig.savefig("Plot2_Critical_Slowing_Down_Multi.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved: Plot2_Critical_Slowing_Down_Multi.pdf")

def plot_3d_parameter_space(df):
    """Plot 4: 3D Surface Plot of Tau vs N and Beta."""
    # Filter for reliable tolerances only, and drop duplicates if any exist
    data = df[df['Tol'] <= 1e-05].groupby(['N', 'Beta'])['Tau_Mix'].mean().reset_index()
    
    if len(data) < 4:
        print("Not enough data points for a 3D surface plot.")
        return

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Tri-Surface plot easily handles non-uniform grid points
    surf = ax.plot_trisurf(data['N'], data['Beta'], data['Tau_Mix'], 
                           cmap='viridis', edgecolor='k', linewidth=0.2, alpha=0.9)

    # Labels and aesthetics
    ax.set_title('Phase Space of the Gibbs Sampler')
    ax.set_xlabel('System Size ($N$)')
    ax.set_ylabel('Inverse Temp ($\\beta$)')
    ax.set_zlabel('Mixing Time $\\tau_{\\text{mix}}$')
    
    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='$\\tau_{\\text{mix}}$')

    # Adjust viewing angle for best perspective
    ax.view_init(elev=25, azim=-45)

    plt.savefig("Plot4_3D_Phase_Space.pdf", dpi=300)
    plt.close()
    print("Saved: Plot4_3D_Phase_Space.pdf")

if __name__ == "__main__":
    # Load the data
    df_raw, df_clean = load_and_clean_data("research_data.csv")
    
    # Generate the physics plots (using strictly valid data)
    plot_finite_size_scaling(df_clean)
    plot_critical_slowing_down(df_clean)
    
    # Generate the engineering plot (using the raw data to show the failure modes)
    # plot_algorithmic_stability(df_raw)

    plot_3d_parameter_space(df_clean)