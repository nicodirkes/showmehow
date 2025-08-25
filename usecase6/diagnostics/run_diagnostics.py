import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import arviz as az
import seaborn as sns
from pathlib import Path

def setup_output_directory(path="diagnostics_assesment"):
    output_dir = Path(path)
    output_dir.mkdir(exist_ok=True)
    print(f"Results will be saved to: {output_dir.absolute()}")
    return output_dir

def load_mcmc_results(filename):
    print(f"Loading MCMC results from {filename}...")
    data = np.load(filename)
    trace = data['trace']
    samples = data['samples']
    lnprob = data['lnprob']
    print(f"Trace shape: {trace.shape}")
    print(f"Samples shape: {samples.shape}")
    print(f"Log prob shape: {lnprob.shape}")
    return trace, samples, lnprob

def create_inference_data(samples, lnprob, param_names):
    print("Converting to ArviZ InferenceData...")
    posterior_dict = {param: samples[:, :, i] for i, param in enumerate(param_names)}
    sample_stats_dict = {"lp": lnprob}
    inference_data = az.from_dict(posterior=posterior_dict, sample_stats=sample_stats_dict)
    print(f"Created InferenceData: {samples.shape[0]} chains, {samples.shape[1]} draws, {samples.shape[2]} parameters")
    return inference_data

def run_arviz_diagnostics(inference_data, output_dir):
    print("\n" + "="*50)
    print("ARVIZ CONVERGENCE DIAGNOSTICS")
    print("="*50)
    rhat = az.rhat(inference_data)
    print("\nR-hat (Gelman-Rubin diagnostic):")
    print("-" * 35)
    for param, value in rhat.items():
        status = "Good" if value < 1.01 else "Questionable" if value < 1.1 else "Poor"
        print(f"{param:10}: {float(value):.4f} ({status})")
    ess_bulk = az.ess(inference_data, method="bulk")
    ess_tail = az.ess(inference_data, method="tail")
    print(f"\nEffective Sample Size:")
    print("-" * 40)
    print(f"{'Parameter':<12} {'Bulk ESS':<10} {'Tail ESS':<10} {'Status'}")
    print("-" * 40)
    for param in inference_data.posterior.data_vars:
        bulk_val = float(ess_bulk[param])
        tail_val = float(ess_tail[param])
        min_ess = min(bulk_val, tail_val)
        status = "Good" if min_ess > 400 else "Low" if min_ess > 100 else "Very Low"
        print(f"{param:<12} {bulk_val:<10.0f} {tail_val:<10.0f} {status}")
    mcse = az.mcse(inference_data)
    print(f"\nMonte Carlo Standard Error:")
    print("-" * 30)
    for param, value in mcse.items():
        print(f"{param:10}: {float(value):.6f}")
    summary_df = pd.DataFrame({
        'rhat': [float(rhat[param]) for param in rhat],
        'ess_bulk': [float(ess_bulk[param]) for param in ess_bulk],
        'ess_tail': [float(ess_tail[param]) for param in ess_tail],
        'mcse': [float(mcse[param]) for param in mcse]
    }, index=list(rhat.keys()))
    summary_df.to_csv(output_dir / "convergence_diagnostics.csv")
    print(f"\nDiagnostics saved to {output_dir / 'convergence_diagnostics.csv'}")
    return rhat, ess_bulk, ess_tail, mcse

def create_arviz_plots(inference_data, noise_model, species, output_dir):
    print("\nCreating ArviZ diagnostic plots...")
    sns.set_theme(style="darkgrid", palette="deep")
    az.style.use("arviz-darkgrid")
    print("- Creating trace plots...")
    az.plot_trace(inference_data, figsize=(12, 8))
    plt.suptitle(f"{species}: {noise_model} - MCMC Trace Plots", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "trace_plots.png", dpi=150, bbox_inches='tight')
    print("- Creating posterior plots...")
    az.plot_posterior(inference_data, figsize=(12, 4), hdi_prob=0.94)
    plt.suptitle(f"{species}: {noise_model} - Posterior Distributions", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "posterior_distributions.png", dpi=150, bbox_inches='tight')
    print("- Creating pair plot...")
    az.plot_pair(inference_data, kind="scatter", marginals=True, figsize=(10, 10))
    plt.suptitle(f"{species}: {noise_model} - Parameter Correlations", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "parameter_correlations.png", dpi=150, bbox_inches='tight')
    print("- Creating autocorrelation plots...")
    az.plot_autocorr(inference_data, figsize=(12, 4))
    plt.suptitle(f"{species}: {noise_model} - Autocorrelation Functions", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "autocorrelation.png", dpi=150, bbox_inches='tight')
    print("- Creating rank plots...")
    az.plot_rank(inference_data, figsize=(12, 4))
    plt.suptitle(f"{species}: {noise_model} - Rank Plots", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "rank_plots.png", dpi=150, bbox_inches='tight')
    try:
        print("- Creating energy plot...")
        az.plot_energy(inference_data, figsize=(8, 6))
        plt.title("Energy Plot")
        plt.tight_layout()
        plt.savefig(output_dir / "energy_plot.png", dpi=150, bbox_inches='tight')

    except Exception as e:
        print(f"  Energy plot skipped: {e}")

def create_summary_table(inference_data, trace, param_names, output_dir):
    print("\nCreating summary statistics...")
    summary = az.summary(inference_data, round_to=4)
    print("\n" + "="*70)
    print("POSTERIOR SUMMARY STATISTICS")
    print("="*70)
    print(summary)
    summary.to_csv(output_dir / "posterior_summary_arviz.csv")
    df = pd.DataFrame(trace, columns=param_names)
    custom_summary = {
        param: {
            'mean': df[param].mean(),
            'std': df[param].std(),
            'min': df[param].min(),
            'max': df[param].max(),
            'q05': df[param].quantile(0.05),
            'q25': df[param].quantile(0.25),
            'q50': df[param].quantile(0.50),
            'q75': df[param].quantile(0.75),
            'q95': df[param].quantile(0.95)
        }
        for param in param_names
    }
    custom_df = pd.DataFrame(custom_summary).T
    custom_df.to_csv(output_dir / "posterior_summary_detailed.csv")
    corr_matrix = df.corr()
    corr_matrix.to_csv(output_dir / "parameter_correlations.csv")
    print(f"\nCorrelation Matrix:")
    print("-" * 20)
    print(corr_matrix.round(3))
    return summary, custom_df, corr_matrix

def save_samples(trace, param_names, output_dir):
    df_samples = pd.DataFrame(trace, columns=param_names)
    df_samples.to_csv(output_dir / "posterior_samples.csv", index=False)
    print(f"Posterior samples saved to {output_dir / 'posterior_samples.csv'}")

def assess_convergence(rhat, ess_bulk, ess_tail):
    print("\n" + "="*50)
    print("CONVERGENCE ASSESSMENT")
    print("="*50)
    max_rhat = max(float(rhat[param]) for param in rhat)
    min_ess_bulk = min(float(ess_bulk[param]) for param in ess_bulk)
    min_ess_tail = min(float(ess_tail[param]) for param in ess_tail)
    min_ess = min(min_ess_bulk, min_ess_tail)
    print(f"Max R-hat: {max_rhat:.4f}")
    print(f"Min Bulk ESS: {min_ess_bulk:.0f}")
    print(f"Min Tail ESS: {min_ess_tail:.0f}")
    if max_rhat < 1.01 and min_ess > 400:
        assessment, color, recommendation = "EXCELLENT", "✓", "Chains have converged very well. Results are reliable."
    elif max_rhat < 1.1 and min_ess > 100:
        assessment, color, recommendation = "GOOD", "✓", "Chains show good convergence. Results are trustworthy."
    elif max_rhat < 1.2 and min_ess > 50:
        assessment, color, recommendation = "ACCEPTABLE", "⚠", "Chains show reasonable convergence, but consider longer runs."
    else:
        assessment, color, recommendation = "POOR", "✗", "Convergence issues detected. Increase burn-in and sampling steps."
    print(f"\n{color} OVERALL ASSESSMENT: {assessment}")
    print(f"Recommendation: {recommendation}")
    if assessment in ["ACCEPTABLE", "POOR"]:
        print("\nSuggestions to improve:")
        print("- Increase number of steps (--nsteps)")
        print("- Increase burn-in period (--nburn)")
        print("- Use more walkers (--nwalkers)")
        print("- Check parameter identifiability")
    return assessment

def main():
    import argparse
    parser = argparse.ArgumentParser(description='MCMC Diagnostics')
    parser.add_argument('--mcmc_results', help='.npz file with mcmc_results')
    parser.add_argument('--outdir', help='ouput_directory for diagnostics assesment')
    parser.add_argument('--noise_model')
    parser.add_argument('--species')
    args = parser.parse_args()

    param_names = ["theta_0", "theta_1", "theta_2"]

    try:
        output_dir = setup_output_directory(args.outdir)
        trace, samples, lnprob = load_mcmc_results(args.mcmc_results)
        inference_data = create_inference_data(samples, lnprob, param_names)
        rhat, ess_bulk, ess_tail, mcse = run_arviz_diagnostics(inference_data, output_dir)
        create_arviz_plots(inference_data, args.noise_model, args.species, output_dir)
        create_summary_table(inference_data, trace, param_names, output_dir)
        save_samples(trace, param_names, output_dir)
        assessment = assess_convergence(rhat, ess_bulk, ess_tail)
        assessment_text = f"""
        MCMC Convergence Assessment
        ==========================
        Overall Assessment: {assessment}
        Max R-hat: {max(float(rhat[param]) for param in rhat):.4f}
        Min Bulk ESS: {min(float(ess_bulk[param]) for param in ess_bulk):.0f}
        Min Tail ESS: {min(float(ess_tail[param]) for param in ess_tail):.0f}

        Target values:
        - R-hat < 1.01 (excellent) or < 1.1 (good)
        - ESS > 400 (excellent) or > 100 (acceptable)
        """
        with open(output_dir / "assessment.txt", "w") as f:
            f.write(assessment_text)
        print(f"\nAll diagnostics complete! Results saved in: {output_dir.absolute()}")
    except FileNotFoundError:
        print("Make sure you've run the MCMC process first.")
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
