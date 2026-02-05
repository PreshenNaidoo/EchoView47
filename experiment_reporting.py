"""Plotting and reporting helpers for noise/temperature experiment sweeps."""

import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from experiment_discovery import discover_experiments

def _load_acc_for_one_run(
    exp_abs_path: str,
    subfolder: str,
    eval_folder: str,
    noise_levels,
    folder_template="ap{0}_pl{0}_ps{0}_do{0}_mm{0}",
    results_file="results_0.json",
):
    """
    Returns list of accuracies aligned to noise_levels (may include None if missing).
    """
    accs = []
    for noise in noise_levels:
        noise_folder = folder_template.format(noise)
        p = os.path.join(exp_abs_path, noise_folder, subfolder, eval_folder, results_file)
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    d = json.load(f)
                accs.append(d.get("acc", None))
            except Exception:
                accs.append(None)
        else:
            accs.append(None)
    return accs

def _mean_std_over_runs(acc_lists):
    """
    acc_lists: list of lists, each inner list length = len(noise_levels)
    returns (means, stds) with NaN when all missing
    """
    A = np.array(acc_lists, dtype=float)  # None -> nan
    means = np.nanmean(A, axis=0)
    stds  = np.nanstd(A, axis=0, ddof=0)
    # if all-nan for a column, nanmean gives nan (fine)
    return means.tolist(), stds.tolist()

def plot_accuracy_by_noise(
    save_dir: str,
    name: str,
    exp_df: pd.DataFrame,
    title: str = None,
    noise_levels=(0, 10, 20, 30, 40, 50),
    subfolder="downstream_with_pretrained_model_0",
    eval_folders=(
        "eval test files",
        "eval all experts agree",
        "eval majority Vote",
        "eval match any expert",
    ),
    # --- OLD-FILE DEFAULTS ---
    label_fontsize=22,
    tick_fontsize=20,
    lineweight=3,
    marker_size=10,
    colours_temps=None,
    # If you want to fully mimic old plots (no error bars), set show_errorbars=False
    show_errorbars=True,
):
    """
    Old-style aesthetics, newer-style aggregation (mean +/- std over runs).
    Produces one figure per eval folder per (arch, method_variant):
      {save_dir}/{name}_{arch}_{method}_{safe_eval}.png
    """
    os.makedirs(save_dir, exist_ok=True)

    if exp_df.empty:
        print("[WARN] exp_df is empty; nothing to plot.")
        return

    if colours_temps is None:
        colours_temps = {
            "0.1": "goldenrod",
            "0.2": "cadetblue",
            "0.3": "indianred",
            "0.35": "skyblue",
            "0.4": "mediumpurple",
            "0.5": "yellowgreen",
        }

    def temp_to_color(t):
        key = str(t).rstrip("0").rstrip(".")  # 0.30 -> "0.3"
        return colours_temps.get(key, None)

    group_keys = ["arch", "method_variant"]

    for eval_folder in eval_folders:
        safe_eval = eval_folder.replace(" ", "_").lower()

        for (arch, method_variant), g in exp_df.groupby(group_keys):
            fig, ax = plt.subplots(figsize=(8, 8))  # OLD: (8,8)

            for temp, gt in g.groupby("temp"):
                run_curves = []
                for _, row in gt.iterrows():
                    run_curve = _load_acc_for_one_run(
                        exp_abs_path=row["abs_path"],
                        subfolder=subfolder,
                        eval_folder=eval_folder,
                        noise_levels=noise_levels,
                    )
                    run_curves.append(run_curve)

                means, stds = _mean_std_over_runs(run_curves)

                # skip if all nan
                if all((m is None) or (isinstance(m, float) and math.isnan(m)) for m in means):
                    continue

                label = rf"$\tau = {temp:g}$"
                c = temp_to_color(temp)

                x = list(noise_levels)
                y = means

                if show_errorbars:
                    ax.errorbar(
                        x,
                        y,
                        yerr=stds,
                        marker="o",
                        markersize=marker_size,
                        linewidth=lineweight,
                        label=label,
                        color=c,
                        # Old-style: no visible caps; set to 0 to stay visually close
                        capsize=0,
                        elinewidth=lineweight,
                    )
                else:
                    ax.plot(
                        x,
                        y,
                        marker="o",
                        markersize=marker_size,
                        linewidth=lineweight,
                        label=label,
                        color=c,
                    )

            # skip empty plots
            if len(ax.lines) == 0:
                plt.close(fig)
                continue

            # OLD file: title was just `title` (if provided), not arch/method by default
            plot_title = title if title else None
            _apply_old_plot_style(
                ax,
                title=plot_title,
                xlabel="Noise Percentage",
                ylabel="Accuracy",
                label_fontsize=label_fontsize,
                tick_fontsize=tick_fontsize,
            )

            fig.tight_layout()
            out = os.path.join(save_dir, f"{name}_{arch}_{method_variant}_{safe_eval}.png")
            fig.savefig(out, dpi=300)  # OLD: dpi=300
            plt.close(fig)
            print(f"[SAVED] {out}")

def load_baseline_mean_std(
    exp_df: pd.DataFrame,
    arch: str,
    baseline_subfolder: str,  # "downstream_with_imagenet" or "downstream_with_random_init"
    noise_levels=(0, 10, 20, 30, 40, 50),
    eval_folder="eval test files",
    results_file="results_0.json",
):
    """
    Baselines are stored only under T01 folders (tau=0.1), and don't depend on loss.
    We'll pick one loss to avoid duplicates (prefer logsum if present).
    Computes mean+/-std over runs for the baseline at each noise level.
    """

    # filter to arch + tau=0.1 (T01) folders
    g = exp_df[(exp_df["arch"] == arch) & (np.isclose(exp_df["temp"], 0.1))].copy()
    if g.empty:
        raise RuntimeError(f"No T01 folders found for arch={arch}")

    # prefer logsum folders if they exist, else just take what's there
    if "loss" in g.columns and (g["loss"] == "logsum").any():
        g = g[g["loss"] == "logsum"]

    # keep one folder per run (run 0/1), because loss variants would duplicate
    g = g.sort_values(["run", "folder"]).drop_duplicates(subset=["run"], keep="first")

    run_curves = []
    for _, row in g.iterrows():
        curve = _load_acc_for_one_run(
            exp_abs_path=row["abs_path"],
            subfolder=baseline_subfolder,
            eval_folder=eval_folder,
            noise_levels=noise_levels,
            results_file=results_file,
        )
        run_curves.append(curve)

    means, stds = _mean_std_over_runs(run_curves)
    return means, stds

def create_best_csv(
    save_dir: str,
    exp_df: pd.DataFrame,
    noise_levels=(0, 10, 20, 30, 40, 50),
    subfolder="downstream_with_pretrained_model_0",
    eval_folders=(
        "eval test files",
        "eval all experts agree",
        "eval majority Vote",
        "eval match any expert",
    ),
    folder_template="ap{0}_pl{0}_ps{0}_do{0}_mm{0}",
    results_file="results_0.json",
):
    """
    For each (arch, method_variant, eval_folder):
      - choose the BEST temperature at each noise level by maximising
        mean accuracy on "eval test files" (selection criterion),
      - then report mean + std (over runs) for that chosen temperature
        for EACH eval_folder.

    Outputs CSV + a plot with error bars for each eval_folder:
      grouped_best_accuracy_{arch}_{safe_eval}.csv
      grouped_best_accuracy_{arch}_{safe_eval}.png

    Note: selection uses "eval test files" only, like your original code.
    """
    os.makedirs(save_dir, exist_ok=True)

    if exp_df.empty:
        print("[WARN] exp_df is empty; nothing to summarize.")
        return

    # Precompute all accuracy curves for speed:
    # key: (folder, eval_folder) -> acc_list
    acc_cache = {}
    def get_curve(folder_abs, eval_folder):
        key = (folder_abs, eval_folder)
        if key not in acc_cache:
            acc_cache[key] = _load_acc_for_one_run(
                exp_abs_path=folder_abs,
                subfolder=subfolder,
                eval_folder=eval_folder,
                noise_levels=noise_levels,
                folder_template=folder_template,
                results_file=results_file,
            )
        return acc_cache[key]

    # For each arch & method_variant, find the best temp PER noise based on eval test files
    for arch, g_arch in exp_df.groupby("arch"):
        for eval_folder in eval_folders:
            safe_eval = eval_folder.replace(" ", "_").lower()
            rows = []

            # --- Baselines (per arch) ---
            rand_means, rand_stds = load_baseline_mean_std(
                exp_df=exp_df,
                arch=arch,
                baseline_subfolder="downstream_with_random_init",
                noise_levels=noise_levels,
                eval_folder=eval_folder,
            )

            img_means, img_stds = load_baseline_mean_std(
                exp_df=exp_df,
                arch=arch,
                baseline_subfolder="downstream_with_imagenet",
                noise_levels=noise_levels,
                eval_folder=eval_folder,
            )

            # Append baseline rows into df_out-like structure (no chosen tau)
            rows.insert(0, {  # insert at top if you want baselines first
                "Arch": arch,
                "Method": "Rand Init",
                "ChosenTemps": [None] * len(noise_levels),
                **{f"Noise_{n}_mean": rand_means[i] for i, n in enumerate(noise_levels)},
                **{f"Noise_{n}_std": rand_stds[i] for i, n in enumerate(noise_levels)},
                **{f"Noise_{n}_chosen_tau": None for n in noise_levels},
            })

            rows.insert(1, {
                "Arch": arch,
                "Method": "ImageNet",
                "ChosenTemps": [None] * len(noise_levels),
                **{f"Noise_{n}_mean": img_means[i] for i, n in enumerate(noise_levels)},
                **{f"Noise_{n}_std": img_stds[i] for i, n in enumerate(noise_levels)},
                **{f"Noise_{n}_chosen_tau": None for n in noise_levels},
            })

            for method_variant, g_m in g_arch.groupby("method_variant"):
                # Build a dict temp -> list of run curves (for selection)
                temps = sorted(g_m["temp"].unique())
                temp_to_run_curves_test = {}

                for t in temps:
                    gt = g_m[g_m["temp"] == t]
                    curves = []
                    for _, r in gt.iterrows():
                        curves.append(get_curve(r["abs_path"], "eval test files"))
                    temp_to_run_curves_test[t] = curves

                # choose best temp per noise by mean(test) over runs
                chosen_temps = []
                best_means_test = []
                best_stds_test = []

                for j, noise in enumerate(noise_levels):
                    best_t = None
                    best_mean = None
                    best_std = None
                    for t, run_curves in temp_to_run_curves_test.items():
                        vals = [rc[j] for rc in run_curves]
                        vals = np.array([np.nan if v is None else float(v) for v in vals], dtype=float)
                        m = np.nanmean(vals)
                        s = np.nanstd(vals, ddof=0)
                        if np.isnan(m):
                            continue
                        if best_mean is None or m > best_mean:
                            best_mean = float(m)
                            best_std = float(s)
                            best_t = float(t)

                    chosen_temps.append(best_t)
                    best_means_test.append(best_mean)
                    best_stds_test.append(best_std)

                # Now compute mean/std for THIS eval_folder at those chosen temps
                means_eval = []
                stds_eval = []
                for j, noise in enumerate(noise_levels):
                    t = chosen_temps[j]
                    if t is None:
                        means_eval.append(np.nan)
                        stds_eval.append(np.nan)
                        continue

                    gt = g_m[g_m["temp"] == t]
                    run_vals = []
                    for _, r in gt.iterrows():
                        curve = get_curve(r["abs_path"], eval_folder)
                        v = curve[j]
                        run_vals.append(np.nan if v is None else float(v))

                    run_vals = np.array(run_vals, dtype=float)
                    means_eval.append(float(np.nanmean(run_vals)))
                    stds_eval.append(float(np.nanstd(run_vals, ddof=0)))

                row = {
                    "Arch": arch,
                    "Method": method_variant,
                    "ChosenTemps": chosen_temps,
                }
                for j, noise in enumerate(noise_levels):
                    row[f"Noise_{noise}_mean"] = means_eval[j]
                    row[f"Noise_{noise}_std"]  = stds_eval[j]
                    row[f"Noise_{noise}_chosen_tau"] = chosen_temps[j]
                rows.append(row)

            df_out = pd.DataFrame(rows)
            csv_path = os.path.join(save_dir, f"grouped_best_accuracy_{arch}_{safe_eval}.csv")
            df_out.to_csv(csv_path, index=False)
            print(f"[SAVED] {csv_path}")

            # Plot (mean +/- std) per method_variant - OLD STYLE
            fig, ax = plt.subplots(figsize=(8, 8))  # OLD: (8,8)

            for _, r in df_out.iterrows():
                means = [r[f"Noise_{n}_mean"] for n in noise_levels]
                stds = [r[f"Noise_{n}_std"] for n in noise_levels]

                ax.errorbar(
                    list(noise_levels),
                    means,
                    yerr=stds,
                    marker="o",
                    linewidth=3,  # OLD default-ish thickness
                    markersize=10,  # OLD default-ish marker size
                    label=r["Method"],
                    capsize=0,  # OLD look: no caps
                    elinewidth=3,
                )

            # OLD file didn't use multi-line titles and used ax.set_title(title, fontsize=label_fontsize-1)
            # Here we keep a simple title; if you want NONE like some old plots, set title=None
            _apply_old_plot_style(
                ax,
                title=None,
                xlabel="Noise Percentage",
                ylabel="Accuracy",
                label_fontsize=22,
                tick_fontsize=20,
            )

            fig.tight_layout()
            plot_path = os.path.join(save_dir, f"grouped_best_accuracy_{arch}_{safe_eval}.png")
            fig.savefig(plot_path, dpi=300)  # OLD: dpi=300
            plt.close(fig)
            print(f"[SAVED] {plot_path}")

def _apply_old_plot_style(
    ax,
    title=None,
    xlabel="Noise Percentage",
    ylabel="Accuracy",
    label_fontsize=22,
    tick_fontsize=20,
):
    """
    Match the exact plot aesthetics from the older file.
    """
    if title:
        # Old code used ax.set_title(title, fontsize=label_fontsize-1)
        ax.set_title(title, fontsize=label_fontsize - 1)

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    ax.grid(True)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))

    ax.legend(fontsize=tick_fontsize, loc="lower left")

    # Old file: thick ticks
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.tick_params(axis="both", width=4, size=8)

def create_plots(
    base_dir="/mnt/ssd/preshen/backup",
    figures_dir="/mnt/ssd/preshen/backup/run_view_with_noise_FIGURES",
    folder_list_json=None,  # e.g. "/path/to/folders_exp.json"
    architectures=("xception", "convnexttiny", "swintransformerv2tiny", "efficientnetv2s"),
    temps=(0.1, 0.2, 0.3, 0.35, 0.4, 0.5),
    runs=(0, 1),
    subfolder="downstream_with_pretrained_model_0",
):
    """
    End-to-end:
      1) discover experiments
      2) plot temp sweeps with mean+/-std over runs
      3) compute best-per-noise CSVs + plots (mean+/-std)

    Uses folder naming like those listed in your folders_exp.json.
    """
    exp_df = discover_experiments(
        base_dir=base_dir,
        folder_list_json=folder_list_json,
        allowed_archs=architectures,
        allowed_temps=temps,
        allowed_runs=runs,
    )
    print(f"[INFO] Discovered {len(exp_df)} experiment folders")

    # 1) temp-sweep plots (mean+/-std)
    plot_accuracy_by_noise(
        save_dir=figures_dir,
        name="tempsweep",
        exp_df=exp_df,
        subfolder=subfolder,
    )

    # 2) best-per-noise CSVs + plots (mean+/-std)
    create_best_csv(
        save_dir=figures_dir,
        exp_df=exp_df,
        subfolder=subfolder,
    )

    return exp_df
