"""Selection-aware statistical significance tests for best-configuration results.

This module provides:
- paired permutation tests that re-select best temperatures per resample,
- McNemar exact tests on the selected temperatures,
- bootstrap confidence intervals for accuracy deltas.
"""

import os
import json
import math
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional

# ----------------------------
# Folder parsing (same as your newer file)
# ----------------------------

EXPERIMENT_RE = re.compile(
    r"^run_view_with_noise_Aug-True_"
    r"T(?P<T>\d{2,3})_"
    r"(?P<loss>supcon|logsum|dropcon)_"
    r"(?P<arch>[^_]+?)"
    r"(?:_k(?P<k>\d+)_v(?P<v>\d+))?"
    r"_run_(?P<run>\d+)$"
)

def _parse_temperature(T_str: str) -> float:
    """Decode folder-formatted temperature strings such as T01 or T035."""

    d = int(T_str)
    if len(T_str) == 2:
        return d / 10.0
    elif len(T_str) == 3:
        return d / 100.0
    else:
        return float(d)

def discover_experiments(
    base_dir: str,
    folder_list_json: Optional[str] = None,
    allowed_archs=None,
    allowed_losses=("supcon", "logsum", "dropcon"),
    allowed_temps=(0.1, 0.2, 0.3, 0.35, 0.4, 0.5),
    allowed_runs=(0, 1),
) -> pd.DataFrame:
    """Discover run folders and return parsed metadata as a DataFrame."""

    if folder_list_json is not None:
        with open(folder_list_json, "r") as f:
            folders = json.load(f).get("exp", [])
    else:
        folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    rows = []
    for name in folders:
        m = EXPERIMENT_RE.match(name)
        if not m:
            continue

        loss = m.group("loss")
        if loss not in allowed_losses:
            continue

        arch = m.group("arch")
        if allowed_archs is not None and arch not in set(allowed_archs):
            continue

        temp = _parse_temperature(m.group("T"))
        if allowed_temps is not None and float(temp) not in set(allowed_temps):
            continue

        run = int(m.group("run"))
        if allowed_runs is not None and run not in set(allowed_runs):
            continue

        k = m.group("k")
        v = m.group("v")
        k = int(k) if k is not None else None
        v = int(v) if v is not None else None

        if loss == "dropcon":
            method_variant = f"dropcon_k{k}_v{v}" if (k is not None and v is not None) else "dropcon"
        else:
            method_variant = loss

        rows.append({
            "folder": name,
            "loss": loss,
            "method_variant": method_variant,
            "arch": arch,
            "temp": float(temp),
            "run": run,
            "k": k,
            "v": v,
            "abs_path": os.path.join(base_dir, name),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("[WARN] No experiments discovered. Check base_dir / filters / naming.")
        return df
    return df.sort_values(["arch", "method_variant", "temp", "run"]).reset_index(drop=True)


# ----------------------------
# Load per-image correctness
# ----------------------------

def _noise_folder(n: int, folder_template="ap{0}_pl{0}_ps{0}_do{0}_mm{0}") -> str:
    """Build the per-noise subfolder name from the configured template."""

    return folder_template.format(n)

def _pred_json_path(
    exp_abs_path: str,
    noise: int,
    subfolder: str,
    eval_folder: str,
    folder_template="ap{0}_pl{0}_ps{0}_do{0}_mm{0}",
    pred_file="all_predictions_0.json",
) -> str:
    """Build the path to one run's prediction JSON file."""

    return os.path.join(
        exp_abs_path,
        _noise_folder(noise, folder_template),
        subfolder,
        eval_folder,
        pred_file
    )

def _load_pred_json(pred_path: str) -> Dict:
    """Load and return one prediction JSON file."""

    with open(pred_path, "r") as f:
        return json.load(f)

def _build_correctness_map(pred_json: Dict) -> Tuple[Dict[str, int], Optional[Dict[str, int]]]:
    """
    Returns:
      correct_by_file: {file: 0/1}
      truth_by_file:   {file: y_true} if present else None
    """
    files = pred_json.get("files")
    y_pred = pred_json.get("y_pred")
    y_true = pred_json.get("y_true", None)

    if files is None or y_pred is None:
        raise ValueError("Prediction JSON missing required keys 'files' and/or 'y_pred'.")

    files = list(files)
    y_pred = [int(x) for x in y_pred]

    truth_by_file = None
    if y_true is not None:
        y_true = [int(x) for x in y_true]
        truth_by_file = {f: yt for f, yt in zip(files, y_true)}
        correct_by_file = {f: int(yp == truth_by_file[f]) for f, yp in zip(files, y_pred)}
    else:
        # If your JSON lacks y_true, you must supply truth some other way.
        # For safety, we fail explicitly (better than silently wrong stats).
        raise ValueError("Prediction JSON has no 'y_true'. Add it or implement a y_true_provider.")
    return correct_by_file, truth_by_file


def load_run_correctness(
    exp_abs_path: str,
    noise: int,
    subfolder: str,
    eval_folder: str,
    folder_template="ap{0}_pl{0}_ps{0}_do{0}_mm{0}",
    pred_file="all_predictions_0.json",
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Loads correctness per file for ONE run directory.
    Returns:
      correct_by_file, truth_by_file
    """
    p = _pred_json_path(exp_abs_path, noise, subfolder, eval_folder, folder_template, pred_file)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    d = _load_pred_json(p)
    correct_by_file, truth_by_file = _build_correctness_map(d)
    return correct_by_file, truth_by_file


def mean_over_runs_vector(
    run_maps: List[Dict[str, int]],
    require_all_runs: bool = True
) -> Tuple[List[str], np.ndarray]:
    """
    Given per-run correctness maps for the same method+temp, return:
      files_common, per_image_mean_correctness (float in [0,1])
    If require_all_runs=True, intersects files across all runs.
    """
    if len(run_maps) == 0:
        return [], np.array([], dtype=float)

    if require_all_runs:
        common = set(run_maps[0].keys())
        for m in run_maps[1:]:
            common &= set(m.keys())
    else:
        common = set()
        for m in run_maps:
            common |= set(m.keys())

    files = sorted(common)
    if not files:
        return [], np.array([], dtype=float)

    A = np.zeros((len(files), len(run_maps)), dtype=float)
    for j, m in enumerate(run_maps):
        A[:, j] = [float(m[f]) for f in files]
    return files, A.mean(axis=1)


# ----------------------------
# Stats: McNemar (exact) + Holm
# ----------------------------

def _binom_two_sided_pvalue(k: int, n: int) -> float:
    """
    Exact two-sided binomial test p-value for k successes in n trials under p=0.5,
    implemented without scipy.
    """
    if n == 0:
        return 1.0
    # probability mass function
    def pmf(x):
        return math.comb(n, x) * (0.5 ** n)

    p_obs = pmf(k)
    # two-sided: sum of probabilities <= p_obs
    p = 0.0
    for x in range(n + 1):
        if pmf(x) <= p_obs + 1e-15:
            p += pmf(x)
    return min(1.0, p)

def mcnemar_exact(correct_A: np.ndarray, correct_B: np.ndarray) -> Dict:
    """
    correct_A/B are per-image correctness values in {0,1} (or floats).
    We threshold at >0.5 to treat mean-over-runs as "correct if majority of runs correct".
    If you want strict per-run treatment, use run-level testing instead.
    """
    a = (correct_A > 0.5).astype(int)
    b = (correct_B > 0.5).astype(int)

    # contingency:
    # A=1,B=0 -> n10 ; A=0,B=1 -> n01
    n10 = int(np.sum((a == 1) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n = n10 + n01
    # under null, n10 ~ Binom(n, 0.5)
    k = min(n10, n01)  # symmetric for two-sided
    p = _binom_two_sided_pvalue(k, n)
    return {"n10_A_correct_B_wrong": n10, "n01_A_wrong_B_correct": n01, "n_discordant": n, "p_two_sided": p}

def holm_adjust(pvals: List[Optional[float]]) -> List[Optional[float]]:
    """
    Holm adjustment; keeps None as None.
    """
    idx = [i for i, p in enumerate(pvals) if p is not None and not (isinstance(p, float) and np.isnan(p))]
    if not idx:
        return [None for _ in pvals]

    pv = np.array([pvals[i] for i in idx], dtype=float)
    order = np.argsort(pv)
    m = len(pv)

    adj_sorted = np.zeros(m, dtype=float)
    for r, j in enumerate(order):
        adj_sorted[r] = (m - r) * pv[j]
    adj_sorted = np.maximum.accumulate(adj_sorted)
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)

    adj = np.zeros(m, dtype=float)
    for r, j in enumerate(order):
        adj[j] = adj_sorted[r]

    out = [None for _ in pvals]
    for j, i in enumerate(idx):
        out[i] = float(adj[j])
    return out


# ----------------------------
# Selection-aware permutation + bootstrap CI
# ----------------------------

def _select_best_temp(
    temp_to_vec: Dict[float, np.ndarray],
) -> Tuple[Optional[float], float]:
    """
    Select temperature that maximizes mean accuracy.
    Returns (best_temp, best_mean).
    """
    best_t = None
    best_m = None
    for t, v in temp_to_vec.items():
        if v.size == 0:
            continue
        m = float(np.mean(v))
        if best_m is None or m > best_m:
            best_m = m
            best_t = float(t)
    if best_t is None:
        return None, float("nan")
    return best_t, float(best_m)

def selection_aware_permutation_test(
    A_temp_to_vec: Dict[float, np.ndarray],
    B_temp_to_vec: Dict[float, np.ndarray],
    n_perm: int = 5000,
    seed: int = 0,
    two_sided: bool = True,
) -> Dict:
    """
    Selection-aware test:
      - Choose best temp for A and B based on mean accuracy (observed).
      - Permute by swapping A/B labels per-image (same swap mask across all temps),
        re-select best temps under permuted data, compute delta.
    Returns p-values and chosen temps.
    """
    temps_A = sorted(A_temp_to_vec.keys())
    temps_B = sorted(B_temp_to_vec.keys())

    # observed selection
    tA_obs, mA_obs = _select_best_temp(A_temp_to_vec)
    tB_obs, mB_obs = _select_best_temp(B_temp_to_vec)
    delta_obs = mA_obs - mB_obs

    # if either missing
    if tA_obs is None or tB_obs is None:
        return {
            "chosen_temp_A": tA_obs,
            "chosen_temp_B": tB_obs,
            "mean_A": mA_obs,
            "mean_B": mB_obs,
            "delta": delta_obs,
            "p_perm_two_sided": None,
            "p_perm_one_sided_A_gt_B": None,
            "n_perm": n_perm,
        }

    # ensure all vectors same length N (we enforce intersection upstream)
    N = next(iter(A_temp_to_vec.values())).shape[0]
    for t in temps_A:
        assert A_temp_to_vec[t].shape[0] == N
    for t in temps_B:
        assert B_temp_to_vec[t].shape[0] == N

    rng = np.random.default_rng(seed)
    deltas = np.zeros(n_perm, dtype=float)

    for k in range(n_perm):
        swap = rng.integers(0, 2, size=N).astype(bool)  # True => swap A/B for that image

        A_perm = {}
        B_perm = {}
        for t in temps_A:
            a = A_temp_to_vec[t].copy()
            b = B_temp_to_vec.get(t, None)  # NOTE: temps differ across methods; handle separately below
            # We can't swap with b here if B doesn't have that temp. We'll swap across each method's own temps
            # using the same swap mask against the corresponding other-method temp vectors via broadcasting later.
            A_perm[t] = a
        for t in temps_B:
            B_perm[t] = B_temp_to_vec[t].copy()

        # To make swapping well-defined even when temp sets differ,
        # we swap each method's vectors against the other method's vectors at the SAME INDEXED images,
        # but only for temps existing in both. For temps unique to one method,
        # the selection-aware null becomes less clean. In practice your temp grids are shared (0.1..0.5).
        # We'll enforce shared temps intersection to be safe:
        # shared_temps = sorted(set(temps_A) & set(temps_B))
        # if len(shared_temps) == 0:
        #     break
        shared_temps = sorted(set(temps_A) & set(temps_B))
        if len(shared_temps) == 0:
            return {
                "chosen_temp_A": tA_obs,
                "chosen_temp_B": tB_obs,
                "mean_A": float(mA_obs),
                "mean_B": float(mB_obs),
                "delta": float(delta_obs),
                "p_perm_two_sided": None,
                "p_perm_one_sided_A_gt_B": None,
                "n_perm": int(n_perm),
                "shared_temps_used": [],
                "N_images": int(N) if "N" in locals() else None,
                "error": "No shared temps between methods in permutation test."
            }

        for t in shared_temps:
            a = A_perm[t]
            b = B_perm[t]
            a_sw = a.copy()
            b_sw = b.copy()
            a_sw[swap] = b[swap]
            b_sw[swap] = a[swap]
            A_perm[t] = a_sw
            B_perm[t] = b_sw

        tA_p, mA_p = _select_best_temp({t: A_perm[t] for t in shared_temps})
        tB_p, mB_p = _select_best_temp({t: B_perm[t] for t in shared_temps})
        deltas[k] = mA_p - mB_p

    # p-values
    if two_sided:
        p2 = (np.sum(np.abs(deltas) >= abs(delta_obs)) + 1.0) / (n_perm + 1.0)
    else:
        p2 = None
    p1 = (np.sum(deltas >= delta_obs) + 1.0) / (n_perm + 1.0)  # one-sided A > B

    return {
        "chosen_temp_A": tA_obs,
        "chosen_temp_B": tB_obs,
        "mean_A": float(mA_obs),
        "mean_B": float(mB_obs),
        "delta": float(delta_obs),
        "p_perm_two_sided": float(p2) if p2 is not None else None,
        "p_perm_one_sided_A_gt_B": float(p1),
        "n_perm": int(n_perm),
        "shared_temps_used": sorted(list(set(temps_A) & set(temps_B))),
        "N_images": int(N),
    }

def selection_aware_bootstrap_ci(
    A_temp_to_vec: Dict[float, np.ndarray],
    B_temp_to_vec: Dict[float, np.ndarray],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict:
    """
    Bootstrap CI for delta (A-B) with selection re-done inside each resample.
    Assumes all vectors share length N (enforced upstream).
    Uses only shared temps intersection (safe).
    """
    temps = sorted(set(A_temp_to_vec.keys()) & set(B_temp_to_vec.keys()))
    if len(temps) == 0:
        return {"ci_low": None, "ci_high": None, "n_boot": n_boot}

    N = A_temp_to_vec[temps[0]].shape[0]
    rng = np.random.default_rng(seed)
    deltas = np.zeros(n_boot, dtype=float)

    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)  # resample images with replacement
        A_bs = {t: A_temp_to_vec[t][idx] for t in temps}
        B_bs = {t: B_temp_to_vec[t][idx] for t in temps}
        _, mA = _select_best_temp(A_bs)
        _, mB = _select_best_temp(B_bs)
        deltas[b] = mA - mB

    lo = float(np.quantile(deltas, alpha / 2))
    hi = float(np.quantile(deltas, 1 - alpha / 2))
    return {"ci_low": lo, "ci_high": hi, "n_boot": int(n_boot)}


# ----------------------------
# Main orchestrator
# ----------------------------

def run_best_config_significance_tests(
    base_dir: str,
    folder_list_json: str,
    out_dir: str,
    architectures=("xception", "convnexttiny", "swintransformerv2tiny", "efficientnetv2s"),
    methods=("supcon", "logsum", "dropcon_k1_v2", "dropcon_k1_v3"),
    # if you want to collapse dropcon variants into a single "dropcon" bucket, see note below
    method_pairs=(("supcon","logsum"), ("supcon","dropcon_k1_v2"), ("logsum","dropcon_k1_v2")),
    noise_levels=(0, 10, 20, 30, 40, 50),
    subfolder="downstream_with_pretrained_model_0",
    eval_folders=("eval test files",),
    folder_template="ap{0}_pl{0}_ps{0}_do{0}_mm{0}",
    pred_file="all_predictions_0.json",
    require_all_runs=True,
    n_perm=5000,
    n_boot=2000,
    seed=0,
):
    """
    Produces JSON files with selection-aware permutation tests and McNemar tests.

    Output structure per JSON:
      {
        "meta": {...},
        "results": {
          "<arch>": {
             "<methodA>__vs__<methodB>": [
                { "noise": 0, "N_images": ..., "selected": {...}, "perm_test": {...}, "mcnemar": {...}, "bootstrap_ci": {...} },
                ...
             ],
             ...
          },
          ...
        }
      }
    """
    os.makedirs(out_dir, exist_ok=True)

    exp_df = discover_experiments(
        base_dir=base_dir,
        folder_list_json=folder_list_json,
        allowed_archs=architectures,
        allowed_temps=(0.1,0.2,0.3,0.35,0.4,0.5),
        allowed_runs=(0,1),
    )
    if exp_df.empty:
        raise RuntimeError("No experiments discovered; cannot run significance tests.")

    # filter methods of interest
    exp_df = exp_df[exp_df["method_variant"].isin(methods)].copy()
    if exp_df.empty:
        raise RuntimeError("After filtering methods, exp_df is empty.")

    # cache: (abs_path, noise, eval_folder) -> correctness_map
    corr_cache = {}

    def get_correctness_map(abs_path, noise, eval_folder):
        key = (abs_path, int(noise), eval_folder)
        if key in corr_cache:
            return corr_cache[key]
        correct_by_file, truth_by_file = load_run_correctness(
            exp_abs_path=abs_path,
            noise=int(noise),
            subfolder=subfolder,
            eval_folder=eval_folder,
            folder_template=folder_template,
            pred_file=pred_file,
        )
        corr_cache[key] = (correct_by_file, truth_by_file)
        return corr_cache[key]

    # Build per-arch, per-method, per-temp: list of runs (abs paths)
    # We'll later load each (noise, eval_folder) to correctness maps and average over runs per-image.
    out_all_eval = {}

    for eval_folder in eval_folders:
        results = {arch: {} for arch in architectures}

        for arch in architectures:
            g_arch = exp_df[exp_df["arch"] == arch]
            if g_arch.empty:
                continue

            # Pre-index: method -> temp -> list of run abs paths
            method_temp_runs = {}
            for method in methods:
                g_m = g_arch[g_arch["method_variant"] == method]
                if g_m.empty:
                    continue
                method_temp_runs[method] = {}
                for temp, gt in g_m.groupby("temp"):
                    # list run paths
                    method_temp_runs[method][float(temp)] = gt["abs_path"].tolist()

            # For each method-pair and noise, build temp->vector (mean correctness per image)
            for (mA, mB) in method_pairs:
                if mA not in method_temp_runs or mB not in method_temp_runs:
                    continue

                pair_key = f"{mA}__vs__{mB}"
                results[arch][pair_key] = []

                shared_temps = sorted(set(method_temp_runs[mA].keys()) & set(method_temp_runs[mB].keys()))
                if len(shared_temps) == 0:
                    print(f"[WARN] No shared temps for {arch} {mA} vs {mB}; skipping.")
                    continue

                for noise in noise_levels:
                    # load each temp's per-image correctness (mean over runs)
                    A_temp_to_files = {}
                    A_temp_to_vec = {}
                    B_temp_to_files = {}
                    B_temp_to_vec = {}

                    # First build per-temp vectors (each with its own file intersection across runs)
                    for t in shared_temps:
                        # A
                        run_maps_A = []
                        truthA = None
                        for abs_path in method_temp_runs[mA][t]:
                            try:
                                corr_map, truth_map = get_correctness_map(abs_path, noise, eval_folder)
                                run_maps_A.append(corr_map)
                                truthA = truth_map if truthA is None else truthA
                            except FileNotFoundError:
                                continue
                        files_A, vec_A = mean_over_runs_vector(run_maps_A, require_all_runs=require_all_runs)
                        if len(files_A) == 0:
                            continue
                        A_temp_to_files[t] = files_A
                        A_temp_to_vec[t] = vec_A

                        # B
                        run_maps_B = []
                        truthB = None
                        for abs_path in method_temp_runs[mB][t]:
                            try:
                                corr_map, truth_map = get_correctness_map(abs_path, noise, eval_folder)
                                run_maps_B.append(corr_map)
                                truthB = truth_map if truthB is None else truthB
                            except FileNotFoundError:
                                continue
                        files_B, vec_B = mean_over_runs_vector(run_maps_B, require_all_runs=require_all_runs)
                        if len(files_B) == 0:
                            continue
                        B_temp_to_files[t] = files_B
                        B_temp_to_vec[t] = vec_B

                    # If missing temps at this noise, skip
                    usable_temps = sorted(set(A_temp_to_vec.keys()) & set(B_temp_to_vec.keys()))
                    if len(usable_temps) == 0:
                        results[arch][pair_key].append({
                            "noise": int(noise),
                            "status": "no_data",
                            "N_images": 0,
                        })
                        continue

                    # Now enforce a SINGLE common file set across ALL usable temps (selection-aware stability)
                    common_files = None
                    for t in usable_temps:
                        sA = set(A_temp_to_files[t])
                        sB = set(B_temp_to_files[t])
                        s = sA & sB
                        common_files = s if common_files is None else (common_files & s)

                    common_files = sorted(list(common_files)) if common_files is not None else []
                    if len(common_files) == 0:
                        results[arch][pair_key].append({
                            "noise": int(noise),
                            "status": "no_common_files_across_temps",
                            "N_images": 0,
                        })
                        continue

                    # Rebuild vectors on the common file set for each temp
                    def restrict(files, vec, common):
                        idx = {f:i for i,f in enumerate(files)}
                        return np.array([vec[idx[f]] for f in common], dtype=float)

                    A_temp_common = {t: restrict(A_temp_to_files[t], A_temp_to_vec[t], common_files) for t in usable_temps}
                    B_temp_common = {t: restrict(B_temp_to_files[t], B_temp_to_vec[t], common_files) for t in usable_temps}

                    # Selection-aware permutation test + bootstrap CI
                    perm = selection_aware_permutation_test(
                        A_temp_to_vec=A_temp_common,
                        B_temp_to_vec=B_temp_common,
                        n_perm=n_perm,
                        seed=seed,
                        two_sided=True,
                    )
                    ci = selection_aware_bootstrap_ci(
                        A_temp_to_vec=A_temp_common,
                        B_temp_to_vec=B_temp_common,
                        n_boot=n_boot,
                        seed=seed,
                        alpha=0.05,
                    )

                    # McNemar on the chosen best temps (secondary)
                    tA = perm.get("chosen_temp_A", None)
                    tB = perm.get("chosen_temp_B", None)
                    if tA is None or tB is None:
                        mcn = None
                    else:
                        mcn = mcnemar_exact(A_temp_common[tA], B_temp_common[tB])

                    results[arch][pair_key].append({
                        "noise": int(noise),
                        "status": "ok",
                        "N_images": int(len(common_files)),
                        "selected": {
                            "chosen_temp_A": perm.get("chosen_temp_A"),
                            "chosen_temp_B": perm.get("chosen_temp_B"),
                            "mean_A": perm.get("mean_A"),
                            "mean_B": perm.get("mean_B"),
                            "delta_A_minus_B": perm.get("delta"),
                        },
                        "perm_test_selection_aware": perm,
                        "bootstrap_ci_selection_aware": ci,
                        "mcnemar_on_selected": mcn,
                    })

                # Holm-adjust across the 6 noises for this arch + pair (selection-aware permutation p-values)
                pvals = []
                for row in results[arch][pair_key]:
                    if row.get("status") != "ok":
                        pvals.append(None)
                    else:
                        pvals.append(row["perm_test_selection_aware"].get("p_perm_two_sided", None))
                adj = holm_adjust(pvals)
                for i, row in enumerate(results[arch][pair_key]):
                    row["perm_test_selection_aware"]["holm_adj_p_two_sided_across_noises"] = adj[i]

        out = {
            "meta": {
                "base_dir": base_dir,
                "folder_list_json": folder_list_json,
                "subfolder": subfolder,
                "eval_folder": eval_folder,
                "pred_file": pred_file,
                "noise_levels": list(noise_levels),
                "architectures": list(architectures),
                "methods": list(methods),
                "method_pairs": [list(x) for x in method_pairs],
                "n_perm": int(n_perm),
                "n_boot": int(n_boot),
                "seed": int(seed),
                "require_all_runs": bool(require_all_runs),
                "note": "Permutation/CI are selection-aware (reselect best temp inside resamples). Holm adjustment is across noises per arch+pair."
            },
            "results": results
        }

        safe_eval = eval_folder.replace(" ", "_").lower()
        out_path = os.path.join(out_dir, f"significance_best_config_{safe_eval}.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[SAVED] {out_path}")

        out_all_eval[eval_folder] = out_path

    return out_all_eval


# ----------------------------
# Example usage
# ----------------------------
# if __name__ == "__main__":
#     # Edit these paths to match your environment
#     BASE_DIR = "/mnt/ssd/preshen/backup"
#     FOLDERS_JSON = "/mnt/data/folders_exp.json"   # <- your uploaded file path
#     OUT_DIR = os.path.join(BASE_DIR, "run_view_with_noise_STATS")
#
#     # If you want "DropCon" as a single bucket rather than k1_v2 vs k1_v3,
#     # set methods=("supcon","logsum","dropcon_k1_v2","dropcon_k1_v3") and compare pairs as you like,
#     # OR run two separate comparisons against each dropcon variant.
#
#     run_best_config_significance_tests(
#         base_dir=BASE_DIR,
#         folder_list_json=FOLDERS_JSON,
#         out_dir=OUT_DIR,
#         architectures=("xception", "convnexttiny", "swintransformerv2tiny", "efficientnetv2s"),
#         methods=("supcon", "logsum", "dropcon_k1_v2", "dropcon_k1_v3"),
#         method_pairs=(
#             ("supcon","logsum"),
#             ("supcon","dropcon_k1_v2"),
#             ("logsum","dropcon_k1_v2"),
#             # add these if you also want v3 comparisons:
#             ("supcon","dropcon_k1_v3"),
#             ("logsum","dropcon_k1_v3"),
#             ("dropcon_k1_v2","dropcon_k1_v3"),
#         ),
#         noise_levels=(0,10,20,30,40,50),
#         subfolder="downstream_with_pretrained_model_0",
#         eval_folders=("eval test files",),  # add others if you want
#         pred_file="all_predictions_0.json",
#         require_all_runs=True,
#         n_perm=5000,
#         n_boot=2000,
#         seed=0,
#     )
