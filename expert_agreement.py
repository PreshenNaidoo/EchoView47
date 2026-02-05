"""Agreement metrics between model predictions and expert annotations."""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

def compute_cohens_kappa_across_noise_intersection_experts(
    base_prediction_json_path,
    expert1_labels,
    expert2_labels,
    expert3_labels,
    test_files,
    class_lookup,
    noise_levels=(0, 10, 20, 30, 40, 50),
    output_dir=None
):
    """
    Computes Cohen's kappa (model vs each expert) on the intersection subset
    where all THREE experts have valid labels, across multiple noise folders.

    Creates a CSV with columns: comparison, n, 0, 10, 20, 30, 40, 50
    """

    # ---------- Derive the noise subfolder position in the provided path ----------
    # Expect a component like: "ap0_pl0_ps0_do0_mm0"
    path_parts = os.path.normpath(base_prediction_json_path).split(os.sep)
    try:
        noise_idx = next(i for i, p in enumerate(path_parts)
                         if p.startswith("ap") and "_pl" in p and "_ps" in p and "_do" in p and "_mm" in p)
    except StopIteration:
        raise ValueError("Could not find the 'ap*_pl*_ps*_do*_mm*' noise subfolder in the given path.")

    def noise_folder(n: int) -> str:
        return f"ap{n}_pl{n}_ps{n}_do{n}_mm{n}"

    # ---------- Build the experts-only intersection (fixed across all runs) ----------
    valid_expert_mask = [
        (e1 is not None) and (e2 is not None) and (e3 is not None)
        for e1, e2, e3 in zip(expert1_labels, expert2_labels, expert3_labels)
    ]
    # Indices of the intersection subset
    idxs = [i for i, keep in enumerate(valid_expert_mask) if keep]
    n_intersection = len(idxs)

    # Pre-extract the experts' labels and the corresponding filenames for that subset
    inter_files = [test_files[i] for i in idxs]
    inter_e1    = [int(expert1_labels[i]) for i in idxs]
    inter_e2    = [int(expert2_labels[i]) for i in idxs]
    inter_e3    = [int(expert3_labels[i]) for i in idxs]

    labels_all = list(range(len(class_lookup)))

    # Prepare aggregation
    comparisons = ["Model vs Expert1", "Model vs Expert2", "Model vs Expert3"]
    kappas_by_noise = {name: [] for name in comparisons}

    # ---------- Loop over noise levels, load predictions, align, compute kappa ----------
    for n in noise_levels:
        # Build the prediction file path for this noise level
        parts_n = path_parts.copy()
        parts_n[noise_idx] = noise_folder(n)
        preds_path = os.path.join(*parts_n)

        # Load predictions
        if not os.path.exists(preds_path):
            # If missing, append None for all comparisons for this noise and continue
            for name in comparisons:
                kappas_by_noise[name].append(None)
            print(f"[WARN] Missing predictions json for noise {n}: {preds_path}")
            continue

        with open(preds_path, "r") as f:
            preds_json = json.load(f)

        file_to_pred = {fname: int(pred) for fname, pred in zip(preds_json["files"], preds_json["y_pred"])}

        # Align model preds to the experts-only intersection subset (drop any missing predictions)
        aligned_model = []
        aligned_e1 = []
        aligned_e2 = []
        aligned_e3 = []
        for f, e1, e2, e3 in zip(inter_files, inter_e1, inter_e2, inter_e3):
            m = file_to_pred.get(f, None)
            if m is None:
                continue
            aligned_model.append(int(m))
            aligned_e1.append(e1)
            aligned_e2.append(e2)
            aligned_e3.append(e3)

        # Compute kappa for this noise
        if len(aligned_model) == 0:
            k1 = k2 = k3 = None
        else:
            k1 = cohen_kappa_score(aligned_e1, aligned_model, labels=labels_all)
            k2 = cohen_kappa_score(aligned_e2, aligned_model, labels=labels_all)
            k3 = cohen_kappa_score(aligned_e3, aligned_model, labels=labels_all)

        kappas_by_noise["Model vs Expert1"].append(k1)
        kappas_by_noise["Model vs Expert2"].append(k2)
        kappas_by_noise["Model vs Expert3"].append(k3)

    # ---------- Build and save the CSV ----------
    rows = []
    for name in comparisons:
        row = {"comparison": name, "n": n_intersection}
        for i, n in enumerate(noise_levels):
            row[str(n)] = kappas_by_noise[name][i]
        rows.append(row)

    df = pd.DataFrame(rows, columns=["comparison", "n"] + [str(n) for n in noise_levels])

    out_dir = output_dir or os.path.dirname(base_prediction_json_path)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cohens_kappa_intersection_4882_across_noise.csv")
    df.to_csv(out_path, index=False)
    print(f"[OK] Saved Cohen's kappa across noise levels to {out_path}")

def compute_mwi_across_noise_intersection_experts(
    base_prediction_json_path,
    expert1_labels,
    expert2_labels,
    expert3_labels,
    test_files,
    class_lookup,
    noise_levels=(0, 10, 20, 30, 40, 50),
    output_dir=None,
    zero_tol=1e-12
):
    """
    Computes modified Williams Index (mWI) per noise level on the experts-only
    intersection subset, aligning to files that have model predictions at each noise level.

    Output CSV columns:
      noise, n_aligned, kappa_ME_mean, kappa_EE_mean, mWI,
      kappa_M_E1, kappa_M_E2, kappa_M_E3, kappa_E1_E2, kappa_E1_E3, kappa_E2_E3
    """

    # Locate the 'ap*_pl*_ps*_do*_mm*' component
    path_parts = os.path.normpath(base_prediction_json_path).split(os.sep)
    try:
        noise_idx = next(
            i for i, p in enumerate(path_parts)
            if p.startswith("ap") and "_pl" in p and "_ps" in p and "_do" in p and "_mm" in p
        )
    except StopIteration:
        raise ValueError("Could not find the 'ap*_pl*_ps*_do*_mm*' noise subfolder in the given path.")

    def noise_folder(n: int) -> str:
        return f"ap{n}_pl{n}_ps{n}_do{n}_mm{n}"

    # Experts-only intersection (no None)
    valid_expert_mask = [
        (e1 is not None) and (e2 is not None) and (e3 is not None)
        for e1, e2, e3 in zip(expert1_labels, expert2_labels, expert3_labels)
    ]
    idxs = [i for i, keep in enumerate(valid_expert_mask) if keep]

    inter_files = [test_files[i] for i in idxs]
    inter_e1    = [int(expert1_labels[i]) for i in idxs]
    inter_e2    = [int(expert2_labels[i]) for i in idxs]
    inter_e3    = [int(expert3_labels[i]) for i in idxs]

    labels_all = list(range(len(class_lookup)))

    rows = []
    for n in noise_levels:
        parts_n = path_parts.copy()
        parts_n[noise_idx] = noise_folder(n)
        preds_path = os.path.join(*parts_n)

        if not os.path.exists(preds_path):
            rows.append({
                "noise": n, "n_aligned": 0,
                "kappa_ME_mean": None, "kappa_EE_mean": None, "mWI": None,
                "kappa_M_E1": None, "kappa_M_E2": None, "kappa_M_E3": None,
                "kappa_E1_E2": None, "kappa_E1_E3": None, "kappa_E2_E3": None
            })
            print(f"[WARN] Missing predictions json for noise {n}: {preds_path}")
            continue

        with open(preds_path, "r") as f:
            preds_json = json.load(f)
        file_to_pred = {fname: int(pred) for fname, pred in zip(preds_json["files"], preds_json["y_pred"])}

        # Align to the files that have predictions at this noise level
        aligned_model, aligned_e1, aligned_e2, aligned_e3 = [], [], [], []
        for f, e1, e2, e3 in zip(inter_files, inter_e1, inter_e2, inter_e3):
            m = file_to_pred.get(f, None)
            if m is None:
                continue
            aligned_model.append(int(m))
            aligned_e1.append(e1)
            aligned_e2.append(e2)
            aligned_e3.append(e3)

        n_aligned = len(aligned_model)
        if n_aligned == 0:
            rows.append({
                "noise": n, "n_aligned": 0,
                "kappa_ME_mean": None, "kappa_EE_mean": None, "mWI": None,
                "kappa_M_E1": None, "kappa_M_E2": None, "kappa_M_E3": None,
                "kappa_E1_E2": None, "kappa_E1_E3": None, "kappa_E2_E3": None
            })
            continue

        # Model-Expert kappa
        k_M_E1 = cohen_kappa_score(aligned_e1, aligned_model, labels=labels_all)
        k_M_E2 = cohen_kappa_score(aligned_e2, aligned_model, labels=labels_all)
        k_M_E3 = cohen_kappa_score(aligned_e3, aligned_model, labels=labels_all)
        k_ME_mean = (k_M_E1 + k_M_E2 + k_M_E3) / 3.0

        # Expert-Expert kappa
        k_E1_E2 = cohen_kappa_score(aligned_e1, aligned_e2, labels=labels_all)
        k_E1_E3 = cohen_kappa_score(aligned_e1, aligned_e3, labels=labels_all)
        k_E2_E3 = cohen_kappa_score(aligned_e2, aligned_e3, labels=labels_all)
        k_EE_mean = (k_E1_E2 + k_E1_E3 + k_E2_E3) / 3.0

        # mWI (guard near-zero / NaN)
        if (k_EE_mean is None) or np.isnan(k_EE_mean) or np.isclose(k_EE_mean, 0.0, atol=zero_tol):
            mwi = None
        else:
            mwi = k_ME_mean / k_EE_mean

        rows.append({
            "noise": n, "n_aligned": n_aligned,
            "kappa_ME_mean": k_ME_mean, "kappa_EE_mean": k_EE_mean, "mWI": mwi,
            "kappa_M_E1": k_M_E1, "kappa_M_E2": k_M_E2, "kappa_M_E3": k_M_E3,
            "kappa_E1_E2": k_E1_E2, "kappa_E1_E3": k_E1_E3, "kappa_E2_E3": k_E2_E3
        })

    df = pd.DataFrame(rows, columns=[
        "noise","n_aligned","kappa_ME_mean","kappa_EE_mean","mWI",
        "kappa_M_E1","kappa_M_E2","kappa_M_E3","kappa_E1_E2","kappa_E1_E3","kappa_E2_E3"
    ])

    out_dir = output_dir or os.path.dirname(base_prediction_json_path)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "modified_williams_index_across_noise.csv")
    df.to_csv(out_path, index=False)
    print(f"[OK] Saved modified Williams Index across noise levels to {out_path}")
    return df
