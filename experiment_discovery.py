"""Experiment-folder discovery and metadata parsing utilities."""

import json
import os
import re

import pandas as pd

EXPERIMENT_RE = re.compile(
    r"^run_view_with_noise_Aug-True_"
    r"T(?P<T>\d{2,3})_"
    r"(?P<loss>supcon|logsum|dropcon)_"
    r"(?P<arch>[^_]+?)"
    r"(?:_k(?P<k>\d+)_v(?P<v>\d+))?"
    r"_run_(?P<run>\d+)$"
)

def _parse_temperature(T_str: str) -> float:
    """
    Folder encodes temperature as:
      T01 -> 0.1
      T02 -> 0.2
      T03 -> 0.3
      T04 -> 0.4
      T05 -> 0.5
      T035 -> 0.35
      T005 -> 0.05 (if present)
    """
    encoded = int(T_str)
    if len(T_str) == 2:
        return encoded / 10.0
    if len(T_str) == 3:
        return encoded / 100.0
    return float(encoded)

def discover_experiments(
    base_dir: str,
    folder_list_json: str = None,
    allowed_archs=None,
    allowed_losses=("supcon", "logsum", "dropcon"),
    allowed_temps=(0.1, 0.2, 0.3, 0.35, 0.4, 0.5),
    allowed_runs=(0, 1),
):
    """
    Returns a DataFrame where each row is a discovered experiment folder
    with parsed metadata.

    If folder_list_json is provided, uses that list (as in your attached folders_exp.json).
    Otherwise scans base_dir.
    """
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
        if allowed_temps is not None and temp not in set(allowed_temps):
            continue

        run = int(m.group("run"))
        if allowed_runs is not None and run not in set(allowed_runs):
            continue

        k = m.group("k")
        v = m.group("v")
        k = int(k) if k is not None else None
        v = int(v) if v is not None else None

        # method_variant: supcon / logsum / dropcon_k1_v2 / dropcon_k1_v3, etc
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

    # consistent ordering
    df = df.sort_values(["arch", "method_variant", "temp", "run"]).reset_index(drop=True)
    return df
