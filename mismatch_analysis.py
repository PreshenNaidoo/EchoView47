"""Mismatch and disagreement analysis utilities for qualitative diagnostics."""

import json
import os
import random
import shutil

import matplotlib.pyplot as plt
import pandas as pd

def save_disagreement_examples(
    prediction_json_path,
    expert1_labels,
    expert2_labels,
    expert3_labels,
    test_files,
    class_lookup,
    output_folder,
    num_samples=10,
    seed=42
):
    """Copy sampled disagreement cases into category folders and save label CSVs."""

    os.makedirs(output_folder, exist_ok=True)
    random.seed(seed)

    # Load predictions and align to test_files
    with open(prediction_json_path, 'r') as f:
        preds = json.load(f)

    file_to_pred = {fname: int(pred) for fname, pred in zip(preds["files"], preds["y_pred"])}
    model_preds = [file_to_pred.get(f, None) for f in test_files]

    int_to_class = {int(k): v for k, v in class_lookup.items()}

    categories = {
        "complete_disagreement": [],
        "complete_agreement": [],
        "partial_disagreement_1": [],
        "partial_disagreement_2": [],
        "partial_disagreement_3": [],
        "experts_agree_model_disagrees": [],
        "model_agrees_with_2": []
    }

    for file, e1, e2, e3, m in zip(test_files, expert1_labels, expert2_labels, expert3_labels, model_preds):
        if None in (e1, e2, e3, m):
            continue

        unique_experts = set([e1, e2, e3])
        expert_agreement = len(unique_experts) == 1
        model_agrees = lambda x: m == x

        if len(unique_experts) == 3 and m not in unique_experts:
            categories["complete_disagreement"].append((file, m, e1, e2, e3))

        elif expert_agreement and m == e1:
            categories["complete_agreement"].append((file, m, e1, e2, e3))

        elif not expert_agreement:
            if model_agrees(e1):
                categories["partial_disagreement_1"].append((file, m, e1, e2, e3))
            elif model_agrees(e2):
                categories["partial_disagreement_2"].append((file, m, e1, e2, e3))
            elif model_agrees(e3):
                categories["partial_disagreement_3"].append((file, m, e1, e2, e3))

        if expert_agreement and m != e1:
            categories["experts_agree_model_disagrees"].append((file, m, e1, e2, e3))

        if sum([m == e1, m == e2, m == e3]) == 2:
            categories["model_agrees_with_2"].append((file, m, e1, e2, e3))

    for category, items in categories.items():
        cat_folder = os.path.join(output_folder, category)
        os.makedirs(cat_folder, exist_ok=True)
        chosen = random.sample(items, min(num_samples, len(items)))

        csv_data = []

        for i, (file, m, e1, e2, e3) in enumerate(chosen):
            filename = os.path.basename(file)
            new_filename = f"{i:02d}_{filename}"
            dest_path = os.path.join(cat_folder, new_filename)

            try:
                shutil.copy(file, dest_path)
            except Exception as e:
                print(f"Failed to copy {file}: {e}")
                continue

            csv_data.append({
                "filename": new_filename,
                "model": int_to_class[m],
                "expert1": int_to_class[e1],
                "expert2": int_to_class[e2],
                "expert3": int_to_class[e3]
            })

        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(os.path.join(cat_folder, "labels.csv"), index=False)

    print("[OK] Disagreement examples saved.")

def plot_mismatch_analysis_from_json(json_path, output_dir="figures"):
    """
    Load mismatch data from JSON and generate four plots using matplotlib only:
    1. Top 10 fine-grained mismatches (bar plot)
    2. Intra vs. inter-group mismatch (bar plot)
    3. Group-level (7-view) confusion heatmap
    4. Per-class mismatch frequency (top 15)
    """
    # Font size config
    title_size = 18
    label_size = 14
    tick_size = 12
    text_size = 12

    os.makedirs(output_dir, exist_ok=True)

    # Load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # Flatten to DataFrame
    records = []
    for file_name, entry in data.items():
        records.append({
            "file": file_name,
            "gt_class_index": entry["gt_class_index"],
            "gt_label": entry["gt_label"],
            "cluster_dominant_class_index": entry["cluster_dominant_class_index"],
            "cluster_dominant_label": entry["cluster_dominant_label"],
            "cluster_id": entry["cluster_id"]
        })
    df = pd.DataFrame(records)

    # Coarse group mapping
    coarse_group_mapping = {
        "a2ch-full": "apical", "a2ch-la": "apical", "a2ch-lv": "apical",
        "a3ch-full": "apical", "a3ch-la": "apical", "a3ch-lv": "apical", "a3ch-outflow": "apical",
        "a4ch-full": "apical", "a4ch-la": "apical", "a4ch-lv": "apical", "a4ch-ias": "apical", "a4ch-ra": "apical", "a4ch-rv": "apical",
        "a5ch-full": "apical", "a5ch-outflow": "apical", "apex": "apical",
        "plax-full-out": "plax", "plax-full-lv": "plax", "plax-full-la": "plax", "plax-full-mv": "plax",
        "plax-full-rv-ao": "plax", "plax-valves-av": "plax", "plax-valves-mv": "plax", "plax-tv": "plax",
        "psax-all": "psax", "psax-lv-base": "psax", "psax-lv-mid": "psax", "psax-lv-apex": "psax",
        "psax-av": "psax", "psax-tv": "psax", "psax-pv": "psax",
        "subcostal-ivc": "subcostal", "subcostal-heart": "subcostal",
        "suprasternal": "suprasternal",
        "mmode-a4ch-rv": "mmode", "mmode-ivc": "mmode", "mmode-plax-av": "mmode",
        "mmode-plax-lv": "mmode", "mmode-plax-mitral": "mmode",
        "doppler-ao-descending": "doppler", "doppler-av": "doppler", "doppler-mv": "doppler", "doppler-pv": "doppler",
        "doppler-tv": "doppler", "doppler-tissue-lateral": "doppler",
        "doppler-tissue-rv": "doppler", "doppler-tissue-septal": "doppler"
    }

    # View group annotations
    df["gt_group"] = df["gt_label"].map(coarse_group_mapping)
    df["cluster_group"] = df["cluster_dominant_label"].map(coarse_group_mapping)

    # Display alias for plots
    display_alias = {
        "doppler-ao-descending": "doppler-ao"
    }
    df["gt_label_display"] = df["gt_label"].replace(display_alias)
    df["cluster_dominant_label_display"] = df["cluster_dominant_label"].replace(display_alias)

    # Plot 1: Top 10 fine-grained mismatches
    mismatch_counts = df.groupby(["gt_label_display", "cluster_dominant_label_display"]).size().reset_index(name="count")
    top_mismatches = mismatch_counts.sort_values(by="count", ascending=False).head(10)
    labels = [f"{row['gt_label_display']} -> {row['cluster_dominant_label_display']}" for _, row in top_mismatches.iterrows()]
    counts = top_mismatches["count"].tolist()

    plt.figure(figsize=(10, 6))
    plt.barh(labels[::-1], counts[::-1])
    plt.xlabel("Count", fontsize=label_size)
    plt.title("Top 10 Fine-Grained View Mismatches", fontsize=title_size)
    plt.xticks(fontsize=tick_size+4)
    plt.yticks(fontsize=tick_size+4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_10_mismatches.png"))
    plt.close()

    # Plot 2: Intra vs Inter group mismatch
    is_intragroup = df["gt_group"] == df["cluster_group"]
    counts = [is_intragroup.sum(), (~is_intragroup).sum()]

    plt.figure(figsize=(6, 4))
    plt.bar(["Intra-group", "Inter-group"], counts)
    plt.title("Mismatch Distribution: Intra vs Inter Group", fontsize=title_size)
    plt.ylabel("Count", fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "intra_vs_inter_group.png"))
    plt.close()

    # Plot 3: Group-level confusion heatmap
    group_conf = df.groupby(["gt_group", "cluster_group"]).size().unstack(fill_value=0)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(group_conf, cmap='Blues')
    plt.xticks(ticks=range(len(group_conf.columns)), labels=group_conf.columns, rotation=45, ha="right", fontsize=tick_size)
    plt.yticks(ticks=range(len(group_conf.index)), labels=group_conf.index, fontsize=tick_size)
    for i in range(len(group_conf.index)):
        for j in range(len(group_conf.columns)):
            plt.text(j, i, group_conf.iloc[i, j], ha='center', va='center', color='black', fontsize=text_size)
    plt.colorbar(im)
    plt.title("Group-Level Confusion Heatmap", fontsize=title_size)
    plt.xlabel("Cluster Group", fontsize=label_size)
    plt.ylabel("Ground Truth Group", fontsize=label_size)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "group_confusion_heatmap.png"))
    plt.close()

    # Plot 4: Per-class mismatch frequency (Top 15)
    class_counts = df["gt_label_display"].value_counts().sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    plt.barh(class_counts.index[::-1], class_counts.values[::-1])
    plt.xlabel("Mismatch Count", fontsize=label_size)
    plt.title("Top 15 Most Frequently Mismatched Classes", fontsize=title_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_class_mismatches.png"))
    plt.close()

def evaluate_hierarchical_accuracy(all_preds_json_path, class_lookup):
    """
    Compute accuracy under 7-view and 20-view groupings, by mapping fine-grained predictions and labels.
    """
    # ---------------------- #
    # Define hierarchy maps
    # ---------------------- #

    # Fine (47 class) -> Intermediate (20 class)
    view_20_mapping = {
        # Apical group
        "a2ch-full": "a2ch", "a2ch-la": "a2ch", "a2ch-lv": "a2ch",
        "a3ch-full": "a3ch", "a3ch-la": "a3ch", "a3ch-lv": "a3ch", "a3ch-outflow": "a3ch",
        "a4ch-full": "a4ch", "a4ch-la": "a4ch", "a4ch-lv": "a4ch", "a4ch-ias": "a4ch", "a4ch-ra": "a4ch", "a4ch-rv": "a4ch",
        "a5ch-full": "a5ch", "a5ch-outflow": "a5ch",
        "apex": "apex",

        # PLAX
        "plax-full-out": "PLAX", "plax-full-lv": "PLAX", "plax-full-la": "PLAX", "plax-full-mv": "PLAX",
        "plax-full-rv-ao": "PLAX", "plax-valves-av": "PLAX", "plax-valves-mv": "PLAX", "plax-tv": "PLAX",

        # PSAX
        "psax-all": "PSAX", "psax-lv-base": "PSAX", "psax-lv-mid": "PSAX", "psax-lv-apex": "PSAX",
        "psax-av": "PSAX", "psax-tv": "PSAX", "psax-pv": "PSAX",

        # Subcostal & suprasternal
        "subcostal-heart": "subcostal", "subcostal-ivc": "subcostal",
        "suprasternal": "suprasternal",

        # M-mode
        "mmode-a4ch-rv": "mmode-a4ch-rv", "mmode-ivc": "mmode-ivc",
        "mmode-plax-av": "mmode-plax", "mmode-plax-lv": "mmode-plax", "mmode-plax-mitral": "mmode-plax",

        # Doppler
        "doppler-ao-descending": "doppler-ao",
        "doppler-av": "doppler-av",
        "doppler-mv": "doppler-mv",
        "doppler-tv": "doppler-tv",
        "doppler-pv": "doppler-pv",
        "doppler-tissue-lateral": "doppler-tissue-lat",
        "doppler-tissue-rv": "doppler-tissue-rv",
        "doppler-tissue-septal": "doppler-tissue-septal"
    }

    # Intermediate (20) -> Coarse (7)
    view_7_mapping = {
        "a2ch": "apical", "a3ch": "apical", "a4ch": "apical", "a5ch": "apical", "apex": "apical",
        "PLAX": "plax",
        "PSAX": "psax",
        "subcostal": "subcostal",
        "suprasternal": "suprasternal",
        "mmode-a4ch-rv": "mmode",
        "mmode-ivc": "mmode",
        "mmode-plax": "mmode",
        "doppler-ao": "doppler", "doppler-av": "doppler", "doppler-mv": "doppler", "doppler-tv": "doppler",
        "doppler-pv": "doppler", "doppler-tissue-lat": "doppler", "doppler-tissue-rv": "doppler", "doppler-tissue-septal": "doppler"
    }

    # ---------------------- #
    # Load prediction data
    # ---------------------- #
    with open(all_preds_json_path, 'r') as f:
        data = json.load(f)

    y_true = data['y_true']
    y_pred = data['y_pred']

    # Index -> Class name
    idx_to_class = {int(k): v for k, v in class_lookup.items()}

    y_true_names = [idx_to_class[i] for i in y_true]
    y_pred_names = [idx_to_class[i] for i in y_pred]

    # ---------------------- #
    # Compute hierarchical accuracy
    # ---------------------- #
    def compute_accuracy(mapping_dict):
        true_grouped = [mapping_dict.get(name, None) for name in y_true_names]
        pred_grouped = [mapping_dict.get(name, None) for name in y_pred_names]

        valid_indices = [i for i, (t, p) in enumerate(zip(true_grouped, pred_grouped)) if t is not None and p is not None]
        correct = sum(1 for i in valid_indices if true_grouped[i] == pred_grouped[i])
        total = len(valid_indices)
        return correct / total if total > 0 else 0.0

    acc_20 = compute_accuracy(view_20_mapping)
    acc_7 = compute_accuracy({k: view_7_mapping.get(v, None) for k, v in view_20_mapping.items()})

    print(f"[20-View Accuracy]: {acc_20:.4f}")
    print(f"[7-View Accuracy]: {acc_7:.4f}")

    save_folder = os.path.dirname(all_preds_json_path)
    save_path = os.path.join(save_folder, "hierarchical_accuracy.json")
    with open(save_path, 'w') as f:
        json.dump({
            "accuracy_20_view": acc_20,
            "accuracy_7_view": acc_7
        }, f, indent=2)
    print(f"[OK] Hierarchical accuracy saved to {save_path}")
