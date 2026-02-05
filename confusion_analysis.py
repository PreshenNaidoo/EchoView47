"""Confusion-matrix utilities for model-vs-expert and expert-vs-expert comparisons."""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

import utils


def _build_display_class_names(class_lookup):
    """Map class indices to display labels and apply minor alias cleanup."""

    class_names = [class_lookup[i] for i in range(len(class_lookup))]
    return ["doppler-ao" if name == "doppler-ao-descending" else name for name in class_names]


def _reorder_psax_class_names(class_names):
    """Reorder PSAX labels to a clinically intuitive sequence for plots."""

    psax_order = [
        "psax-all",
        "psax-tv",
        "psax-av",
        "psax-pv",
        "psax-lv-base",
        "psax-lv-mid",
        "psax-lv-apex",
    ]
    reordered_classes = class_names.copy()
    psax_actual = [name for name in reordered_classes if name.startswith("psax")]
    if not psax_actual:
        return reordered_classes

    psax_start = reordered_classes.index(psax_actual[0])
    for name in psax_actual:
        reordered_classes.remove(name)
    for i, psax_name in enumerate(psax_order):
        if psax_name in class_names:
            reordered_classes.insert(psax_start + i, psax_name)
    return reordered_classes


def _group_indices_from_names(class_names):
    """Return index ranges for high-level view groups used in matrix overlays."""

    group_definitions = {
        "apical": lambda name: name.startswith("a"),
        "plax": lambda name: name.startswith("plax"),
        "psax": lambda name: name.startswith("psax"),
        "doppler": lambda name: name.startswith("doppler"),
        "mmode": lambda name: name.startswith("mmode"),
        "subcostal": lambda name: name.startswith("subcostal"),
        "suprasternal": lambda name: name.startswith("suprasternal"),
    }
    groups = {
        group: [i for i, name in enumerate(class_names) if matcher(name)]
        for group, matcher in group_definitions.items()
    }
    return {group: indices for group, indices in groups.items() if indices}


def generate_confusion_matrices_from_prediction_json(
    prediction_json_path,
    expert1_labels,
    expert2_labels,
    expert3_labels,
    test_files,
    class_lookup,
    output_folder
):
    """Save confusion matrices using all available labels for each pairwise comparison."""

    # Load predictions
    with open(prediction_json_path, 'r') as f:
        preds = json.load(f)

    # Align model predictions to test_files order
    file_to_pred = {fname: int(pred) for fname, pred in zip(preds["files"], preds["y_pred"])}
    model_preds = [file_to_pred.get(f, None) for f in test_files]

    comparisons = [
        ("Model vs Expert1", model_preds, expert1_labels, "model_vs_expert1"),
        ("Model vs Expert2", model_preds, expert2_labels, "model_vs_expert2"),
        ("Model vs Expert3", model_preds, expert3_labels, "model_vs_expert3"),
        ("Expert1 vs Expert2", expert1_labels, expert2_labels, "expert1_vs_expert2"),
        ("Expert1 vs Expert3", expert1_labels, expert3_labels, "expert1_vs_expert3"),
        ("Expert2 vs Expert3", expert2_labels, expert3_labels, "expert2_vs_expert3"),
    ]

    os.makedirs(output_folder, exist_ok=True)

    class_names = _build_display_class_names(class_lookup)
    reordered_classes = _reorder_psax_class_names(class_names)

    sizes_dict = {}

    for title, preds_a, preds_b, fname in comparisons:
        filtered = [(a, b) for a, b in zip(preds_a, preds_b) if a is not None and b is not None]
        if not filtered:
            continue

        sizes_dict[title] = len(filtered)
        a_labels, b_labels = zip(*filtered)
        a_labels = [int(x) for x in a_labels]
        b_labels = [int(x) for x in b_labels]

        cm = confusion_matrix(b_labels, a_labels, labels=list(range(len(class_lookup))))
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.index = ["doppler-ao" if x == "doppler-ao-descending" else x for x in cm_df.index]
        cm_df.columns = ["doppler-ao" if x == "doppler-ao-descending" else x for x in cm_df.columns]
        cm_df = cm_df.loc[reordered_classes, reordered_classes]

        annot = cm_df.copy().astype(str)
        annot[cm_df == 0] = ""

        group_indices = _group_indices_from_names(cm_df.index.tolist())

        # Determine axis labels
        if title.startswith("Model"):
            xlabel = "Predicted"
            ylabel = "Ground Truth"
        elif "vs" in title:
            parts = title.split(" vs ")
            ylabel = parts[0].strip()  # First expert
            xlabel = parts[1].strip()  # Second expert
        else:
            xlabel = "Predicted"
            ylabel = "True"

        fig, ax = plt.subplots(figsize=(18, 18))
        sns.heatmap(
            cm_df,
            annot=annot,
            fmt='',
            cmap='Blues',
            cbar=False,
            xticklabels=cm_df.columns,
            yticklabels=cm_df.index,
            linewidths=0.5,
            linecolor='gray',
            ax=ax,
            annot_kws={"size": 13}
        )

        #ax.set_title(title, fontsize=22)
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.tick_params(axis='x', rotation=90, labelsize=18)
        ax.tick_params(axis='y', labelsize=18)

        for group, idxs in group_indices.items():
            min_idx = min(idxs)
            max_idx = max(idxs)
            width = max_idx - min_idx + 1
            ax.add_patch(plt.Rectangle(
                (min_idx, min_idx), width, width,
                fill=False, edgecolor='black', linewidth=1.8, linestyle='--'
            ))

        plt.tight_layout()
        fig.savefig(os.path.join(output_folder, f"{fname}.png"), dpi=300)
        plt.close(fig)

        cm_df.to_csv(os.path.join(output_folder, f"{fname}.csv"))

    utils.write_dict_to_json(sizes_dict, output_folder, 'lens_dict.json')

def generate_confusion_matrices_intersection_subset(
    prediction_json_path,
    expert1_labels,
    expert2_labels,
    expert3_labels,
    test_files,
    class_lookup,
    output_folder
):
    """Save confusion matrices on the strict intersection where all experts and model are present."""

    # Load predictions
    with open(prediction_json_path, 'r') as f:
        preds = json.load(f)

    # Align model predictions to test_files order
    file_to_pred = {fname: int(pred) for fname, pred in zip(preds["files"], preds["y_pred"])}
    model_preds = [file_to_pred.get(f, None) for f in test_files]

    # Compute valid mask: only include samples where all 4 sources are not None
    valid_mask = [
        all(v is not None for v in [e1, e2, e3, m])
        for e1, e2, e3, m in zip(expert1_labels, expert2_labels, expert3_labels, model_preds)
    ]

    # Apply mask to all label sets and file names
    filtered_e1 = [l for l, keep in zip(expert1_labels, valid_mask) if keep]
    filtered_e2 = [l for l, keep in zip(expert2_labels, valid_mask) if keep]
    filtered_e3 = [l for l, keep in zip(expert3_labels, valid_mask) if keep]
    filtered_model = [l for l, keep in zip(model_preds, valid_mask) if keep]

    comparisons = [
        ("Model vs Expert1", filtered_model, filtered_e1, "model_vs_expert1"),
        ("Model vs Expert2", filtered_model, filtered_e2, "model_vs_expert2"),
        ("Model vs Expert3", filtered_model, filtered_e3, "model_vs_expert3"),
        ("Expert1 vs Expert2", filtered_e1, filtered_e2, "expert1_vs_expert2"),
        ("Expert1 vs Expert3", filtered_e1, filtered_e3, "expert1_vs_expert3"),
        ("Expert2 vs Expert3", filtered_e2, filtered_e3, "expert2_vs_expert3"),
    ]

    os.makedirs(output_folder, exist_ok=True)

    class_names = _build_display_class_names(class_lookup)
    reordered_classes = _reorder_psax_class_names(class_names)

    sizes_dict = {}

    for title, preds_a, preds_b, fname in comparisons:
        sizes_dict[title] = len(preds_a)

        cm = confusion_matrix(preds_b, preds_a, labels=list(range(len(class_lookup))))
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.index = ["doppler-ao" if x == "doppler-ao-descending" else x for x in cm_df.index]
        cm_df.columns = ["doppler-ao" if x == "doppler-ao-descending" else x for x in cm_df.columns]
        cm_df = cm_df.loc[reordered_classes, reordered_classes]

        annot = cm_df.copy().astype(str)
        annot[cm_df == 0] = ""

        group_indices = _group_indices_from_names(cm_df.index.tolist())

        if title.startswith("Model"):
            xlabel = "Predicted"
            ylabel = "Ground Truth"
        elif "vs" in title:
            parts = title.split(" vs ")
            ylabel = parts[0].strip()
            xlabel = parts[1].strip()
        else:
            xlabel = "Predicted"
            ylabel = "True"

        fig, ax = plt.subplots(figsize=(18, 18))
        sns.heatmap(
            cm_df,
            annot=annot,
            fmt='',
            cmap='Blues',
            cbar=False,
            xticklabels=cm_df.columns,
            yticklabels=cm_df.index,
            linewidths=0.5,
            linecolor='gray',
            ax=ax,
            annot_kws={"size": 13}
        )

        #ax.set_title(title, fontsize=22)
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.tick_params(axis='x', rotation=90, labelsize=18)
        ax.tick_params(axis='y', labelsize=18)

        for group, idxs in group_indices.items():
            min_idx = min(idxs)
            max_idx = max(idxs)
            width = max_idx - min_idx + 1
            ax.add_patch(plt.Rectangle(
                (min_idx, min_idx), width, width,
                fill=False, edgecolor='black', linewidth=1.8, linestyle='--'
            ))

        plt.tight_layout()
        fig.savefig(os.path.join(output_folder, f"{fname}.png"), dpi=300)
        plt.close(fig)
        cm_df.to_csv(os.path.join(output_folder, f"{fname}.csv"))

    utils.write_dict_to_json(sizes_dict, output_folder, 'lens_dict.json')
