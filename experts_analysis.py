"""Expert-label loading helpers used by the main training/evaluation script."""

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

ExpertLabelBundle = Tuple[List, List, List, List, List, List, List, List, List, List]

def _load_expert_labels_from_json(path: str) -> Optional[ExpertLabelBundle]:
    """Load expert label lists from a JSON file if it exists."""

    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    required_keys = [
        "expert1_labels",
        "expert2_labels",
        "expert3_labels",
        "gt_vote_labels",
        "majority_vote_files",
        "all_agree_files",
        "no_maj_vote_files",
        "expert1_other_labels",
        "expert2_other_labels",
        "expert3_other_labels",
    ]
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")

    return (
        payload["expert1_labels"],
        payload["expert2_labels"],
        payload["expert3_labels"],
        payload["gt_vote_labels"],
        payload["majority_vote_files"],
        payload["all_agree_files"],
        payload["no_maj_vote_files"],
        payload["expert1_other_labels"],
        payload["expert2_other_labels"],
        payload["expert3_other_labels"],
    )


def _fallback_to_dataset_labels(
    test_files: Sequence[str], test_labels: Sequence[int], save_dir: Optional[str] = None
) -> ExpertLabelBundle:
    """Return a safe fallback bundle by treating dataset labels as all expert labels."""

    test_files = list(test_files)
    test_labels = [int(x) for x in test_labels]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        note_path = os.path.join(save_dir, "expert_labels_fallback_note.json")
        with open(note_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "note": (
                        "Expert label file was not found. Falling back to dataset labels for all experts. "
                        "Add an expert JSON file to replace this fallback."
                    ),
                    "num_files": len(test_files),
                },
                handle,
                indent=2,
            )

    return (
        list(test_labels),  # expert1 labels
        list(test_labels),  # expert2 labels
        list(test_labels),  # expert3 labels
        list(test_labels),  # majority vote labels
        list(test_files),   # majority vote files
        list(test_files),   # all experts agree
        [],                 # no majority vote files
        [],                 # expert1 labels for no-majority files
        [],                 # expert2 labels for no-majority files
        [],                 # expert3 labels for no-majority files
    )


def get_expert_labels(
    test_files: Sequence[str],
    test_labels: Sequence[int],
    class_lookup: Optional[Dict[int, str]] = None,
    reversed_class_lookup: Optional[Dict[str, int]] = None,
    save_dir: Optional[str] = None,
    fallback_strategy: str = "expert1",
    create_pdf: bool = False,
    expert_labels_json: Optional[str] = None,
):
    """Load expert annotations from disk, with an explicit dataset-label fallback."""

    del class_lookup
    del reversed_class_lookup
    del fallback_strategy
    del create_pdf

    candidates = []
    if expert_labels_json:
        candidates.append(expert_labels_json)
    candidates.extend(
        [
            "expert_labels.json",
            "view_classification_experts_labels.json",
            os.path.join("results_plots", "expert_labels.json"),
        ]
    )

    for path in candidates:
        try:
            loaded = _load_expert_labels_from_json(path)
            if loaded is not None:
                print(f"[INFO] Loaded expert labels from: {path}")
                return loaded
        except Exception as exc:
            print(f"[WARN] Failed to load expert labels from {path}: {exc}")

    print("[WARN] No expert label JSON found. Using dataset labels as fallback.")
    return _fallback_to_dataset_labels(test_files=test_files, test_labels=test_labels, save_dir=save_dir)
