"""Generate a reusable subject-to-fold manifest for image or tabular pipelines."""

import argparse
from pathlib import Path

import pandas as pd

from datasets import build_dataset_source, build_rotating_cv_split_indices
from train import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} exists; pass --force to replace it")

    config = load_config(args.config)
    config["cv"] = {**config["cv"]}
    config["cv"].pop("manifest_path", None)
    source = build_dataset_source(config)
    state = build_rotating_cv_split_indices(source.items, config)
    labels = {
        source.subject_id(index): int(item["label"])
        for index, item in enumerate(source.items)
    }
    rows = [
        {
            "subject_id": subject,
            "label": labels[subject],
            "fold": fold,
            "n_splits": state["n_splits"],
            "cv_random_seed": config["cv"].get("random_seed"),
        }
        for subject, fold in sorted(state["subject_folds"].items())
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(
        f"Wrote {len(rows)} subjects across {state['n_splits']} folds to {args.output}"
    )


if __name__ == "__main__":
    main()
