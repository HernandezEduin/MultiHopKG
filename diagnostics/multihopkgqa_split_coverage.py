"""Report semantic/path overlap across MultiHopKGQA train/dev/test splits."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter

import pandas as pd

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--qa_path", required=True)
    return p.parse_args()


def parse_chain(row):
    value = row.get("Path-Key")
    if pd.notna(value):
        text = str(value).strip()
        if text.startswith("["):
            parsed = ast.literal_eval(text)
            return tuple(str(x).strip() for x in parsed)
        return tuple(x.strip() for x in text.split("->") if x.strip())
    paths = ast.literal_eval(row["Paths"]) if isinstance(row["Paths"], str) else row["Paths"]
    return tuple(str(step[1]) for step in paths)


def main():
    args = parse_args()
    df = pd.read_csv(args.qa_path)
    if "SplitLabel" not in df.columns:
        raise ValueError("QA CSV must contain SplitLabel")
    df = df.copy()
    df["_chain"] = df.apply(parse_chain, axis=1)
    df["_relations"] = df["_chain"].map(frozenset)

    splits = {name: part.reset_index(drop=True) for name, part in df.groupby("SplitLabel")}
    if "train" not in splits:
        raise ValueError("SplitLabel must contain train")

    train = splits["train"]
    train_chains = set(train["_chain"])
    train_relations = set().union(*train["_relations"]) if len(train) else set()

    result = {
        "rows": {name: int(len(part)) for name, part in splits.items()},
        "unique_chains": {name: int(part["_chain"].nunique()) for name, part in splits.items()},
        "unique_relations": {
            name: int(len(set().union(*part["_relations"]))) if len(part) else 0
            for name, part in splits.items()
        },
        "chain_length_counts": {
            name: dict(Counter(str(len(c)) for c in part["_chain"]))
            for name, part in splits.items()
        },
    }

    for name in ("dev", "test"):
        if name not in splits:
            continue
        part = splits[name]
        chain_seen = part["_chain"].map(lambda x: x in train_chains)
        all_rel_seen = part["_relations"].map(lambda rs: rs.issubset(train_relations))
        result[name] = {
            "full_chain_seen_in_train_fraction": float(chain_seen.mean()),
            "full_chain_unseen_but_all_relations_seen_fraction": float((~chain_seen & all_rel_seen).mean()),
            "contains_unseen_relation_fraction": float((~all_rel_seen).mean()),
            "unique_unseen_chains": int(len(set(part.loc[~chain_seen, "_chain"]))),
        }

    overlap_fields = [
        field for field in (
            "Question-Number", "Question", "Source-Entity", "Answer-Entity", "Path-Key"
        ) if field in df.columns
    ]
    overlaps = {}
    for name in ("dev", "test"):
        if name not in splits:
            continue
        overlaps[name] = {}
        for field in overlap_fields:
            train_values = set(train[field].astype(str))
            overlaps[name][f"{field}_seen_in_train_fraction"] = float(
                splits[name][field].astype(str).isin(train_values).mean()
            )
    result["field_overlap"] = overlaps

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
