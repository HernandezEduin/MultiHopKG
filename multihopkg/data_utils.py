"""Data processing utilities.

Historical MultiHopKG helpers are re-exported from ``_legacy_data_utils`` while
current MultiHopKGQA preprocessing for Kinship and MQuAKE-ST is implemented
here directly.  This keeps ``multihopkg.data_utils`` as the canonical API used
by training and data preparation.
"""

from __future__ import annotations

import ast
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, PreTrainedTokenizer

# Preserve unrelated historical utility functions/constants.
from multihopkg._legacy_data_utils import *  # noqa: F401,F403
from multihopkg.itl_typing import DFSplit
from multihopkg.multihopkgqa_vocab import load_or_create_dictionaries
from multihopkg.utils.setup import get_git_root


def _parse_list_cell(value: Any) -> List[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a serialized list, got {type(parsed).__name__}: {value}")
    return parsed


def _is_list_column(series: pd.Series) -> bool:
    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    def is_list(value: Any) -> bool:
        if isinstance(value, (list, tuple, np.ndarray)):
            return True
        if not isinstance(value, str):
            return False
        text = value.strip()
        return text.startswith("[") and text.endswith("]")

    return bool(non_null.map(is_list).all())


def _relative_to_repo(path: str) -> str:
    repo_root = get_git_root()
    if repo_root is None:
        return path
    try:
        return os.path.relpath(path, repo_root)
    except ValueError:
        return path


def _resolve_cached_path(metadata_path: str, saved_path: str) -> str:
    if os.path.isabs(saved_path):
        return saved_path
    repo_root = get_git_root()
    if repo_root is not None:
        repo_candidate = os.path.join(repo_root, saved_path)
        if os.path.exists(repo_candidate):
            return repo_candidate
    return os.path.join(os.path.dirname(metadata_path), saved_path)


def _restore_nested_lists(df: pd.DataFrame) -> pd.DataFrame:
    return df.map(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)


def paraphrase2question(df: pd.DataFrame) -> pd.DataFrame:
    """Expand ``Question-Paraphrased`` into one row per paraphrase."""
    if "Question-Paraphrased" not in df.columns:
        raise ValueError("DataFrame does not contain 'Question-Paraphrased'")
    out = df.copy()
    out["Question-Paraphrased"] = out["Question-Paraphrased"].map(
        lambda value: _parse_list_cell(value)
        if not isinstance(value, list)
        else value
    )
    empty = out["Question-Paraphrased"].map(len) == 0
    out.loc[empty, "Question-Paraphrased"] = out.loc[empty, "Question"].map(lambda q: [q])
    out = out.explode("Question-Paraphrased", ignore_index=True)
    out["Question"] = out["Question-Paraphrased"]
    return out


def process_and_cache_triviaqa_data(
    raw_QAData_path: str,
    cached_toked_qatriples_metadata_path: str,
    question_tokenizer: PreTrainedTokenizer,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    seed: Optional[int] = None,
    override_split: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Tuple[DFSplit, Dict[str, Any]]:
    """Process current MultiHopKGQA Kinship/MQuAKE-ST CSV data.

    Supported modern fields include ``Question-Number``, paraphrased and
    disambiguated questions, readable Source/Answer labels, multi-answer
    ``Answer-Entity`` lists, entity-level ``Paths``, ``Paths-Label``, relation
    chain ``Path-Key``, ``Hops`` and ``SplitLabel``.
    """
    csv_df = pd.read_csv(raw_QAData_path)
    required = {"Question", "Source-Entity", "Answer-Entity"}
    missing = required.difference(csv_df.columns)
    if missing:
        raise ValueError(f"Missing required QA columns: {sorted(missing)}")

    is_multi_answer = _is_list_column(csv_df["Answer-Entity"])

    question_number = (
        csv_df["Question-Number"]
        if "Question-Number" in csv_df.columns
        else pd.Series(range(len(csv_df)), name="Question-Number")
    )
    source_label = (
        csv_df["Source"]
        if "Source" in csv_df.columns
        else csv_df["Source-Entity"].rename("Source")
    )

    if is_multi_answer:
        answer_entities = csv_df["Answer-Entity"].map(_parse_list_cell)
        answer_label = (
            csv_df["Answer"].map(_parse_list_cell)
            if "Answer" in csv_df.columns
            else answer_entities.rename("Answer")
        )
    else:
        answer_entities = csv_df["Answer-Entity"]
        answer_label = (
            csv_df["Answer"]
            if "Answer" in csv_df.columns
            else csv_df["Answer-Entity"].rename("Answer")
        )

    questions = csv_df["Question"].map(
        lambda q: question_tokenizer.encode(str(q), add_special_tokens=False)
    ).rename("Question")
    source_entities = csv_df["Source-Entity"].map(entity2id.__getitem__).rename("Source-Entity")
    if is_multi_answer:
        mapped_answers = answer_entities.map(
            lambda answers: [entity2id[a] for a in answers]
        ).rename("Answer-Entity")
    else:
        mapped_answers = answer_entities.map(entity2id.__getitem__).rename("Answer-Entity")

    columns: List[pd.Series] = [
        question_number.rename("Question-Number"),
        questions,
        source_entities,
        mapped_answers,
        source_label.rename("Source"),
        answer_label.rename("Answer"),
    ]

    if "Question-Paraphrased" in csv_df.columns:
        paraphrases = csv_df["Question-Paraphrased"].map(_parse_list_cell).map(
            lambda values: [
                question_tokenizer.encode(str(q), add_special_tokens=False)
                for q in values
            ]
        ).rename("Question-Paraphrased")
        columns.append(paraphrases)

    if "Question-Disambiguated" in csv_df.columns:
        disambiguated = csv_df["Question-Disambiguated"].map(
            lambda q: question_tokenizer.encode(str(q), add_special_tokens=False)
        ).rename("Question-Disambiguated")
        columns.append(disambiguated)

    if "Paths" in csv_df.columns:
        paths = csv_df["Paths"].map(_parse_list_cell).map(
            lambda path: [
                [entity2id[head], relation2id[rel], entity2id[tail]]
                for head, rel, tail in path
            ]
        ).rename("Paths")
        columns.append(paths)

    if "Paths-Label" in csv_df.columns:
        columns.append(csv_df["Paths-Label"].rename("Paths-Label"))

    if "Path-Key" in csv_df.columns:
        def map_path_key(value: Any) -> List[int]:
            if isinstance(value, (list, tuple, np.ndarray)):
                relations = list(value)
            else:
                text = str(value).strip()
                relations = (
                    _parse_list_cell(text)
                    if text.startswith("[") and text.endswith("]")
                    else [rel for rel in text.split("->") if rel]
                )
            return [relation2id[rel] for rel in relations]

        columns.append(csv_df["Path-Key"].map(map_path_key).rename("Path-Key"))

    if "Hops" in csv_df.columns:
        columns.append(csv_df["Hops"].rename("Hops"))
    if "SplitLabel" in csv_df.columns:
        columns.append(csv_df["SplitLabel"].rename("SplitLabel"))

    new_df = pd.concat(columns, axis=1)
    new_df = new_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    has_split_labels = (
        "SplitLabel" in new_df.columns
        and new_df["SplitLabel"].notna().any()
        and not new_df["SplitLabel"].fillna("").eq("").all()
    )
    dev_splitted = False
    if override_split and has_split_labels:
        train_df = new_df[new_df["SplitLabel"] == "train"].reset_index(drop=True)
        labels = set(new_df["SplitLabel"].dropna().astype(str))
        if {"dev", "test"}.issubset(labels):
            dev_df = new_df[new_df["SplitLabel"] == "dev"].reset_index(drop=True)
            test_df = new_df[new_df["SplitLabel"] == "test"].reset_index(drop=True)
            dev_splitted = True
            if logger:
                logger.info("Using SplitLabel column for train/dev/test splitting")
        else:
            test_df = new_df[new_df["SplitLabel"] != "train"].reset_index(drop=True)
    else:
        train_df, test_df = train_test_split(new_df, test_size=0.2, random_state=seed)
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    if not dev_splitted:
        if len(test_df) < 100:
            dev_df = test_df.copy()
            if logger:
                logger.warning("Test set too small (<100), using it as dev set")
        else:
            dev_df, test_df = train_test_split(test_df, test_size=0.5, random_state=seed)
            dev_df = dev_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)

    cache_dir = os.path.dirname(cached_toked_qatriples_metadata_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    timestamp = str(int(datetime.now().timestamp()))
    base = (
        cached_toked_qatriples_metadata_path[:-5]
        if cached_toked_qatriples_metadata_path.endswith(".json")
        else cached_toked_qatriples_metadata_path
    )
    absolute_split_paths = {
        split_name: f"{base}_Split-{split_name}_date-{timestamp}.parquet"
        for split_name in ("train", "dev", "test")
    }
    for split_name, frame in {
        "train": train_df,
        "dev": dev_df,
        "test": test_df,
    }.items():
        frame.to_parquet(absolute_split_paths[split_name], index=False)

    metadata: Dict[str, Any] = {
        "schema": "MultiHopKGQA-v2",
        "question_tokenizer": question_tokenizer.name_or_path,
        "question_number_column": "Question-Number",
        "question_column": "Question",
        "question_paraphrased_column": (
            "Question-Paraphrased" if "Question-Paraphrased" in csv_df.columns else None
        ),
        "question_disambiguated_column": (
            "Question-Disambiguated" if "Question-Disambiguated" in csv_df.columns else None
        ),
        "source_label_column": "Source",
        "source_entities_column": "Source-Entity",
        "answer_label_column": "Answer",
        "answer_entity_column": "Answer-Entity",
        "paths_column": "Paths" if "Paths" in csv_df.columns else None,
        "paths_label_column": "Paths-Label" if "Paths-Label" in csv_df.columns else None,
        "path_keys_column": "Path-Key" if "Path-Key" in csv_df.columns else None,
        "hops_column": "Hops" if "Hops" in csv_df.columns else None,
        "splitLabel_column": "SplitLabel" if "SplitLabel" in csv_df.columns else None,
        "is_multi_answer": is_multi_answer,
        "date_processed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "saved_paths": {
            split_name: _relative_to_repo(path)
            for split_name, path in absolute_split_paths.items()
        },
        "timestamp": timestamp,
    }
    with open(cached_toked_qatriples_metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return DFSplit(train=train_df, dev=dev_df, test=test_df), metadata


def process_and_cache_supervised_triviaqa_data(
    raw_QAData_path: str,
    cached_toked_qatriples_metadata_path: str,
    question_tokenizer: PreTrainedTokenizer,
    answer_tokenizer: Optional[PreTrainedTokenizer],
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    override_split: bool = True,
    logger: Optional[logging.Logger] = None,
    seed: Optional[int] = None,
) -> Tuple[DFSplit, Dict[str, Any]]:
    """Supervised compatibility wrapper around ``process_and_cache_triviaqa_data``."""
    del answer_tokenizer
    return process_and_cache_triviaqa_data(
        raw_QAData_path=raw_QAData_path,
        cached_toked_qatriples_metadata_path=cached_toked_qatriples_metadata_path,
        question_tokenizer=question_tokenizer,
        entity2id=entity2id,
        relation2id=relation2id,
        seed=seed,
        override_split=override_split,
        logger=logger,
    )


# Backward compatibility only. New code uses the correctly spelled name above.
process_and_cache_suprvised_triviaqa_data = process_and_cache_supervised_triviaqa_data


def load_qa_data(
    cached_metadata_path: str,
    raw_QAData_path: str,
    question_tokenizer_name: str,
    answer_tokenizer_name: Optional[str],
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    logger: Optional[logging.Logger],
    force_recompute: bool = False,
    override_split: bool = True,
    supervised: bool = True,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load cached QA data or run the canonical preprocessing implementation."""
    if os.path.exists(cached_metadata_path) and not force_recompute:
        with open(cached_metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        train_df = pd.read_parquet(_resolve_cached_path(cached_metadata_path, metadata["saved_paths"]["train"]))
        dev_df = pd.read_parquet(_resolve_cached_path(cached_metadata_path, metadata["saved_paths"]["dev"]))
        test_df = pd.read_parquet(_resolve_cached_path(cached_metadata_path, metadata["saved_paths"]["test"]))
        return (
            _restore_nested_lists(train_df),
            _restore_nested_lists(dev_df),
            _restore_nested_lists(test_df),
            metadata,
        )

    question_tokenizer = AutoTokenizer.from_pretrained(question_tokenizer_name)
    if supervised:
        split, metadata = process_and_cache_supervised_triviaqa_data(
            raw_QAData_path=raw_QAData_path,
            cached_toked_qatriples_metadata_path=cached_metadata_path,
            question_tokenizer=question_tokenizer,
            answer_tokenizer=None,
            entity2id=entity2id,
            relation2id=relation2id,
            override_split=override_split,
            logger=logger,
            seed=seed,
        )
    else:
        # Preserve the historical unsupervised path until it is separately modernized.
        answer_tokenizer = AutoTokenizer.from_pretrained(
            answer_tokenizer_name or question_tokenizer_name
        )
        split, metadata = process_and_cache_unsuprvised_triviaqa_data(
            raw_QAData_path,
            cached_metadata_path,
            question_tokenizer,
            answer_tokenizer,
            entity2id,
            relation2id,
            override_split=override_split,
        )
    return split.train, split.dev, split.test, metadata


def load_dictionaries(
    raw_data_path: str,
) -> Tuple[Dict[int, str], Dict[str, int], Dict[int, str], Dict[str, int]]:
    """Load dictionaries and create missing zero-based mappings automatically."""
    return load_or_create_dictionaries(raw_data_path)
