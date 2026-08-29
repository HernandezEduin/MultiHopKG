"""Dataset preparation entry point for MultiHopKG.

The current MultiHopKGQA workflow for Kinship and MQuAKE-ST lives here rather
than in a separate preprocessing script.  The underlying implementation is the
canonical ``multihopkg.data_utils`` API.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Callable, Dict

from multihopkg import data_utils
from multihopkg.logging import setup_logger
from multihopkg.utils.setup import get_git_root


logger = logging.getLogger(__name__)


def process_traditional_kb_data(
    data_dir: str,
    test: bool,
    model: str,
    add_reverse_relations: bool,
) -> None:
    """Run the historical KB-environment preparation path."""
    raw_kb_path = os.path.join(data_dir, "raw.kb")
    train_path = data_utils.get_train_path(data_dir, test, model)
    dev_path = os.path.join(data_dir, "dev.triples")
    test_path = os.path.join(data_dir, "test.triples")
    data_utils.prepare_kb_envrioment(
        raw_kb_path,
        train_path,
        dev_path,
        test_path,
        test,
        add_reverse_relations,
    )


def process_qa_data(
    data_dir: str,
    raw_qa_path: str,
    cache_path: str,
    question_tokenizer: str,
    seed: int = 42,
    force_recompute: bool = False,
    override_split: bool = True,
) -> None:
    """Preprocess current Kinship/MQuAKE-ST MultiHopKGQA data.

    ``data_utils.load_dictionaries`` automatically creates missing
    ``entity2id.txt`` / ``relation2id.txt`` files from ``triplets.txt`` or,
    when it is absent, from ``train.txt`` + ``dev.txt`` + ``test.txt``.
    """
    id2entity, entity2id, id2relation, relation2id = data_utils.load_dictionaries(
        data_dir
    )
    logger.info(
        "Loaded dictionaries: %d entities, %d relations",
        len(id2entity),
        len(id2relation),
    )

    train_df, dev_df, test_df, metadata = data_utils.load_qa_data(
        cached_metadata_path=cache_path,
        raw_QAData_path=raw_qa_path,
        question_tokenizer_name=question_tokenizer,
        answer_tokenizer_name=None,
        entity2id=entity2id,
        relation2id=relation2id,
        logger=logger,
        force_recompute=force_recompute,
        override_split=override_split,
        supervised=True,
        seed=seed,
    )

    logger.info("MultiHopKGQA preprocessing succeeded")
    logger.info("Schema: %s", metadata.get("schema", "legacy"))
    logger.info("Multi-answer: %s", metadata.get("is_multi_answer", False))
    logger.info(
        "Train/dev/test: %d/%d/%d", len(train_df), len(dev_df), len(test_df)
    )
    logger.info("Columns: %s", list(train_df.columns))
    logger.info("Cached splits: %s", metadata["saved_paths"])

    if len(train_df):
        logger.info("Example processed training row: %s", train_df.iloc[0].to_dict())


def all_arguments(valid_operations: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MultiHopKG datasets")
    repo_root = get_git_root()
    assert repo_root is not None, "Could not find the root of the git repository"

    parser.add_argument(
        "--operation",
        required=True,
        choices=valid_operations,
        help="Preparation operation to perform",
    )
    parser.add_argument(
        "--logging_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    # Common dataset directory. For MultiHopKGQA this is where graph triples and
    # entity/relation dictionaries live.
    parser.add_argument(
        "--data_dir",
        default=os.path.join(repo_root, "data"),
        help="Dataset directory containing graph data",
    )

    # Current MultiHopKGQA QA preprocessing arguments.
    parser.add_argument(
        "--raw_qa",
        "--raw_QAPathData_path",
        dest="raw_qa",
        default=None,
        help="Raw MultiHopKGQA CSV (Kinship or MQuAKE-ST)",
    )
    parser.add_argument(
        "--cache",
        "--cached_QAPathData_path",
        dest="cache",
        default=None,
        help="QA preprocessing metadata JSON cache path",
    )
    parser.add_argument(
        "--question_tokenizer",
        "--text_tokenizer",
        dest="question_tokenizer",
        default="bert-base-uncased",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument(
        "--ignore_split_labels",
        action="store_true",
        help="Ignore dataset SplitLabel values and generate splits automatically",
    )

    # Historical KB preparation arguments.
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--model", default="point")
    parser.add_argument("--add_reverse_relations", action="store_true")

    args = parser.parse_args()

    if args.operation in {"process_qa_data", "all"}:
        if not args.raw_qa:
            parser.error("--raw_qa is required for process_qa_data/all")
        if not args.cache:
            parser.error("--cache is required for process_qa_data/all")
        cache_dir = os.path.dirname(args.cache)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    return args


def run_operation(args: argparse.Namespace) -> None:
    if args.operation == "process_traditional_kb_data":
        process_traditional_kb_data(
            args.data_dir, args.test, args.model, args.add_reverse_relations
        )
        return

    if args.operation == "process_qa_data":
        process_qa_data(
            data_dir=args.data_dir,
            raw_qa_path=args.raw_qa,
            cache_path=args.cache,
            question_tokenizer=args.question_tokenizer,
            seed=args.seed,
            force_recompute=args.force_recompute,
            override_split=not args.ignore_split_labels,
        )
        return

    if args.operation == "all":
        process_traditional_kb_data(
            args.data_dir, args.test, args.model, args.add_reverse_relations
        )
        process_qa_data(
            data_dir=args.data_dir,
            raw_qa_path=args.raw_qa,
            cache_path=args.cache,
            question_tokenizer=args.question_tokenizer,
            seed=args.seed,
            force_recompute=args.force_recompute,
            override_split=not args.ignore_split_labels,
        )
        return

    raise ValueError(f"Invalid operation: {args.operation}")


def main() -> None:
    valid_operations = ["process_traditional_kb_data", "process_qa_data", "all"]
    args = all_arguments(valid_operations)

    global logger
    logger = setup_logger(
        logger_name=os.path.basename(__file__).replace(".py", ""),
        logging_level=args.logging_level,
    )
    run_operation(args)


if __name__ == "__main__":
    main()
