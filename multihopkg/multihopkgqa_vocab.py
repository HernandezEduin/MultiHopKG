"""Entity/relation index helpers for current MultiHopKGQA datasets.

The latest dataset folders do not always ship the legacy MultiHopKG
``entity2id.txt`` and ``relation2id.txt`` files.  This module creates whichever
mapping is missing directly from the graph data.

Source priority
---------------
1. ``triplets.txt`` when present.
2. Otherwise ``train.txt`` + ``dev.txt`` + ``test.txt`` in that order.

Graph lines follow the current MINERVA/MultiHopKGQA convention::

    head<TAB>relation<TAB>tail

General whitespace is accepted as well. IDs are assigned in first-seen order,
starting at 0. Existing mapping files are never overwritten.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Tuple


ENTITY_INDEX_FILENAME = "entity2id.txt"
RELATION_INDEX_FILENAME = "relation2id.txt"
FULL_GRAPH_FILENAME = "triplets.txt"
SPLIT_GRAPH_FILENAMES = ("train.txt", "dev.txt", "test.txt")


def _graph_source_paths(data_dir: str) -> List[str]:
    """Return graph files in the requested priority/order."""
    full_graph = os.path.join(data_dir, FULL_GRAPH_FILENAME)
    if os.path.isfile(full_graph):
        return [full_graph]

    split_paths = [os.path.join(data_dir, name) for name in SPLIT_GRAPH_FILENAMES]
    missing = [path for path in split_paths if not os.path.isfile(path)]
    if missing:
        expected = ", ".join(SPLIT_GRAPH_FILENAMES)
        missing_names = ", ".join(os.path.basename(path) for path in missing)
        raise FileNotFoundError(
            f"Could not build entity/relation indices in {data_dir!r}: "
            f"{FULL_GRAPH_FILENAME!r} was not found and the fallback requires "
            f"all of [{expected}]. Missing: {missing_names}."
        )
    return split_paths


def _iter_graph_triples(paths: Iterable[str]) -> Iterable[Tuple[str, str, str]]:
    """Yield ``(head, relation, tail)`` triples from graph text files."""
    for path in paths:
        with open(path, "r", encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                fields = re.split(r"\s+", line)
                if len(fields) != 3:
                    raise ValueError(
                        f"Malformed graph line at {path}:{line_number}. Expected "
                        f"exactly 3 whitespace-separated fields in "
                        f"head relation tail order, got {len(fields)}: {line!r}"
                    )
                yield fields[0], fields[1], fields[2]


def build_indices_from_graph(data_dir: str) -> Tuple[Dict[str, int], Dict[str, int], List[str]]:
    """Build first-seen entity/relation mappings from the dataset graph.

    Returns:
        ``(entity2id, relation2id, source_paths)``.
    """
    source_paths = _graph_source_paths(data_dir)
    entity2id: Dict[str, int] = {}
    relation2id: Dict[str, int] = {}

    for head, relation, tail in _iter_graph_triples(source_paths):
        if head not in entity2id:
            entity2id[head] = len(entity2id)
        if tail not in entity2id:
            entity2id[tail] = len(entity2id)
        if relation not in relation2id:
            relation2id[relation] = len(relation2id)

    if not entity2id or not relation2id:
        sources = ", ".join(source_paths)
        raise ValueError(
            f"No usable triples were found while building indices from: {sources}"
        )

    return entity2id, relation2id, source_paths


def _read_index(path: str) -> Dict[str, int]:
    """Read a MultiHopKG ``name<TAB>id`` index file."""
    mapping: Dict[str, int] = {}
    used_ids = set()
    with open(path, "r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = re.split(r"\s+", line)
            if len(fields) != 2:
                raise ValueError(
                    f"Malformed index line at {path}:{line_number}: {line!r}"
                )
            name, raw_idx = fields
            idx = int(raw_idx)
            if name in mapping:
                raise ValueError(f"Duplicate name {name!r} in {path}")
            if idx in used_ids:
                raise ValueError(f"Duplicate ID {idx} in {path}")
            mapping[name] = idx
            used_ids.add(idx)
    return mapping


def _write_index(path: str, mapping: Dict[str, int]) -> None:
    """Write ``name<TAB>id`` ordered by integer ID."""
    with open(path, "w", encoding="utf-8") as file:
        for name, idx in sorted(mapping.items(), key=lambda item: item[1]):
            file.write(f"{name}\t{idx}\n")


def load_or_create_dictionaries(
    data_dir: str,
) -> Tuple[Dict[int, str], Dict[str, int], Dict[int, str], Dict[str, int]]:
    """Load MultiHopKG indices, creating missing files from graph triples.

    Existing index files are preserved. If either mapping is missing, graph data
    is scanned once and only the missing mapping(s) are written.

    Returns the same ordering as ``multihopkg.data_utils.load_dictionaries``::

        id2entity, entity2id, id2relation, relation2id
    """
    entity_path = os.path.join(data_dir, ENTITY_INDEX_FILENAME)
    relation_path = os.path.join(data_dir, RELATION_INDEX_FILENAME)

    entity_exists = os.path.isfile(entity_path)
    relation_exists = os.path.isfile(relation_path)

    generated_entity: Dict[str, int] = {}
    generated_relation: Dict[str, int] = {}
    if not entity_exists or not relation_exists:
        generated_entity, generated_relation, _ = build_indices_from_graph(data_dir)
        if not entity_exists:
            _write_index(entity_path, generated_entity)
        if not relation_exists:
            _write_index(relation_path, generated_relation)

    entity2id = _read_index(entity_path)
    relation2id = _read_index(relation_path)
    id2entity = {idx: name for name, idx in entity2id.items()}
    id2relation = {idx: name for name, idx in relation2id.items()}

    return id2entity, entity2id, id2relation, relation2id
