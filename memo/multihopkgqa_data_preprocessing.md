# MultiHopKGQA dataset preprocessing port

## Purpose

This branch prepares MultiHopKG for the latest MultiHopKGQA versions of **Kinship** and **MQuAKE-ST** using the dataset/schema logic maintained in MINERVA.

Source of truth:

- `HernandezEduin/MINERVA:master/code/data/data_utils.py`

Target repository:

- `HernandezEduin/MultiHopKG`
- branch: `feature/multihopkgqa-data-preprocessing`
- base: `master`

The branch is intentionally independent of `temporary/protate-navigation-patches` so that it can later be merged into both `master` and the pRotatE branch.

## Canonical implementation location

The modern QA preprocessing implementation lives directly in:

```text
multihopkg/data_utils.py
```

The two public preprocessing functions are:

```python
process_and_cache_triviaqa_data(...)
process_and_cache_supervised_triviaqa_data(...)
```

The old misspelled name:

```python
process_and_cache_suprvised_triviaqa_data
```

is retained only as a compatibility alias so older call sites do not break immediately.

`load_qa_data()` routes supervised preprocessing through the corrected `process_and_cache_supervised_triviaqa_data()` function.

Unrelated historical utility functions from the old large `data_utils.py` implementation remain available through the internal `_legacy_data_utils.py` compatibility module. The modern MultiHopKGQA preprocessing itself is **not** stored in a parallel standalone module.

## Current MultiHopKGQA schema support

`process_and_cache_triviaqa_data()` supports:

- `Question-Number`
- `Question`
- `Question-Paraphrased`
- `Question-Disambiguated`
- `Source`
- `Source-Entity`
- `Answer`
- `Answer-Entity`
- `Paths`
- `Paths-Label`
- `Path-Key`
- `Hops`
- `SplitLabel`

### Multi-answer data

When `Answer-Entity` contains serialized lists, for example:

```text
['Q1', 'Q2', 'Q3']
```

the processed value is a list of KGE IDs:

```python
[entity2id['Q1'], entity2id['Q2'], entity2id['Q3']]
```

Metadata records:

```json
"is_multi_answer": true
```

This correctly preprocesses current multi-answer MQuAKE-ST data, although the downstream MultiHopKG environment/evaluator still needs a separate multi-answer compatibility update before end-to-end MQuAKE-ST navigation is semantically correct.

### Paths

Entity-level paths such as:

```python
[['e0', 'r0', 'e1'], ['e1', 'r1', 'e2']]
```

are mapped to integer triples:

```python
[[entity2id['e0'], relation2id['r0'], entity2id['e1']],
 [entity2id['e1'], relation2id['r1'], entity2id['e2']]]
```

### Path-Key

Both relation-chain strings and serialized lists are accepted:

```text
r0->r1->r2
```

or:

```python
['r0', 'r1', 'r2']
```

and converted to relation IDs.

### Question variants

`Question`, `Question-Paraphrased`, and `Question-Disambiguated` are tokenized with the HuggingFace question tokenizer.

`paraphrase2question()` remains available for workflows that intentionally want one row per paraphrase.

## Split behavior

When meaningful `SplitLabel` values exist and `override_split=True`, the dataset-defined splits are respected.

If explicit `train`, `dev`, and `test` labels exist, all three remain independent.

If no explicit dev split exists, current MINERVA-compatible behavior remains:

- held-out set `<100` examples: `dev == test`;
- otherwise held-out data is split into dev/test.

This is useful for compatibility/debugging but should not be used for final unbiased reporting when `dev == test`.

## Automatic entity/relation dictionaries

`multihopkg.data_utils.load_dictionaries(data_dir)` now creates missing dictionaries automatically.

Expected output files:

```text
entity2id.txt
relation2id.txt
```

Source priority:

1. `triplets.txt`, if present;
2. otherwise `train.txt`, then `dev.txt`, then `test.txt`.

Graph lines follow:

```text
head    relation    tail
```

General whitespace is accepted.

IDs are assigned in **first-seen order starting at 0**. For example:

```text
e2  r1  e0
e0  r2  e3
```

creates:

```text
entity2id.txt
e2  0
e0  1
e3  2

relation2id.txt
r1  0
r2  1
```

Existing mapping files are never overwritten. If only one mapping is absent, only that file is generated.

If `triplets.txt` is absent, all three split graph files are required so an incomplete vocabulary is not silently generated.

## Data preparation entry point

There is no separate `preprocess_multihopkgqa.py` script anymore.

The preprocessing CLI is integrated into:

```text
data_preparation.py
```

Use:

```bash
python data_preparation.py \
    --operation process_qa_data \
    --data_dir data/KinshipHinton \
    --raw_qa data/KinshipHinton/kinship_hinton_qa_2hop.csv \
    --cache ./.cache/itl/kinship_hinton_qa_2hop_v2.json \
    --question_tokenizer bert-base-uncased \
    --force_recompute
```

This command:

1. loads or creates entity/relation dictionaries;
2. preprocesses the QA CSV through `multihopkg.data_utils`;
3. writes cached parquet splits and metadata;
4. logs schema, multi-answer status, split sizes, columns, and an example row.

`--operation all` also remains available when the historical KB preparation step is desired first.

## Tests

Schema tests:

```bash
python -m unittest tests.test_multihopkgqa_data -v
```

Vocabulary-generation tests:

```bash
python -m unittest tests.test_multihopkgqa_vocab -v
```

Together:

```bash
python -m unittest \
    tests.test_multihopkgqa_data \
    tests.test_multihopkgqa_vocab \
    -v
```

The tests cover:

- single-answer Kinship compatibility;
- latest multi-answer MQuAKE-style fields;
- corrected supervised preprocessing function name;
- entity/relation/path/Path-Key mapping;
- explicit split handling;
- `triplets.txt` dictionary source priority;
- `train.txt + dev.txt + test.txt` fallback;
- zero-based first-seen IDs;
- preserving an existing mapping file;
- rejecting incomplete fallback graph splits.

## Remaining downstream boundary: multi-answer navigation

The newest MQuAKE-ST format can produce `Answer-Entity` as `List[int]`.

The preprocessing now represents this correctly, but the current MultiHopKG environment and evaluator still assume a single answer ID in several locations. That should be handled separately by storing one answer set per question and counting a rollout as successful when its retrieved entity belongs to that set, following current MINERVA behavior.

Do **not** flatten multi-answer questions into independent fake single-answer questions merely to satisfy the old environment, because that changes the QA semantics and can distort relation-chain/path supervision.

## Merge plan

After local tests against the current real datasets:

1. merge this branch into `master`;
2. merge the same branch into `temporary/protate-navigation-patches`;
3. add proper multi-answer environment/reward/evaluation support;
4. rerun Kinship to verify no regression;
5. run MQuAKE-ST end-to-end.
