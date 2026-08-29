import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from multihopkg.data_utils import (
    process_and_cache_supervised_triviaqa_data,
    process_and_cache_triviaqa_data,
)


class DummyTokenizer:
    name_or_path = "dummy-tokenizer"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [len(token) for token in str(text).split()]


class TestMultiHopKGQAPreprocessing(unittest.TestCase):
    def setUp(self):
        self.entity2id = {
            "e0": 0,
            "e1": 1,
            "e2": 2,
            "e3": 3,
            "e4": 4,
        }
        self.relation2id = {"r0": 0, "r1": 1, "r2": 2}
        self.tokenizer = DummyTokenizer()

    def _run(self, frame):
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        csv_path = os.path.join(tempdir.name, "qa.csv")
        metadata_path = os.path.join(tempdir.name, "qa_cache.json")
        frame.to_csv(csv_path, index=False)

        with mock.patch.object(pd.DataFrame, "to_parquet", autospec=True):
            split, metadata = process_and_cache_triviaqa_data(
                raw_QAData_path=csv_path,
                cached_toked_qatriples_metadata_path=metadata_path,
                question_tokenizer=self.tokenizer,
                entity2id=self.entity2id,
                relation2id=self.relation2id,
                seed=42,
                override_split=True,
            )
        return split, metadata

    def test_single_answer_kinship_compatibility(self):
        frame = pd.DataFrame(
            {
                "Question": ["who follows e0", "who follows e1", "who follows e2"],
                "Source-Entity": ["e0", "e1", "e2"],
                "Answer-Entity": ["e1", "e2", "e3"],
                "Answer": ["one", "two", "three"],
                "Paths": [
                    "[['e0', 'r0', 'e1']]",
                    "[['e1', 'r0', 'e2']]",
                    "[['e2', 'r0', 'e3']]",
                ],
                "Hops": [1, 1, 1],
                "SplitLabel": ["train", "dev", "test"],
            }
        )

        split, metadata = self._run(frame)

        self.assertFalse(metadata["is_multi_answer"])
        self.assertEqual(len(split.train), 1)
        self.assertEqual(len(split.dev), 1)
        self.assertEqual(len(split.test), 1)
        self.assertEqual(split.train.iloc[0]["Source-Entity"], 0)
        self.assertEqual(split.dev.iloc[0]["Answer-Entity"], 2)
        self.assertEqual(split.test.iloc[0]["Paths"], [[2, 0, 3]])
        self.assertIn("Question-Number", split.train.columns)
        self.assertIn("Source", split.train.columns)
        self.assertEqual(metadata["schema"], "MultiHopKGQA-v2")

    def test_latest_multi_answer_mquake_fields(self):
        frame = pd.DataFrame(
            {
                "Question-Number": [10, 11, 12],
                "Question": ["question ten", "question eleven", "question twelve"],
                "Question-Paraphrased": [
                    "['ten paraphrase', 'another ten']",
                    "['eleven paraphrase']",
                    "[]",
                ],
                "Question-Disambiguated": [
                    "question ten entity",
                    "question eleven entity",
                    "question twelve entity",
                ],
                "Source": ["source zero", "source one", "source two"],
                "Source-Entity": ["e0", "e1", "e2"],
                "Answer": [
                    "['answer two', 'answer three']",
                    "['answer three']",
                    "['answer four']",
                ],
                "Answer-Entity": ["['e2', 'e3']", "['e3']", "['e4']"],
                "Paths": [
                    "[['e0', 'r0', 'e1'], ['e1', 'r1', 'e2']]",
                    "[['e1', 'r1', 'e2'], ['e2', 'r2', 'e3']]",
                    "[['e2', 'r2', 'e3'], ['e3', 'r0', 'e4']]",
                ],
                "Paths-Label": ["p10", "p11", "p12"],
                "Path-Key": ["r0->r1", "r1->r2", "['r2', 'r0']"],
                "Hops": [2, 2, 2],
                "SplitLabel": ["train", "dev", "test"],
            }
        )

        split, metadata = self._run(frame)

        self.assertTrue(metadata["is_multi_answer"])
        self.assertEqual(metadata["path_keys_column"], "Path-Key")
        self.assertEqual(metadata["question_paraphrased_column"], "Question-Paraphrased")
        self.assertEqual(metadata["question_disambiguated_column"], "Question-Disambiguated")

        train = split.train.iloc[0]
        self.assertEqual(train["Question-Number"], 10)
        self.assertEqual(train["Answer-Entity"], [2, 3])
        self.assertEqual(train["Path-Key"], [0, 1])
        self.assertEqual(train["Paths"], [[0, 0, 1], [1, 1, 2]])
        self.assertEqual(len(train["Question-Paraphrased"]), 2)
        self.assertIsInstance(train["Question-Disambiguated"], list)

        dev = split.dev.iloc[0]
        self.assertEqual(dev["Answer-Entity"], [3])
        self.assertEqual(dev["Path-Key"], [1, 2])

        test = split.test.iloc[0]
        self.assertEqual(test["Answer-Entity"], [4])
        self.assertEqual(test["Path-Key"], [2, 0])

    def test_supervised_wrapper_uses_corrected_function_name(self):
        frame = pd.DataFrame(
            {
                "Question": ["q0", "q1", "q2"],
                "Source-Entity": ["e0", "e1", "e2"],
                "Answer-Entity": ["e1", "e2", "e3"],
                "SplitLabel": ["train", "dev", "test"],
            }
        )
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        csv_path = os.path.join(tempdir.name, "qa.csv")
        metadata_path = os.path.join(tempdir.name, "qa_cache.json")
        frame.to_csv(csv_path, index=False)

        with mock.patch.object(pd.DataFrame, "to_parquet", autospec=True):
            split, metadata = process_and_cache_supervised_triviaqa_data(
                raw_QAData_path=csv_path,
                cached_toked_qatriples_metadata_path=metadata_path,
                question_tokenizer=self.tokenizer,
                answer_tokenizer=None,
                entity2id=self.entity2id,
                relation2id=self.relation2id,
                seed=42,
            )

        self.assertEqual(metadata["schema"], "MultiHopKGQA-v2")
        self.assertEqual(len(split.train), 1)


if __name__ == "__main__":
    unittest.main()
