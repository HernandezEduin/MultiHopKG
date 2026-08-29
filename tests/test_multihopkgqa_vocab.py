import os
import tempfile
import unittest

from multihopkg.multihopkgqa_vocab import load_or_create_dictionaries


class TestMultiHopKGQAVocab(unittest.TestCase):
    def test_prefers_triplets_and_assigns_first_seen_zero_based_ids(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, "triplets.txt"), "w", encoding="utf-8") as file:
                file.write("e2\tr1\te0\n")
                file.write("e0\tr2\te3\n")
                file.write("e2\tr2\te4\n")

            # These must be ignored when triplets.txt exists.
            with open(os.path.join(tempdir, "train.txt"), "w", encoding="utf-8") as file:
                file.write("ignored\tr9\tignored2\n")
            with open(os.path.join(tempdir, "dev.txt"), "w", encoding="utf-8") as file:
                file.write("ignored2\tr9\tignored3\n")
            with open(os.path.join(tempdir, "test.txt"), "w", encoding="utf-8") as file:
                file.write("ignored3\tr9\tignored4\n")

            id2entity, entity2id, id2relation, relation2id = load_or_create_dictionaries(tempdir)

            self.assertEqual(entity2id, {"e2": 0, "e0": 1, "e3": 2, "e4": 3})
            self.assertEqual(relation2id, {"r1": 0, "r2": 1})
            self.assertEqual(id2entity[0], "e2")
            self.assertEqual(id2relation[1], "r2")
            self.assertNotIn("ignored", entity2id)

            with open(os.path.join(tempdir, "entity2id.txt"), encoding="utf-8") as file:
                self.assertEqual(
                    file.read(),
                    "e2\t0\ne0\t1\ne3\t2\ne4\t3\n",
                )
            with open(os.path.join(tempdir, "relation2id.txt"), encoding="utf-8") as file:
                self.assertEqual(file.read(), "r1\t0\nr2\t1\n")

    def test_falls_back_to_train_dev_test_in_order(self):
        with tempfile.TemporaryDirectory() as tempdir:
            contents = {
                "train.txt": "alice\tparent\tbob\n",
                "dev.txt": "bob\tsibling\tcarol\n",
                "test.txt": "dave\tparent\talice\n",
            }
            for filename, content in contents.items():
                with open(os.path.join(tempdir, filename), "w", encoding="utf-8") as file:
                    file.write(content)

            _, entity2id, _, relation2id = load_or_create_dictionaries(tempdir)

            self.assertEqual(
                entity2id,
                {"alice": 0, "bob": 1, "carol": 2, "dave": 3},
            )
            self.assertEqual(relation2id, {"parent": 0, "sibling": 1})

    def test_preserves_existing_entity_mapping_when_relation_mapping_missing(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, "triplets.txt"), "w", encoding="utf-8") as file:
                file.write("a\tr0\tb\n")
                file.write("b\tr1\tc\n")

            existing_entity = "a\t7\nb\t3\nc\t11\n"
            entity_path = os.path.join(tempdir, "entity2id.txt")
            with open(entity_path, "w", encoding="utf-8") as file:
                file.write(existing_entity)

            _, entity2id, _, relation2id = load_or_create_dictionaries(tempdir)

            self.assertEqual(entity2id, {"a": 7, "b": 3, "c": 11})
            self.assertEqual(relation2id, {"r0": 0, "r1": 1})
            with open(entity_path, encoding="utf-8") as file:
                self.assertEqual(file.read(), existing_entity)

    def test_requires_complete_split_fallback(self):
        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, "train.txt"), "w", encoding="utf-8") as file:
                file.write("a\tr\tb\n")

            with self.assertRaises(FileNotFoundError):
                load_or_create_dictionaries(tempdir)


if __name__ == "__main__":
    unittest.main()
