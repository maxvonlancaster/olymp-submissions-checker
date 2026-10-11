import unittest

from prepare_dataset_manifest import assign_splits, clean_rows


def row(code, label="Human_written", task_name="task-1"):
    return {
        "task_name": task_name,
        "code": code,
        "label": label,
        "source": "test",
        "generator": "Human",
    }


class DatasetManifestTests(unittest.TestCase):
    def test_cross_class_duplicates_are_all_excluded_and_same_class_is_canonical(self):
        rows = [
            row("x = 1\r\n", task_name="a"),
            row("x = 1\n", label="Ai_generated", task_name="b"),
            row("y = 2  \n", task_name="c"),
            row("y = 2\n", task_name="d"),
        ]
        cleaned, excluded, _ = clean_rows(rows)
        self.assertEqual([item["original_row_index"] for item in cleaned], [2])
        self.assertEqual(excluded["cross_class_duplicate"], 2)
        self.assertEqual(excluded["same_class_duplicate"], 1)

    def test_split_is_deterministic_and_task_disjoint(self):
        rows = [row(f"code-{index}", label="Ai_generated" if index % 2 else "Human_written", task_name=f"task-{index // 2}") for index in range(12)]
        first, train_tasks, test_tasks = assign_splits(rows)
        second, second_train, second_test = assign_splits(rows)
        self.assertEqual(first, second)
        self.assertEqual(train_tasks, second_train)
        self.assertEqual(test_tasks, second_test)
        self.assertFalse(train_tasks & test_tasks)

    def test_cleaned_hashes_cannot_cross_splits(self):
        rows = [row(f"code-{index}", task_name=f"task-{index}") for index in range(10)]
        cleaned, _, _ = clean_rows(rows)
        assigned, _, _ = assign_splits(cleaned)
        train_hashes = {item["code_sha256"] for item in assigned if item["split"] == "train"}
        test_hashes = {item["code_sha256"] for item in assigned if item["split"] == "test"}
        self.assertEqual(train_hashes & test_hashes, set())


if __name__ == "__main__":
    unittest.main()