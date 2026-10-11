import json
import unittest
from pathlib import Path

from extract_dataset_features import extract_rows
from style_features import analyze_code_style


def notebook_namespace():
    notebook = json.loads(Path("submissions-checker.ipynb").read_text(encoding="utf-8"))
    namespace = {}
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and index in {3, 5, 7, 9, 11}:
            exec(source, namespace)
    return namespace


class StandaloneFeatureTests(unittest.TestCase):
    def test_corrected_features_match_notebook_for_representative_python(self):
        samples = ["'''module docs\nexplain the module\n'''\n# explain binding\nvalue = 1\nvalue = value + 1\ndef work(argument):\n    result = argument\n    return result\n", "a\na\n b\n b\n", "value = '# text'  # actual comment\n"]
        notebook = notebook_namespace()
        for code in samples:
            with self.subTest(code=code):
                self.assertEqual(analyze_code_style(code, ".py"), notebook["analyze_code_style"](code, ".py"))

    def test_corrected_features_match_notebook_for_declarations_and_comments(self):
        samples = {".cpp": "std::vector<int> values = make_values(3), backup;\nint first = 1, second = 2; // explain both values\n", ".cs": 'System.String name = GetName(), alias;\nstring text = "// not a comment";\n', ".pas": "var\n  count, total: Integer;\nbegin\n  count := count + 1; // increment\nend.\n"}
        notebook = notebook_namespace()
        for extension, code in samples.items():
            with self.subTest(extension=extension):
                self.assertEqual(analyze_code_style(code, extension), notebook["analyze_code_style"](code, extension))

    def test_manifest_extraction_rejects_hash_mismatch(self):
        manifest = [{"original_row_index": "0", "task_name": "task", "label": "Human_written", "source": "test", "generator": "Human", "code_sha256": "wrong", "split": "train"}]
        rows, _, failures = extract_rows(manifest, [{"code": "x = 1\n", "language_name": "Python"}])
        self.assertEqual(rows, [])
        self.assertIn("SHA-256 mismatch", failures[0])


if __name__ == "__main__":
    unittest.main()