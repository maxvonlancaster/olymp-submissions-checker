import json
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).with_name("submissions-checker.ipynb")


def load_feature_namespace():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    wanted_functions = {
        "extract_style_features",
        "indentation_consistency",
        "repetition_score",
        "overexplanation_in_comments",
        "analyze_code_style",
    }
    namespace = {}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell["source"])
        if any(f"def {name}" in source for name in wanted_functions):
            exec(source, namespace)
    return namespace


FEATURES = load_feature_namespace()
analyze_code_style = FEATURES["analyze_code_style"]
extract_style_features = FEATURES["extract_style_features"]


class StyleFeatureTests(unittest.TestCase):
    def test_python_bindings_are_distinct_and_references_are_ignored(self):
        result = extract_style_features(
            "x = 1\nx = x + 1\ndef f(value):\n    y = value\n", ".py"
        )
        self.assertEqual(result["num_variables"], 3)
        self.assertAlmostEqual(result["avg_var_length"], 7 / 3)
        self.assertAlmostEqual(result["comment_density"], 0.0)

    def test_python_comments_and_strings(self):
        code = 'text = "# not a comment"  # actual\nvalue = 2\n'
        result = extract_style_features(code, ".py")
        self.assertAlmostEqual(result["comment_density"], 0.5)
        result = analyze_code_style(code, ".py")
        self.assertEqual(result["overexplanation_score"], 1.0)

    def test_c_cpp_csharp_and_pascal_declarations(self):
        samples = {
            ".c": "int count = 0; // count\ncount++;\n",
            ".cpp": "double total = 0; /* total */\nint count = 1;\n",
            ".cs": 'int count = 0; string text = "// no comment";\n',
            ".pas": "var\n  count, total: Integer;\nbegin\n  count := count + 1; // increment\nend.\n",
        }
        for extension, code in samples.items():
            with self.subTest(extension=extension):
                result = extract_style_features(code, extension)
                self.assertGreaterEqual(result["num_variables"], 1)
                self.assertTrue(all(value >= 0 for value in result.values()))
        self.assertEqual(extract_style_features(samples[".pas"], ".pas")["num_variables"], 2)
        self.assertAlmostEqual(extract_style_features(samples[".c"], ".c")["comment_density"], 0.5)

    def test_empty_and_comment_free_inputs(self):
        result = analyze_code_style("", ".py")
        self.assertEqual(
            list(result),
            [
                "avg_var_length",
                "comment_density",
                "num_variables",
                "indentation_consistency",
                "repetition_score",
                "overexplanation_score",
            ],
        )
        self.assertEqual(
            result,
            {
                "avg_var_length": 0.0,
                "comment_density": 0.0,
                "num_variables": 0,
                "indentation_consistency": 1.0,
                "repetition_score": 0.0,
                "overexplanation_score": 0.0,
            },
        )
        self.assertEqual(extract_style_features('text = "//"\n', ".cs")["comment_density"], 0.0)

    def test_repetition_and_indentation_values(self):
        result = analyze_code_style("a\nb\nc\n", ".py")
        self.assertEqual(result["repetition_score"], 0.0)
        result = analyze_code_style("a\na\n b\n b\n", ".py")
        self.assertAlmostEqual(result["repetition_score"], 0.5)
        self.assertAlmostEqual(result["indentation_consistency"], 1.0)
        result = analyze_code_style("a\n b\n   c\n", ".py")
        self.assertAlmostEqual(result["indentation_consistency"], 0.5)

    def test_malformed_python_is_conservative(self):
        result = extract_style_features("broken = (1 +\n", ".py")
        self.assertEqual(result["num_variables"], 0)
        self.assertEqual(result["avg_var_length"], 0.0)
        self.assertTrue(all(value == value for value in result.values()))


if __name__ == "__main__":
    unittest.main()