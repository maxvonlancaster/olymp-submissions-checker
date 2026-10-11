# Pre-training Quality Report

This read-only check joins `datasets/ai_code_detection_manifest.csv` to
`datasets/ai_code_detection.parquet` using `original_row_index`, and joins the
same key to `datasets/ai_code_detection_features.csv`. All 11,729 rows are
labelled `Python`. Parsing uses `ast.parse` on the Parquet `code` value. Failure
samples are reproducible: within each class, rows are sorted by original row
index and the first five failures are shown.

## AST Parsing Failures

| class | split | rows | AST failures | rate |
|---|---:|---:|---:|---:|
| Human_written | train | 4,535 | 1,082 | 23.859% |
| Human_written | test | 1,136 | 269 | 23.680% |
| Human_written | total | 5,671 | 1,351 | 23.823% |
| Ai_generated | train | 4,846 | 590 | 12.175% |
| Ai_generated | test | 1,212 | 153 | 12.624% |
| Ai_generated | total | 6,058 | 743 | 12.265% |
| **Total** | **all** | **11,729** | **2,094** | **17.854%** |

The train/test rates are close within each class, so the issue is not primarily
a split artifact. Invalidity is nevertheless strongly associated with class:
the human rate is about twice the AI rate.

## Failure Categories

The categories below group related `SyntaxError.msg` values; counts are within
class. `invalid syntax` includes the parser's suggestion variants, and Unicode
whitespace/character errors are grouped together.

| class | category | count | likely interpretation |
|---|---|---:|---|
| Human_written | non-printable or Unicode characters | 936 | Mostly U+00A0 non-breaking spaces; also full-width spaces and other Unicode symbols. |
| Human_written | invalid syntax family | 179 | Old or malformed syntax, including parser suggestion variants. |
| Human_written | Python 2 `print` syntax | 75 | Python 2 programs parsed with Python 3. |
| Human_written | other indentation/delimiter/expected-token errors | 161 | Inconsistent indentation, missing delimiters, and incomplete statements. |
| Ai_generated | invalid syntax family | 686 | Generic malformed or concatenated generated text. |
| Ai_generated | unterminated string literal | 31 | Truncated or malformed string content. |
| Ai_generated | expected indented block | 5 | Incomplete control-flow blocks. |
| Ai_generated | invalid decimal literal | 4 | Numeric/token corruption. |
| Ai_generated | invalid Unicode character | 4 | Curly punctuation or other non-ASCII source text. |
| Ai_generated | other parser errors | 13 | Delimiter, f-string, assignment, and indentation errors. |

Representative exact parser messages include human `invalid non-printable
character U+00A0`, `Missing parentheses in call to 'print'`, `expected ':'`,
`unexpected indent`, and `inconsistent use of tabs and spaces in indentation`;
AI `invalid syntax`, `unterminated string literal`, `invalid decimal literal`,
and `expected 'except' or 'finally' block`.

### Deterministic Failure Sample

| class | original row | split | parser message | normalized source prefix |
|---|---:|---|---|---|
| Ai_generated | 1223 | train | invalid character U+2019 | Prompt asking for the specific task to implement. |
| Ai_generated | 1307 | train | invalid syntax | Multiple statements concatenated on one line, including a malformed conditional expression. |
| Ai_generated | 1396 | train | invalid character U+2019 | Prompt asking for implementation details rather than source code. |
| Ai_generated | 1413 | train | invalid syntax | Shell/Python command examples concatenated as one source file. |
| Ai_generated | 1551 | train | invalid decimal literal | Pygame example with a malformed token near the start of the snippet. |
| Human_written | 0 | train | invalid non-printable character U+00A0 | Otherwise complete Rosetta Code solution containing non-breaking spaces. |
| Human_written | 2 | train | invalid non-printable character U+00A0 | Otherwise complete documented solution containing non-breaking spaces. |
| Human_written | 3 | train | invalid syntax | Interactive `>>>` transcript rather than a standalone program. |
| Human_written | 4 | train | invalid non-printable character U+00A0 | Python source containing non-breaking spaces. |
| Human_written | 5 | train | invalid non-printable character U+00A0 | Python 2 `raw_input` and `print` syntax plus non-breaking spaces. |

These examples indicate a mixture of encoding/whitespace contamination,
Python 2 or interactive transcripts, prompt leakage, and genuinely incomplete
or malformed snippets. They are not all equivalent quality failures.

## Zero Declaration Features

The extracted features intentionally return conservative zeros when Python AST
parsing fails. The observed counts are:

| class | validity | rows | `avg_var_length = 0` | `num_variables = 0` |
|---|---|---:|---:|---:|
| Human_written | valid | 4,320 | 66 (1.528%) | 66 (1.528%) |
| Human_written | invalid | 1,351 | 1,351 (100.000%) | 1,351 (100.000%) |
| Ai_generated | valid | 5,315 | 75 (1.411%) | 75 (1.411%) |
| Ai_generated | invalid | 743 | 743 (100.000%) | 743 (100.000%) |

Across all rows, zero values occur in 1,417/5,671 human rows (24.987%) and
818/6,058 AI rows (13.503%). Thus zero declaration features are a direct
invalidity marker for every failed parse, in addition to representing a small
number of valid programs with no detected bindings.

## Recommendation

Exclude invalid Python programs from the primary experiment and report the
resulting sample counts and class balance. Their rates differ substantially by
class, and the conservative zero fallback makes invalidity directly visible in
two model features. Retain all invalid programs only in a clearly labelled
sensitivity analysis, with validity as an explicitly reported data-quality
stratum. This avoids silently treating encoding errors, Python 2 code, prompts,
and incomplete snippets as ordinary valid Python while preserving transparency
about the original corpus.

No data, manifest, feature CSV, notebook, or model was changed.

## Tests

Executed with `.\\.venv\\Scripts\\python.exe`:

- `.\\.venv\\Scripts\\python.exe -m unittest -v test_style_features.py`: **9 passed**
- `.\\.venv\\Scripts\\python.exe -m unittest -v test_standalone_features.py`: **3 passed**
- `.\\.venv\\Scripts\\python.exe -m unittest -v test_dataset_manifest.py`: **3 passed**
- `.\\.venv\\Scripts\\python.exe -m unittest discover -v`: **15 passed**
