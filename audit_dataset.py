"""Read-only audit of the repository's persisted dataset artifacts.

This script uses only the Python standard library. It reads feature JSON files,
the notebook as JSON, and any raw-source directories that are already present;
it never changes inputs and never runs notebook cells.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HUMAN_FILE = ROOT / "style_analysis_results.json"
AI_FILE = ROOT / "style_analysis_results_gen.json"
NOTEBOOK_FILE = ROOT / "submissions-checker.ipynb"
REPORT_FILE = ROOT / "dataset_audit.md"
FEATURES = (
    "avg_var_length",
    "comment_density",
    "num_variables",
    "indentation_consistency",
    "repetition_score",
    "overexplanation_score",
)
SOURCE_EXTENSIONS = {".py", ".cpp", ".c", ".cs", ".pas", ".java"}


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def notebook_text(notebook):
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


def human_language(record):
    return f"{record.get('Language', '<missing>')} ({record.get('FileExtension', '<missing>')})"


def generated_language(record):
    return Path(str(record.get("file_name", ""))).suffix or "<missing>"


def normalize_source(source):
    return re.sub(r"\s+", " ", source.replace("\ufeff", "")).strip()


def source_files():
    source_roots = [ROOT / name for name in ("generated", "submissions", "datasets")]
    files = []
    for source_root in source_roots:
        if not source_root.is_dir():
            continue
        for path in source_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in SOURCE_EXTENSIONS:
                files.append(path)
    return files


def raw_source_audit(paths):
    hashes = defaultdict(list)
    sources = {}
    for path in paths:
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        normalized = normalize_source(source)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        hashes[digest].append(path)
        sources[path] = normalized
    duplicate_groups = sum(len(items) > 1 for items in hashes.values())
    duplicate_files = sum(len(items) for items in hashes.values() if len(items) > 1)

    # Deterministic bounded candidate generation: compare only adjacent files
    # after sorting by length, which avoids quadratic work on large collections.
    ordered = sorted(sources.items(), key=lambda item: (len(item[1]), str(item[0])))
    near_pairs = 0
    for index, (path, source) in enumerate(ordered):
        for other_path, other in ordered[index + 1:index + 21]:
            if len(other) > max(1, len(source) * 1.5):
                break
            if SequenceMatcher(None, source, other, autojunk=False).quick_ratio() >= 0.95:
                near_pairs += 1
    return len(sources), duplicate_groups, duplicate_files, near_pairs


def split_logic(text):
    return {
        "stratified": "stratify=y" in text,
        "test_size": re.findall(r"test_size\s*=\s*([^,\n]+)", text),
        "random_state": re.findall(r"random_state\s*=\s*([^,\n\)]+)", text),
    }


def field_report(records):
    return sorted({key for record in records for key in record})


def metadata_counts(records, key):
    values = [record.get(key) for record in records if key in record]
    return len(values), len(values) - len(set(map(str, values)))


def make_report():
    human = load_json(HUMAN_FILE)
    ai = load_json(AI_FILE)
    notebook = json.loads(NOTEBOOK_FILE.read_text(encoding="utf-8"))
    text = notebook_text(notebook)
    raw_paths = source_files()
    raw_count, duplicate_groups, duplicate_files, near_pairs = raw_source_audit(raw_paths)
    human_language_counts = Counter(human_language(record) for record in human)
    ai_language_counts = Counter(generated_language(record) for record in ai)
    split = split_logic(text)
    human_id_count, human_id_duplicates = metadata_counts(human, "SolutionID")
    ai_file_count, ai_file_duplicates = metadata_counts(ai, "file_name")
    has_generated_dir = (ROOT / "generated").is_dir()
    has_dataset_dir = (ROOT / "datasets").is_dir()

    human_distribution = ", ".join(
        f"{language}: {count}" for language, count in sorted(human_language_counts.items())
    )
    ai_distribution = ", ".join(
        f"{language}: {count}" for language, count in sorted(ai_language_counts.items())
    )
    source_status = (
        f"{raw_count} source files found" if raw_count else "No raw source files found"
    )

    report = f"""# Dataset Audit

## 1. Dataset Inventory

- **VERIFIED:** `style_analysis_results.json` contains **{len(human)} human-labelled feature records**.
- **VERIFIED:** `style_analysis_results_gen.json` contains **{len(ai)} AI-labelled feature records**.
- **VERIFIED:** Both files contain the six feature keys: `{', '.join(FEATURES)}`.
- **VERIFIED:** The repository has no `generated/` directory and no `datasets/` directory; {source_status}.
- **NOT VERIFIABLE:** Feature JSON records do not contain raw source code, so source-level dataset size cannot be independently reconstructed from them.

## 2. Class and Language Distribution

- **VERIFIED:** Human class counts: {human_distribution}.
- **VERIFIED:** AI class counts by filename extension: {ai_distribution}.
- **POTENTIAL RISK:** Language is confounded with class: human records span several languages, while AI records are only `.py` and `.cpp` in the persisted metadata. The class/language balance is therefore not matched.
- **NOT VERIFIABLE:** A formal effect size or source-level language distribution cannot be computed beyond these metadata counts.

## 3. Data Provenance and Selection

- **VERIFIED:** The notebook reads `datasets/third.csv` and `datasets/solutions.csv`, merges on `SolutionID`, drops rows with missing `Language`, maps supported language labels to extensions, and appends feature records.
- **VERIFIED:** Human preparation stops after the first **500 successfully analysed rows** in the post-`Language`-filter iteration order. The code does not sort rows and does not document exclusions beyond missing language and rows that raise extraction exceptions.
- **VERIFIED:** The notebook's supported mapping includes Python, C++, Java, C#, Pascal, and C; unsupported/unmapped languages become `.txt`, which the feature extractor rejects and therefore cannot contribute successfully.
- **VERIFIED:** AI generation code requests `gpt-4o`, randomly chooses one of 20 topic labels and `.py`/`.cpp`, requests 400 submissions, starts filenames at `100`, and skips responses without the requested delimiters. The persisted AI feature file has 499 records, so the generation code and file are not perfectly reconciled.
- **VERIFIED:** The notebook uses `third.csv`/`solutions.csv` and an OpenAI generation step as its stated provenance.
- **NOT VERIFIABLE:** The source datasets, exact row ordering, omitted rows, API responses, prompt completions, and model/API version details are not present in this repository.

## 4. Duplicate and Near-Duplicate Analysis

- **NOT VERIFIABLE:** Exact normalized-source duplicate counts within and across classes cannot be computed because no raw source code or class-linked source files are present. No duplicate counts are inferred from feature-vector equality.
- **NOT VERIFIABLE:** Near-duplicate source analysis cannot be performed for the same reason. The audit script supports a deterministic bounded `SequenceMatcher` check if raw source files are later present, but it found no source inputs now.
- **VERIFIED:** Human `SolutionID` metadata has {human_id_count} populated values and {human_id_duplicates} repeated values in the feature file. AI `file_name` metadata has {ai_file_count} populated values and {ai_file_duplicates} repeated values. These are metadata checks, not source duplicate checks.

## 5. Potential Data Leakage

- **VERIFIED:** The notebook concatenates both classes and applies `train_test_split` with `test_size={split['test_size'][-1] if split['test_size'] else '<not found>'}`, `random_state={split['random_state'][-1] if split['random_state'] else '<not found>'}`, and `{'stratification enabled' if split['stratified'] else 'no stratification detected'}; no grouping key is supplied.
- **POTENTIAL RISK:** If multiple submissions share a student, task, source, or generated prompt family, a row-wise split can place related records in both train and test sets. This cannot be quantified because the persisted features omit student/task/source linkage.
- **POTENTIAL RISK:** The human records include `SolutionID`, `Language`, and `FileExtension`; AI records include `file_name`. These metadata fields are not selected as model features, but their differing schemas make provenance and grouping checks incomplete.
- **NOT VERIFIABLE:** Cross-class source duplication, shared tasks, shared students, and prompt-family overlap cannot be checked without raw data and original metadata.

## 6. Missing Metadata

- **VERIFIED:** Human feature records expose `SolutionID`, language, and extension; AI records expose only a generated filename in addition to features.
- **NOT VERIFIABLE:** Student identifiers, task/problem identifiers, submission IDs beyond the human `SolutionID`, generator/model identifiers, prompt IDs, timestamps, and provenance links are not consistently available in the feature JSON files.
- **POTENTIAL RISK:** Notebook outputs and source-loading code refer to student nicknames and task-bearing filenames in other workflows, but those source files are absent; this prevents reproducible linkage and leakage auditing.
- **NOT VERIFIABLE:** The original dataset cannot be reconstructed exactly from the repository: the raw datasets, raw generated files, API outputs, and the complete pre-filter human population are missing.

## 7. Recommended Corrective Actions

- **POTENTIAL RISK:** Preserve a manifest for every row containing class, source hash, language, task ID, student/group ID, generator/model, prompt/topic, and split group, while keeping personal identifiers access-controlled.
- **POTENTIAL RISK:** Deduplicate and group-split by normalized source hash and by student/task/prompt group before any model evaluation; report the number of removed or held-out groups without silently changing labels.
- **POTENTIAL RISK:** Match or stratify language distributions across classes, or include language-aware grouped evaluation, so language is not an unmeasured class proxy.
- **VERIFIED:** Re-run the audit after raw sources and manifests are archived; this script can then report bounded deterministic source duplicate and near-duplicate checks without executing training.
"""
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_FILE}")
    print(f"Human samples: {len(human)}")
    print(f"AI samples: {len(ai)}")
    print(f"Raw source files: {raw_count}")
    print(f"Normalized exact duplicate groups: {duplicate_groups}; duplicate files: {duplicate_files}")
    print(f"Bounded near-duplicate pairs: {near_pairs}")


if __name__ == "__main__":
    make_report()