# Dataset Audit

## 1. Dataset Inventory

- **VERIFIED:** `style_analysis_results.json` contains **500 human-labelled feature records**.
- **VERIFIED:** `style_analysis_results_gen.json` contains **499 AI-labelled feature records**.
- **VERIFIED:** Both files contain the six feature keys: `avg_var_length, comment_density, num_variables, indentation_consistency, repetition_score, overexplanation_score`.
- **VERIFIED:** The repository has no `generated/` directory and no `datasets/` directory; No raw source files found.
- **NOT VERIFIABLE:** Feature JSON records do not contain raw source code, so source-level dataset size cannot be independently reconstructed from them.

## 2. Class and Language Distribution

- **VERIFIED:** Human class counts: C (.c): 161, C# (.cs): 9, C++ 4.3.2 (.cpp): 51, C++ 4.9.2 (.cpp): 137, C++14 (.cpp): 87, PAS fpc (.pas): 1, PYTH (.py): 25, PYTH 3.4 (.py): 29.
- **VERIFIED:** AI class counts by filename extension: .cpp: 259, .py: 240.
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
- **VERIFIED:** Human `SolutionID` metadata has 500 populated values and 0 repeated values in the feature file. AI `file_name` metadata has 499 populated values and 0 repeated values. These are metadata checks, not source duplicate checks.

## 5. Potential Data Leakage

- **VERIFIED:** The notebook concatenates both classes and applies `train_test_split` with `test_size=0.2`, `random_state=42`, and `stratification enabled; no grouping key is supplied.
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
