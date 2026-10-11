"""Audit the committed candidate Parquet dataset without modifying it."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as parquet


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "datasets" / "ai_code_detection.parquet"
REPORT = ROOT / "candidate_dataset_report.md"
SEED = 42
TEST_FRACTION = 0.20
REQUIRED_COLUMNS = {
    "task_name",
    "task_description",
    "language_name",
    "code",
    "label",
    "generator",
    "source",
}


def is_missing(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def counter_lines(counter: Counter) -> str:
    return "\n".join(f"| {key} | {value:,} |" for key, value in sorted(counter.items()))


def markdown_table(headers: tuple[str, ...], rows: list[tuple]) -> str:
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    output.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(output)


def cross_table(rows: list[dict], row_key: str, column_key: str, row_label: str, column_label: str) -> str:
    row_values = sorted({row[row_key] for row in rows})
    column_values = sorted({row[column_key] for row in rows})
    matrix = []
    for row_value in row_values:
        matrix.append((row_value, *(sum(row[row_key] == row_value and row[column_key] == column_value for row in rows) for column_value in column_values)))
    return markdown_table((row_label, *column_values), matrix)


def distribution(values: list[int]) -> str:
    ordered = sorted(values)
    return (
        f"min={ordered[0]}, p25={statistics.quantiles(ordered, n=4, method='inclusive')[0]:.1f}, "
        f"median={statistics.median(ordered):.1f}, mean={statistics.mean(ordered):.2f}, "
        f"p75={statistics.quantiles(ordered, n=4, method='inclusive')[2]:.1f}, max={ordered[-1]}"
    )


def normalize_code(code: str) -> str:
    normalized = code.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n"))


def duplicate_groups(rows: list[dict]) -> dict[str, list[int]]:
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        digest = hashlib.sha256(normalize_code(row["code"]).encode("utf-8")).hexdigest()
        groups[digest].append(index)
    return {digest: indexes for digest, indexes in groups.items() if len(indexes) > 1}


def choose_group_split(rows: list[dict], tasks: dict[str, list[int]]) -> tuple[set[str], set[str]]:
    totals = Counter(row["label"] for row in rows)
    target_total = round(len(rows) * TEST_FRACTION)
    target_labels = {label: round(count * TEST_FRACTION) for label, count in totals.items()}
    task_names = list(tasks)
    random.Random(SEED).shuffle(task_names)
    task_counts = {
        task: Counter(rows[index]["label"] for index in indexes)
        for task, indexes in tasks.items()
    }
    target_task_count = round(target_total / statistics.mean(len(indexes) for indexes in tasks.values()))
    test_tasks: set[str] = set(task_names[:target_task_count])
    test_counts: Counter = Counter(
        label
        for task in test_tasks
        for label, count in task_counts[task].items()
        for _ in range(count)
    )
    test_total = sum(len(tasks[task]) for task in test_tasks)

    def objective(total: int, counts: Counter) -> float:
        total_error = (total - target_total) / max(1, target_total)
        label_error = sum(
            ((counts[label] - target_labels[label]) / max(1, target_labels[label])) ** 2
            for label in totals
        )
        return total_error**2 + label_error

    # Improve class balance while preserving whole-task assignment.
    rng = random.Random(SEED)
    current_score = objective(test_total, test_counts)
    best_score = current_score
    best_tasks = set(test_tasks)
    for iteration in range(50_000):
        outgoing = rng.choice(sorted(test_tasks))
        incoming = rng.choice(sorted(set(tasks) - test_tasks))
        candidate_total = test_total - len(tasks[outgoing]) + len(tasks[incoming])
        candidate_counts = Counter(test_counts)
        for label, count in task_counts[outgoing].items():
            candidate_counts[label] -= count
        candidate_counts.update(task_counts[incoming])
        candidate_score = objective(candidate_total, candidate_counts)
        temperature = 0.02 * (1 - iteration / 50_000) + 0.0001
        accept = candidate_score < current_score or rng.random() < math.exp((current_score - candidate_score) / temperature)
        if accept:
            test_tasks.remove(outgoing)
            test_tasks.add(incoming)
            test_counts = candidate_counts
            test_total = candidate_total
            current_score = candidate_score
            if current_score < best_score:
                best_score = current_score
                best_tasks = set(test_tasks)
    return best_tasks, set(tasks) - best_tasks


def main() -> None:
    table = parquet.read_table(DATASET)
    columns = set(table.column_names)
    missing_columns = REQUIRED_COLUMNS - columns
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    rows = table.to_pylist()
    labels = Counter(row["label"] for row in rows)
    languages = Counter(row["language_name"] for row in rows)
    tasks = defaultdict(list)
    for index, row in enumerate(rows):
        tasks[row["task_name"]].append(index)

    missing = {
        column: sum(is_missing(row.get(column)) for row in rows)
        for column in table.column_names
    }
    label_language = cross_table(rows, "label", "language_name", "label", "language")
    label_generator = cross_table(rows, "label", "generator", "label", "generator")
    label_source = cross_table(rows, "label", "source", "label", "source")
    task_label_counts = {
        task: Counter(rows[index]["label"] for index in indexes)
        for task, indexes in tasks.items()
    }
    mixed_tasks = {
        task for task, counts in task_label_counts.items() if len(counts) > 1
    }
    per_task = [len(indexes) for indexes in tasks.values()]
    per_class_task = {
        label: [counts[label] for counts in task_label_counts.values() if counts[label]]
        for label in labels
    }

    duplicates = duplicate_groups(rows)
    cross_class_duplicates = {
        digest: indexes
        for digest, indexes in duplicates.items()
        if len({rows[index]["label"] for index in indexes}) > 1
    }
    duplicate_rows = sum(len(indexes) for indexes in duplicates.values())
    cross_class_rows = sum(len(indexes) for indexes in cross_class_duplicates.values())

    test_tasks, train_tasks = choose_group_split(rows, tasks)
    test_indexes = [index for task in test_tasks for index in tasks[task]]
    train_indexes = [index for task in train_tasks for index in tasks[task]]
    test_labels = Counter(rows[index]["label"] for index in test_indexes)
    train_labels = Counter(rows[index]["label"] for index in train_indexes)
    task_overlap = train_tasks & test_tasks

    schema_rows = [(field.name, str(field.type), missing[field.name]) for field in table.schema]
    label_rows = [(label, count) for label, count in sorted(labels.items())]
    language_rows = [(language, languages[language]) for language in sorted(languages)]
    missing_rows = [(column, count) for column, count in missing.items()]
    split_rows = [
        ("train", len(train_indexes), len(train_tasks), *(train_labels[label] for label in sorted(labels))),
        ("test", len(test_indexes), len(test_tasks), *(test_labels[label] for label in sorted(labels))),
    ]
    split_headers = ("split", "rows", "tasks", *(f"{label} rows" for label in sorted(labels)))

    duplicate_detail = []
    for digest, indexes in sorted(duplicates.items(), key=lambda item: (-len(item[1]), item[0])):
        duplicate_detail.append((digest[:12], len(indexes), ", ".join(sorted({rows[index]["label"] for index in indexes})), ", ".join(sorted({rows[index]["task_name"] for index in indexes})[:4])))
    duplicate_table = markdown_table(("normalized SHA-256 (prefix)", "rows", "labels", "tasks (up to 4)"), duplicate_detail) if duplicate_detail else "No exact duplicate groups found."

    report = f"""# Candidate Dataset Audit

Generated by `inspect_candidate_dataset.py` from `datasets/ai_code_detection.parquet`. The Parquet file was read-only input; no classifier was trained and no code solutions were generated.

## 1. Inventory and Missing Values

- **Rows:** {len(rows):,}
- **Columns:** {', '.join(table.column_names)}

{markdown_table(("column", "Arrow type", "missing (null or blank)"), schema_rows)}

## 2. Labels

{markdown_table(("label value", "rows"), label_rows)}

The exact label values are `Human_written` and `Ai_generated`. The `generator` and `source` fields corroborate this semantic distinction below: human rows are attributed to the human generator/source, while generated rows are attributed to the AI generator/source values present in the file. Labels are treated as categorical strings, not inferred from row order.

## 3. Programming Languages

{markdown_table(("language", "rows"), language_rows)}

Language by class:

{label_language}

## 4. Tasks and Submissions

- **Distinct tasks:** {len(tasks):,}
- **Tasks containing both classes:** {len(mixed_tasks):,}
- **Tasks containing only one class:** {len(tasks) - len(mixed_tasks):,}
- **Submissions per task:** {distribution(per_task)}

Submissions per task, by class:

{markdown_table(("class", "tasks with class", "per-task distribution"), [(label, len(per_class_task[label]), distribution(per_class_task[label])) for label in sorted(per_class_task)])}

## 5. Exact Duplicate Code Samples

Code was normalized in memory by converting CRLF/CR to LF and removing trailing whitespace from every line. The normalized code was hashed with SHA-256. Blank code is included as a value and is separately visible in the missingness table.

- **Duplicate hash groups:** {len(duplicates):,}
- **Rows in duplicate groups:** {duplicate_rows:,}
- **Cross-class duplicate groups:** {len(cross_class_duplicates):,}
- **Rows in cross-class duplicate groups:** {cross_class_rows:,}

{duplicate_table}

## 6. Potential Confounding

Generator by class:

{label_generator}

Source by class:

{label_source}

Task composition is also a concern: {len(mixed_tasks):,} of {len(tasks):,} tasks ({100 * len(mixed_tasks) / len(tasks):.1f}%) contain both labels, so task identity can be informative while still allowing within-task cross-class comparisons. Language is a direct potential proxy where its class counts above are uneven. Generator and source are potential provenance proxies whenever their values are concentrated in one class; they must not be used as model features.

## 7. Task-Disjoint Evaluation Feasibility

A deterministic seeded task-level search (`seed={SEED}`) selected whole tasks for an approximately {TEST_FRACTION:.0%} test set while minimizing deviation from both the overall test-row target and per-class test-row targets. It uses all tasks across train and test and never splits a task.

{markdown_table(split_headers, split_rows)}

- **Task overlap between train and test:** {len(task_overlap)}
- **Test fraction by rows:** {len(test_indexes) / len(rows):.3%}
- **Class proportions in train:** {', '.join(f'{label}={train_labels[label] / len(train_indexes):.3%}' for label in sorted(labels))}
- **Class proportions in test:** {', '.join(f'{label}={test_labels[label] / len(test_indexes):.3%}' for label in sorted(labels))}

This demonstrates feasibility for this artifact. The recommended evaluation is this task-grouped split with `seed={SEED}`, retaining all suitable tasks after exact normalized-code deduplication policy is decided. Report the held-out task list or a stable split manifest so the evaluation can be reproduced. Because tasks can contain both classes and task sizes are unequal, exact row-level 80/20 and exact class balance are not guaranteed by a group-disjoint split.

## Critical Limitations

- Task name is the available grouping identifier; it may not capture duplicate prompts, source provenance, student identity, or generator prompt families.
- Exact duplicate detection catches identical normalized code, not semantic or near duplicates.
- The report is descriptive only. It does not establish that language, source, generator, or task effects are causal.
- Install the only non-standard dependency with `.\\.venv\\Scripts\\python.exe -m pip install pyarrow` if `pyarrow` is unavailable.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT}")
    print(f"Rows: {len(rows)}; tasks: {len(tasks)}; mixed tasks: {len(mixed_tasks)}")
    print(f"Duplicate groups: {len(duplicates)}; cross-class groups: {len(cross_class_duplicates)}")
    print(f"Split: train={len(train_indexes)} rows/{len(train_tasks)} tasks, test={len(test_indexes)} rows/{len(test_tasks)} tasks, overlap={len(task_overlap)}")


if __name__ == "__main__":
    main()