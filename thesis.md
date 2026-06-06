# Detecting AI-Generated Code in Competitive Programming Submissions

## Abstract
This thesis presents an empirical framework for detecting AI-generated code submissions in programming contests. The proposed method combines code style analysis with statistical outlier detection to identify submissions that deviate from the student population. The implementation is drawn from a prototype notebook that processes Python, C++, C, C#, and Pascal source files, computes structural and stylistic features, and flags anomalous solutions for further review.

## Introduction
Recent advances in large language models have made it trivial for students to obtain working code solutions. In competitive programming and academic evaluation, distinguishing machine-generated code from authentic student work is a key integrity challenge. This work explores style-based heuristics and anomaly detection as a lightweight, interpretable approach to complement existing plagiarism and semantic analysis techniques.

## Proposed Detection Framework
The notebook defines an overall scoring model in descriptive form:

Final Score = w1 * StyleScore + w2 * ComplexityScore + w3 * LLMScore + w4 * SimilarityScore + w5 * SemanticScore

Although only the stylistic components are implemented in the prototype, this formula situates style analysis within a broader multi-factor detection system.

### Style Feature Extraction
Style analysis is central to the implemented method. The prototype extracts the following features from source code:

- `avg_var_length`: average length of variable identifiers. AI-generated code often uses systematically named variables and can differ from typical student naming patterns.
- `comment_density`: number of comments per line of code. Generated submissions may exhibit abnormal comment frequency.
- `num_variables`: count of detected variables. Variation in this metric reflects code complexity and author habits.

For Python, the analysis uses AST parsing to identify variable names and counts comment occurrences. For C-family languages, a heuristic parser finds declarations and strips string literals before matching variable definitions.

### Indentation Consistency
The notebook includes an `indentation_consistency` feature that measures whitespace regularity:

- It computes indentation levels for non-empty lines and calculates their variance.
- A consistency score is derived as `1 / (1 + variance)`.

The hypothesis is that AI-generated submissions often display unusually perfect indentation and spacing, while human-authored code shows greater variability.

### Repetition Score
Repetition within code is measured by comparing the number of unique non-empty lines to the total number of lines:

- `repetition_score = 1 - len(unique_lines) / (len(lines) + 1)`

This captures pattern repetition and structural redundancy, which can be higher in machine-generated code due to template-like generation.

### Overexplanation in Comments
The prototype includes a comment verbosity metric:

- It calculates average comment word count across all comments.
- This `overexplanation_score` is intended to detect AI-generated comments that are overly descriptive or pedagogical compared to typical student commentary.

## Dataset and Processing Pipeline
The notebook processes a local submissions directory containing student upload files. The pipeline includes:

- extracting student nickname metadata from filenames,
- reading source content,
- determining file language by extension,
- computing style features for each submission,
- collecting results in a feature list,
- generating box plots for each feature distribution.

## Anomaly Detection and Outlier Reporting
The detection stage uses classical statistical outlier detection based on the interquartile range (IQR):

- For each feature, compute Q1 and Q3,
- derive `lower_bound = Q1 - 1.5 * IQR` and `upper_bound = Q3 + 1.5 * IQR`,
- flag submissions outside these bounds as outliers.

Outliers are reported for each feature, and the notebook writes a JSON file (`outliers.json`) summarizing flagged submissions.

## Discussion
The implemented approach emphasizes interpretability and ease of use:

- Style features are simple to compute and explain,
- the method supports multiple source languages,
- outlier detection highlights candidate submissions rather than issuing deterministic labels.

Limitations include the reliance on surface-level heuristics and the absence of the full composite score components (`ComplexityScore`, `LLMScore`, `SimilarityScore`, `SemanticScore`) in the prototype. Future work should integrate semantic similarity, model-based authorship scoring, and cross-submission comparisons.

## Conclusion
This thesis documents a practical prototype for detecting AI-generated student code using style-based heuristics and statistical anomaly detection. The approach is suitable as a first-pass screening tool in academic and contest environments, flagging unusual submissions for deeper analysis. Further research should validate these features against labeled AI-generated datasets and expand the framework to incorporate semantic and citation-aware metrics.
