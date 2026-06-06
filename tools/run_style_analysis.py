import os
import re
import ast
import json
import pandas as pd
import numpy as np

# Feature extraction functions (adapted from the analysis prototype)

def extract_style_features(code: str, language: str = ".py"):
    if language == ".py":
        return _extract_python_features(code)
    elif language in [".cpp", ".cs", ".pas", ".c", ".java"]:
        return _extract_cpp_features(code)
    else:
        return _fallback_features(code)


def _extract_python_features(code: str):
    try:
        tree = ast.parse(code)
    except Exception:
        return _fallback_features(code)

    var_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            var_names.append(node.id)

    avg_len = sum(len(v) for v in var_names) / (len(var_names) + 1)
    comment_count = len(re.findall(r"#.*", code))
    lines = len(code.split('\n'))
    comment_density = comment_count / (lines + 1)

    return {
        "avg_var_length": avg_len,
        "comment_density": comment_density,
        "num_variables": len(var_names)
    }


def _fallback_features(code: str):
    words = re.findall(r"\b[a-zA-Z_]{2,}\b", code)
    avg_len = sum(len(w) for w in words) / (len(words) + 1)
    comment_count = len(re.findall(r"#|//", code))
    lines = len(code.split('\n'))
    return {
        "avg_var_length": avg_len,
        "comment_density": comment_count / (lines + 1),
        "num_variables": len(words)
    }


def _extract_cpp_features(code: str):
    # Remove strings (to avoid false positives)
    code_clean = re.sub(r'"(\\.|[^\"])*"', '', code)

    var_pattern = r'\b(int|float|double|char|bool|long|auto|string)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    matches = re.findall(var_pattern, code_clean)
    var_names = [m[1] for m in matches]
    avg_len = sum(len(v) for v in var_names) / (len(var_names) + 1)

    single_comments = re.findall(r'//.*', code)
    multi_comments = re.findall(r'/\*[\s\S]*?\*/', code)
    comment_count = len(single_comments) + len(multi_comments)

    lines = len(code.split('\n'))
    comment_density = comment_count / (lines + 1)

    return {
        "avg_var_length": avg_len,
        "comment_density": comment_density,
        "num_variables": len(var_names)
    }


def indentation_consistency(code: str, features: dict):
    indents = []
    for line in code.split('\n'):
        spaces = len(line) - len(line.lstrip(' '))
        if spaces > 0:
            indents.append(spaces)
    if not indents:
        features["indentation_consistency"] = 0.5
        return features
    mu = sum(indents) / len(indents)
    variance = sum((x - mu) ** 2 for x in indents) / len(indents)
    features["indentation_consistency"] = 1 / (1 + variance)
    return features


def repetition_score(code: str, features: dict):
    lines = [l.strip() for l in code.split('\n') if l.strip()]
    unique = set(lines)
    features["repetition_score"] = 1 - len(unique) / (len(lines) + 1)
    return features


def overexplanation_in_comments(code: str, extension: str, features: dict):
    if extension == ".py":
        comments = re.findall(r'#.*', code)
    else:
        comments = re.findall(r'//.*|/\*[\s\S]*?\*/', code)
    overexplained = sum(len(c.split()) for c in comments) / len(comments) if comments else 0
    features["overexplanation_score"] = overexplained
    return features


def analyze_code_style(code: str, language: str = ".py"):
    features = extract_style_features(code, language)
    features = indentation_consistency(code, features)
    features = repetition_score(code, features)
    features = overexplanation_in_comments(code, language, features)
    return features


# Language mapping utility
def map_language_to_extension(lang_str: str) -> str:
    if not isinstance(lang_str, str):
        return ".txt"
    s = lang_str.strip().upper()
    # Remove version suffixes if present (take first token before whitespace)
    token = s.split()[0]
    if 'C++' in token or token.startswith('CPP'):
        return '.cpp'
    if token.startswith('JAVA'):
        return '.java'
    if token.startswith('PY') or token.startswith('PYTH'):
        return '.py'
    if token == 'C':
        return '.c'
    # fallback
    return '.txt'


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_dir = os.path.join(root, 'datasets')
    third_csv = os.path.join(datasets_dir, 'third.csv')
    solutions_csv = os.path.join(root, 'solutions.csv')

    if not os.path.exists(third_csv):
        print('Error: datasets/third.csv not found at', third_csv)
        return
    if not os.path.exists(solutions_csv):
        print('Error: solutions.csv not found at', solutions_csv)
        return

    df_third = pd.read_csv(third_csv)
    df_solutions = pd.read_csv(solutions_csv)

    # Expecting columns: Solutions, SolutionID in third.csv and SolutionID, Language in solutions.csv
    if 'Solutions' not in df_third.columns or 'SolutionID' not in df_third.columns:
        print('third.csv must contain columns: Solutions, SolutionID')
        return
    if 'SolutionID' not in df_solutions.columns or 'Language' not in df_solutions.columns:
        print('solutions.csv must contain columns: SolutionID, Language')
        return

    df = pd.merge(df_third, df_solutions[['SolutionID', 'Language']], on='SolutionID', how='left')

    results = []
    errors = []
    for idx, row in df.iterrows():
        sol_id = row.get('SolutionID')
        code = str(row.get('Solutions', ''))
        lang = row.get('Language', '')
        ext = map_language_to_extension(lang)
        try:
            feats = analyze_code_style(code, ext)
            feats['SolutionID'] = sol_id
            feats['Language'] = lang
            results.append(feats)
        except Exception as e:
            errors.append({'SolutionID': sol_id, 'error': str(e)})

    out_json = os.path.join(root, 'style_analysis_results.json')
    out_csv = os.path.join(root, 'style_analysis_results.csv')

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    pd.DataFrame(results).to_csv(out_csv, index=False)

    print('Processed', len(results), 'solutions')
    if errors:
        print('Errors for', len(errors), 'solutions. See errors list.')
        with open(os.path.join(root, 'style_analysis_errors.json'), 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2)


if __name__ == '__main__':
    main()
