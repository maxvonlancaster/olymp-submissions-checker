"""Corrected stylometric feature extraction used by the dataset pipeline."""

from __future__ import annotations

import ast
import hashlib
import io
import math
import re
import tokenize

FEATURES = ("avg_var_length", "comment_density", "num_variables", "indentation_consistency", "repetition_score", "overexplanation_score")


def extract_style_features(code: str, language: str = ".py"):
    if language == ".py":
        return _extract_python_features(code)
    if language in [".cpp", ".cs", ".pas", ".c"]:
        return _extract_cpp_features(code, language)
    raise ValueError(f"Unsupported language: {language}")


def _empty_variable_features(comment_density=0.0):
    return {"avg_var_length": 0.0, "comment_density": float(comment_density), "num_variables": 0}


def _comment_data(code: str, language: str):
    if not code:
        return "", [], set()
    if language == ".py":
        masked = list(code)
        comments = []
        comment_lines = set()
        line_offsets = [0]
        for line in code.splitlines(True):
            line_offsets.append(line_offsets[-1] + len(line))
        try:
            tokens = tokenize.generate_tokens(io.StringIO(code).readline)
            for token in tokens:
                if token.type == tokenize.COMMENT:
                    comments.append(token.string[1:].strip())
                    comment_lines.add(token.start[0])
                    start = line_offsets[token.start[0] - 1] + token.start[1]
                    end = line_offsets[token.end[0] - 1] + token.end[1]
                    for index in range(start, min(end, len(masked))):
                        if masked[index] not in "\n\r":
                            masked[index] = " "
            return "".join(masked), comments, comment_lines
        except (tokenize.TokenError, IndentationError):
            return "", [], set()
    comment_pairs = {".cpp": [("//", "\n"), ("/*", "*/")], ".c": [("//", "\n"), ("/*", "*/")], ".cs": [("//", "\n"), ("/*", "*/")], ".pas": [("//", "\n"), ("{", "}"), ("(*", "*)")]}
    pairs = comment_pairs[language]
    masked = list(code)
    comments = []
    comment_lines = set()
    index = 0
    while index < len(code):
        if code[index] in "\"'":
            quote = code[index]
            index += 1
            while index < len(code):
                if code[index] == "\\":
                    index += 2
                elif code[index] == quote:
                    index += 1
                    break
                else:
                    if code[index] != "\n":
                        masked[index] = " "
                    index += 1
            continue
        matched = next(((start, end) for start, end in pairs if code.startswith(start, index)), None)
        if matched is None:
            index += 1
            continue
        start_marker, end_marker = matched
        start = index + len(start_marker)
        end = code.find(end_marker, start) if end_marker != "\n" else code.find("\n", start)
        end = len(code) if end < 0 else end
        comments.append(code[start:end].strip())
        start_line = code[:index].count("\n") + 1
        end_line = code[:end].count("\n") + 1
        comment_lines.update(range(start_line, end_line + 1))
        for position in range(index, min(end + (len(end_marker) if end_marker != "\n" and end < len(code) else 0), len(masked))):
            if masked[position] != "\n":
                masked[position] = " "
        index = end + (len(end_marker) if end_marker != "\n" and end < len(code) else 0)
    return "".join(masked), comments, comment_lines


def _python_bindings(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.alias) and node.asname:
            names.add(node.asname)
    return names


def _extract_python_features(code: str):
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError):
        _, _, comment_lines = _comment_data(code, ".py")
        line_count = len(code.splitlines())
        return _empty_variable_features(len(comment_lines) / line_count if line_count else 0.0)
    _, _, comment_lines = _comment_data(code, ".py")
    var_names = _python_bindings(tree)
    avg_len = sum(map(len, var_names)) / len(var_names) if var_names else 0.0
    line_count = len(code.splitlines())
    return {"avg_var_length": avg_len, "comment_density": len(comment_lines) / line_count if line_count else 0.0, "num_variables": len(var_names)}


def _split_declarators(text: str):
    parts, start, depth = [], 0, 0
    for index, character in enumerate(text):
        if character == "(":
            depth += 1
        elif character == ")":
            depth = max(0, depth - 1)
        elif character == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return parts


def _extract_cpp_features(code: str, language: str = ".cpp"):
    code_clean, _, comment_lines = _comment_data(code, language)
    var_names = set()
    if language == ".pas":
        for section in re.findall(r"(?is)\bvar\b(.*?)(?=\bbegin\b|\bprocedure\b|\bfunction\b|\bconst\b|\btype\b|\Z)", code_clean):
            for match in re.finditer(r"\b([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*:\s*[A-Za-z_]\w*", section):
                var_names.update(re.findall(r"[A-Za-z_]\w*", match.group(1)))
    else:
        qualified_type = r"(?:[A-Za-z_]\w*(?:(?:::|\.)[A-Za-z_]\w*)+)(?:\s*<[^;{}]*>)?"
        simple_type = r"(?:bool|boolean|byte|char|double|float|int|long|short|decimal|object|string|void|var|auto|size_t|uint|uint32_t|int32_t|DateTime)"
        type_pattern = rf"(?:{simple_type}|{qualified_type})"
        qualifiers = r"(?:(?:const|static|constexpr|volatile|unsigned|signed|long|short)\s+)*"
        declaration = re.compile(r"\b" + qualifiers + type_pattern + r"\s+([^;{}]+);", re.MULTILINE)
        for match in declaration.finditer(code_clean):
            for declarator in _split_declarators(match.group(1)):
                name = re.match(r"\s*(?:[*&]+\s*)?([A-Za-z_]\w*)\b", declarator)
                if not name:
                    continue
                open_paren, equals = declarator.find("("), declarator.find("=")
                if open_paren >= 0 and (equals < 0 or open_paren < equals):
                    continue
                var_names.add(name.group(1))
    avg_len = sum(map(len, var_names)) / len(var_names) if var_names else 0.0
    line_count = len(code.splitlines())
    return {"avg_var_length": avg_len, "comment_density": len(comment_lines) / line_count if line_count else 0.0, "num_variables": len(var_names)}


def indentation_consistency(code: str, features: dict):
    indents = [len(line) - len(line.lstrip(" ")) for line in code.splitlines() if len(line) - len(line.lstrip(" ")) > 0]
    if len(indents) < 2:
        features["indentation_consistency"] = 1.0
        return features
    mean = sum(indents) / len(indents)
    variance = sum((indent - mean) ** 2 for indent in indents) / len(indents)
    features["indentation_consistency"] = 1.0 / (1.0 + variance)
    return features


def repetition_score(code: str, features: dict):
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    features["repetition_score"] = (1.0 - len(set(lines)) / len(lines)) if lines else 0.0
    return features


def overexplanation_in_comments(code: str, extension: str, features: dict):
    _, comments, _ = _comment_data(code, extension)
    features["overexplanation_score"] = sum(len(comment.split()) for comment in comments) / len(comments) if comments else 0.0
    return features


def analyze_code_style(code: str, language: str = ".py"):
    features = extract_style_features(code, language)
    features = indentation_consistency(code, features)
    features = repetition_score(code, features)
    return overexplanation_in_comments(code, language, features)


def normalize_code(code: str) -> str:
    normalized = code.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n"))


def normalized_code_sha256(code: str) -> str:
    return hashlib.sha256(normalize_code(code).encode("utf-8")).hexdigest()


def is_valid_source(code: str, language: str) -> bool:
    if language == ".py":
        try:
            ast.parse(code)
        except (SyntaxError, ValueError, TypeError):
            return False
    return True


def validate_features(features: dict) -> None:
    if set(features) != set(FEATURES):
        raise ValueError(f"Feature keys mismatch: {sorted(features)}")
    for name in FEATURES:
        value = features[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Malformed {name}: {value!r}")
    if not isinstance(features["num_variables"], int):
        if float(features["num_variables"]).is_integer():
            features["num_variables"] = int(features["num_variables"])
        else:
            raise ValueError("num_variables is not integral")