"""Enforce naming and documentation conventions for the brain_3d package."""

import ast
import re
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "brain_3d"
PASCAL_CASE = re.compile(r"^[A-Z][A-Za-z0-9]*$")
REQUIRED_LOWERCASE_METHODS = {
    "__getattr__",
    "__getitem__",
    "__init__",
    "__len__",
    "forward",
    "test",
    "train",
}


def test_package_api_naming_and_docstrings():
    """Require documented modules, classes, functions, methods, and helpers."""

    issues = []
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if ast.get_docstring(tree) is None:
            issues.append(f"{path.name}: missing module docstring")

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name.lstrip("_")
                if not PASCAL_CASE.fullmatch(class_name):
                    issues.append(f"{path.name}:{node.lineno}: invalid class name {node.name}")
                if ast.get_docstring(node) is None:
                    issues.append(f"{path.name}:{node.lineno}: missing class docstring {node.name}")

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_required_name = node.name in REQUIRED_LOWERCASE_METHODS or node.name.startswith("__")
                is_public = PASCAL_CASE.fullmatch(node.name) is not None
                if not (is_required_name or is_public):
                    issues.append(f"{path.name}:{node.lineno}: invalid function name {node.name}")
                if ast.get_docstring(node) is None:
                    issues.append(f"{path.name}:{node.lineno}: missing function docstring {node.name}")

    assert not issues, "\n" + "\n".join(issues)


def test_source_modules_do_not_contain_campaign_configuration():
    """Keep editable paths and experiment selection out of reusable modules."""
    forbidden = (
        "PARAMETERS TO EDIT",
        "/data/",
        "analisi_dati_Mark",
        "analisi_dati_3Brain",
        "EXPERIMENT_CONFIG",
    )
    issues = []
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for marker in forbidden:
            if marker in source:
                issues.append(f"{path.name}: campaign configuration marker {marker!r}")
    assert not issues, "\n" + "\n".join(issues)


def test_docstring_parameter_names_match_function_signatures():
    """Require documented parameter names to preserve signature capitalization."""
    issues = []
    section_end = {"Returns", "Raises", "Notes", "Examples", "Attributes", "Yields"}
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            signature_names = {
                argument.arg
                for argument in node.args.args + node.args.kwonlyargs
            }
            lines = (ast.get_docstring(node) or "").splitlines()
            documented = []
            for index, line in enumerate(lines):
                if line.strip() not in {"Parameters", "Args"}:
                    continue
                cursor = index + 2
                while cursor < len(lines):
                    raw = lines[cursor]
                    if raw.strip() in section_end:
                        break
                    if raw and not raw[0].isspace():
                        match = re.match(r"^([*A-Za-z_][A-Za-z0-9_, *]*)\s*:", raw)
                        if match:
                            documented.extend(
                                name.strip().lstrip("*")
                                for name in match.group(1).split(",")
                            )
                    cursor += 1
            for name in documented:
                if name and name not in signature_names:
                    issues.append(
                        f"{path.name}:{node.lineno}: documented {name!r} is not in "
                        f"the {node.name} signature"
                    )
    assert not issues, "\n" + "\n".join(issues)


def test_source_imports_are_declared_at_module_level():
    """Prevent imports from being hidden inside functions or classes."""
    issues = []
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                child.parent = parent
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            parent = getattr(node, "parent", None)
            while parent is not None and not isinstance(parent, ast.Module):
                if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    issues.append(
                        f"{path.name}:{node.lineno}: import must be declared at module level"
                    )
                    break
                parent = getattr(parent, "parent", None)
    assert not issues, "\n" + "\n".join(issues)
