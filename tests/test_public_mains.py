"""Structural tests for the configurable analysis scripts."""

import ast
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = REPOSITORY_ROOT / "main"


def test_all_mains_use_editable_configuration_without_argument_parsing():
    """Keep main scripts directly configurable and free of CLI parsers."""
    for script in sorted(MAIN_ROOT.glob("main_*.py")):
        source = script.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert "argparse" not in imports, script.name
        assert "parse_args" not in source, script.name
        assert 'if __name__ == "__main__":' in source, script.name


def test_main_scripts_do_not_define_functions():
    """Keep main scripts limited to configuration and orchestration statements."""
    issues = []
    for script in sorted(MAIN_ROOT.glob("main_*.py")):
        tree = ast.parse(script.read_text(encoding="utf-8"))
        definitions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for definition in definitions:
            issues.append(f"{script.name}:{definition.lineno}: {definition.name}")
    assert not issues, "\n" + "\n".join(issues)


def test_public_mains_do_not_contain_project_specific_data_paths():
    """Prevent personal or campaign-specific paths from entering shared mains."""
    forbidden = ("/data/", "analisi_dati_Mark", "analisi_dati_3Brain")
    for script in sorted(MAIN_ROOT.glob("main_*.py")):
        source = script.read_text(encoding="utf-8")
        assert not any(value in source for value in forbidden), script.name
