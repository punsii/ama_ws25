"""CLI script to extract lightweight context for the ama-tlbx package.

Modes
-----
- packages: Overview of symbols per module (classes, functions, CONSTANTS)
- classes : Classes with docstrings (first paragraph) and public method names

Implementation uses Python's AST only (no imports/exec), so it's safe and fast.
Outputs Markdown to stdout for copy-paste into issues or docs.
"""

import argparse
import ast
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


EXCLUDE_PATHS: list[str] = ["__init__.py"]


@dataclass
class FunctionInfo:
    name: str
    doc: str | None = None


@dataclass
class ClassInfo:
    name: str
    doc: str | None = None
    methods: list[FunctionInfo] = field(default_factory=list)


@dataclass
class ModuleInfo:
    module: str  # dotted path relative to root
    path: Path
    doc: str | None
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    constants: list[str] = field(default_factory=list)


def _is_constant_name(name: str) -> bool:
    return name.isupper() and name.replace("_", "").isalpha()


def _first_paragraph(s: str | None) -> str | None:
    if not s:
        return None
    s = s.strip()
    parts = [p.strip() for p in s.split("\n\n") if p.strip()]
    return parts[0] if parts else s


def parse_module(py_path: Path, root: Path) -> ModuleInfo | None:
    try:
        text = py_path.read_text(encoding="utf-8")
    except Exception:
        return None
    try:
        node = ast.parse(text, filename=str(py_path))
    except SyntaxError:
        return None

    mod_doc = ast.get_docstring(node)
    classes: list[ClassInfo] = []
    functions: list[FunctionInfo] = []
    constants: list[str] = []

    for stmt in node.body:
        if isinstance(stmt, ast.ClassDef):
            c_doc = ast.get_docstring(stmt)
            methods: list[FunctionInfo] = []
            for c_stmt in stmt.body:
                if isinstance(c_stmt, ast.FunctionDef) and not c_stmt.name.startswith("_"):
                    methods.append(FunctionInfo(c_stmt.name, ast.get_docstring(c_stmt)))
            classes.append(ClassInfo(stmt.name, c_doc, methods))
        elif isinstance(stmt, ast.FunctionDef):
            if not stmt.name.startswith("_"):
                functions.append(FunctionInfo(stmt.name, ast.get_docstring(stmt)))
        elif isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name) and _is_constant_name(t.id):
                    constants.append(t.id)

    rel = py_path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    module_name = ".".join(parts)
    return ModuleInfo(
        module=module_name,
        path=py_path,
        doc=mod_doc,
        classes=classes,
        functions=functions,
        constants=constants,
    )


def scan_root(root: Path) -> list[ModuleInfo]:
    modules: list[ModuleInfo] = []
    for py in root.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        rel_path = py.relative_to(root).as_posix()
        if rel_path in EXCLUDE_PATHS:
            continue
        info = parse_module(py, root)
        if info and info.module:
            modules.append(info)
    modules.sort(key=lambda m: m.module)
    return modules


def print_packages_overview(mods: list[ModuleInfo]) -> None:
    print("# Package symbol overview\n")
    by_pkg: dict[str, list[ModuleInfo]] = {}
    for m in mods:
        top = m.module.split(".")[0]
        by_pkg.setdefault(top, []).append(m)

    for pkg, mlist in sorted(by_pkg.items()):
        print(f"## {pkg}\n")
        for m in mlist:
            print(f"### {m.module}")
            if m.doc:
                for line in m.doc.strip().splitlines():
                    print(f"> {line}")
                print()
            if m.classes:
                cls_names = ", ".join(c.name for c in m.classes)
                print(f"- Classes: {cls_names}")
            if m.functions:
                fn_names = ", ".join(f.name for f in m.functions)
                print(f"- Functions: {fn_names}")
            if m.constants:
                consts = ", ".join(m.constants)
                print(f"- Constants: {consts}")
            print()


def _get_signature(fn_node: ast.FunctionDef) -> str:
    args = []
    for arg in fn_node.args.args:
        if arg.arg != "self":
            args.append(arg.arg)
    if fn_node.args.vararg:
        args.append("*" + fn_node.args.vararg.arg)
    for kw in fn_node.args.kwonlyargs:
        args.append(kw.arg + "=…")
    if fn_node.args.kwarg:
        args.append("**" + fn_node.args.kwarg.arg)
    return "(" + ", ".join(args) + ")"


def print_classes_with_docs(mods: list[ModuleInfo], max_doc: int, full_doc: bool = False) -> None:
    print("# Classes with docstrings\n")
    for m in mods:
        for c in m.classes:
            if full_doc and c.doc:
                cdoc = c.doc.strip()
            else:
                cdoc = _first_paragraph(c.doc) or "—"
                if not full_doc and len(cdoc) > max_doc:
                    cdoc = cdoc[: max_doc - 1] + "…"
            print(f"## {m.module}.{c.name}")
            print(cdoc)
            if c.methods:
                method_lines = []
                for fn in c.methods:
                    sig = "()"
                    # Try to get signature from AST if possible
                    # But we don't have AST nodes, only docstrings; so signature fallback is ()
                    # Optionally, could parse c.doc for signature, but that's unreliable.
                    method_lines.append(f"- {fn.name}{sig}")
                print("\nMethods:")
                for line in method_lines:
                    print(line)
            print()


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Extract AMA toolbox source context (AST-based)")
    p.add_argument("mode", choices=["packages", "classes"], help="packages: symbol overview; classes: class docs")
    p.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1] / "ama_tlbx"),
        help="Root package directory",
    )
    p.add_argument("--max-doc", type=int, default=400, help="Max characters for docstrings in classes mode")
    p.add_argument(
        "--full-doc",
        action="store_true",
        help="Show full class docstrings instead of truncated first paragraph",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"error: root directory not found: {root}", file=sys.stderr)
        return 2

    mods = scan_root(root)
    if args.mode == "packages":
        print_packages_overview(mods)
    else:
        print_classes_with_docs(mods, args.max_doc, args.full_doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
