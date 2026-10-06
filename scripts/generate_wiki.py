"""Generate the GitHub wiki API reference from `docs/catalog.md` and docstrings.

One wiki page per `## Section` of the catalog: the section's introductory
text, then every public name of its tables (in table order) with its module,
signature, full docstring and public methods/properties, then the section's
"Docs:" line. `Home.md` links the pages. Relative documentation links are
rewritten to absolute GitHub URLs, since wiki pages live in another repository.

    python scripts/generate_wiki.py ../Jax-PWA.wiki    # then commit/push the wiki repo

Fails if a name exported in `jaxpwa.__all__` is missing from the catalog.
"""

from __future__ import annotations

import argparse
import inspect
import re
from pathlib import Path

import jaxpwa

REPOSITORY = "https://github.com/JuanBSLeite/Jax-PWA/blob/main"
ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs" / "catalog.md"


def page_name(title: str) -> str:
    """GitHub wiki page name of a section title ('Vetoes', 'ROOT-tree-I-O', ...)."""
    return re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-")


def absolute_links(text: str) -> str:
    """Rewrite catalog-relative markdown links to absolute repository URLs."""

    def rewrite(match):
        label, target = match.group(1), match.group(2)
        if re.match(r"[a-z]+://|#", target):
            return match.group(0)
        if target.startswith("../"):
            path = target[3:]
        else:
            path = f"docs/{target}"
        return f"[{label}]({REPOSITORY}/{path})"

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", rewrite, text)


def parse_catalog(text: str):
    """[(title, intro, [names], docs_line)] for every '## ' section.

    A table row's first cell may list aliases (`ZemachP` / `Zemach_P`); each
    name is a list of (name, aliases).
    """
    sections = []
    for block in re.split(r"^## ", text, flags=re.MULTILINE)[1:]:
        title, _, body = block.partition("\n")
        intro, names, docs = [], [], ""
        in_table = False
        for line in body.splitlines():
            if line.startswith("|"):
                in_table = True
                cell = line.split("|")[1]
                found = re.findall(r"`([A-Za-z_][A-Za-z0-9_.]*)`", cell)
                if found:
                    names.append((found[0], tuple(found[1:])))
                continue
            if line.startswith("Docs:"):
                docs = line
            elif not in_table and line.strip():
                intro.append(line)
        sections.append((title.strip(), "\n".join(intro), names, docs))
    return sections


def signature(obj) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(...)"


def public_members(cls):
    """[(name, signature, first docstring line)] of the public methods/properties."""
    members = []
    for name, member in sorted(vars(cls).items()):
        if name.startswith("_"):
            continue
        if isinstance(member, property):
            function, sig = member.fget, None
        elif isinstance(member, (classmethod, staticmethod)):
            function = member.__func__
            sig = signature(function)
        elif inspect.isfunction(member):
            function, sig = member, signature(member)
        else:
            continue
        doc = inspect.getdoc(function) or ""
        first = doc.strip().splitlines()[0] if doc.strip() else ""
        if sig is None:
            sig = signature(function) if function is not None else "(self)"
        members.append((name, sig, first))
    return members


def entry(name: str, aliases=()) -> str:
    obj = getattr(jaxpwa, name)
    lines = [f"### `{name}`", "", f"*Defined in* `{getattr(obj, '__module__', 'jaxpwa')}`", ""]
    if aliases:
        lines += ["Also exported as " + ", ".join(f"`{alias}`" for alias in aliases) + " (the same object).", ""]
    if inspect.isclass(obj) or callable(obj):
        lines += ["```python", f"{name}{signature(obj)}", "```", ""]
    doc = inspect.getdoc(obj)
    if doc and not (inspect.isclass(obj) and doc.startswith(f"{name}(")):
        lines += [doc, ""]
    if inspect.isclass(obj):
        members = public_members(obj)
        if members:
            lines += ["**Public methods/properties:**", ""]
            for member, sig, first in members:
                lines.append(f"- `{member}{sig}`")
                if first:
                    lines.append(f"  {first}")
            lines.append("")
    return "\n".join(lines)


def section_page(title, intro, names, docs) -> str:
    parts = [f"# {title}", ""]
    if intro:
        parts += [absolute_links(intro), ""]
    for name, aliases in names:
        if name in jaxpwa.__all__:
            parts.append(entry(name, aliases))
    if docs:
        parts += ["---", "", absolute_links(docs), ""]
    return "\n".join(parts).rstrip() + "\n"


def home_page(sections) -> str:
    lines = [
        "# Jax-PWA API reference",
        "",
        "Auto-generated by `scripts/generate_wiki.py` from the public API's own docstrings "
        "(`import jaxpwa; jaxpwa.__all__`), organized the same way as "
        f"[`docs/catalog.md`]({REPOSITORY}/docs/catalog.md). This is a browsable reference, not a "
        "tutorial -- for conventions, formulas and worked examples, follow the `docs/*.md` links "
        "quoted on each page back to the main repository, and see the "
        f"[nine-part notebook course]({REPOSITORY}/notebooks/tutorials/TUTORIALS.md) for a guided "
        "introduction.",
        "",
        "## Sections",
        "",
    ]
    lines += [f"- [{title}]({page_name(title)})" for title, *_ in sections]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wiki_dir", type=Path, help="checkout of the Jax-PWA.wiki repository")
    args = parser.parse_args()

    sections = parse_catalog(CATALOG.read_text())
    listed = {item for _, _, names, _ in sections for name, aliases in names for item in (name, *aliases)}
    missing = sorted(set(jaxpwa.__all__) - listed)
    if missing:
        raise SystemExit(f"names in jaxpwa.__all__ missing from docs/catalog.md: {missing}")

    args.wiki_dir.mkdir(parents=True, exist_ok=True)
    written = {"Home.md"}
    (args.wiki_dir / "Home.md").write_text(home_page(sections))
    for title, intro, names, docs in sections:
        filename = f"{page_name(title)}.md"
        (args.wiki_dir / filename).write_text(section_page(title, intro, names, docs))
        written.add(filename)
    stale = sorted(path.name for path in args.wiki_dir.glob("*.md") if path.name not in written)
    print(f"wrote {len(written)} pages to {args.wiki_dir}")
    if stale:
        print(f"pages not generated from the catalog (left untouched): {stale}")


if __name__ == "__main__":
    main()
