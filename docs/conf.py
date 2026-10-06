"""Sphinx configuration of the Jax-PWA documentation (Read the Docs).

The narrative pages are the Markdown files of docs/ (MyST). Two kinds of page
are generated at build time into docs/_generated/ (git-ignored):

- api/<section>.md: one page per `## Section` of docs/catalog.md, with an
  autodoc entry for every public name of its tables (the same source as the
  GitHub wiki, scripts/generate_wiki.py);
- overview.md: a copy of README.md.

Relative links that leave docs/ (notebooks, README, sources) are rewritten to
GitHub URLs, since only docs/ is part of the site.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
GENERATED = DOCS / "_generated"
REPOSITORY = "https://github.com/JuanBSLeite/Jax-PWA/blob/main"

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(ROOT / "src"))

project = "Jax-PWA"
author = "Juan B. S. Leite"
copyright = "2026, Juan B. S. Leite"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
myst_enable_extensions = ["dollarmath", "colon_fence"]
myst_heading_anchors = 4
exclude_patterns = ["_build", "**/.ipynb_checkpoints"]
# Docstrings mix NumPy-style sections and free text.
napoleon_numpy_docstring = True
napoleon_google_docstring = False
autodoc_member_order = "alphabetical"
autodoc_typehints = "signature"
autodoc_default_options = {"members": True, "show-inheritance": False}
# Narrative pages link notebooks and sources (rewritten to GitHub below) and use
# headings for layout, not as a strict hierarchy.
suppress_warnings = ["myst.header", "myst.xref_missing", "autodoc.import_object"]

html_theme = "furo"
html_title = "Jax-PWA"


def _load_wiki_generator():
    spec = importlib.util.spec_from_file_location("generate_wiki", ROOT / "scripts" / "generate_wiki.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rewrite_links(text: str, source: Path) -> str:
    """Relative links of `source` that point outside docs/ -> GitHub URLs."""

    def rewrite(match):
        label, target = match.group(1), match.group(2)
        if re.match(r"[a-z]+:|#", target):
            return match.group(0)
        path, _, anchor = target.partition("#")
        resolved = (source.parent / path).resolve()
        try:
            resolved.relative_to(DOCS)
            return match.group(0)  # a page of this site: MyST resolves it
        except ValueError:
            pass
        try:
            relative = resolved.relative_to(ROOT).as_posix()
        except ValueError:
            return match.group(0)
        return f"[{label}]({REPOSITORY}/{relative}{'#' + anchor if anchor else ''})"

    return re.sub(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)", rewrite, text)


def generate_pages():
    """docs/_generated/{overview.md, api/*.md} from README.md and docs/catalog.md."""
    wiki = _load_wiki_generator()
    api = GENERATED / "api"
    api.mkdir(parents=True, exist_ok=True)
    overview = rewrite_links((ROOT / "README.md").read_text(), ROOT / "README.md")
    (GENERATED / "overview.md").write_text(overview)

    sections = wiki.parse_catalog((DOCS / "catalog.md").read_text())
    pages = []
    for title, intro, names, docs in sections:
        slug = wiki.page_name(title).lower()
        lines = [f"# {title}", ""]
        if intro:
            lines += [rewrite_links(intro, DOCS / "catalog.md"), ""]
        for name, aliases in names:
            if name not in wiki.jaxpwa.__all__:
                continue
            obj = getattr(wiki.jaxpwa, name)
            directive = "autoclass" if isinstance(obj, type) else "autofunction"
            lines += ["```{eval-rst}", f".. {directive}:: jaxpwa.{name}", "```", ""]
            if aliases:
                lines += ["Also exported as " + ", ".join(f"`{alias}`" for alias in aliases)
                          + " (the same object).", ""]
        if docs:
            lines += ["---", "", rewrite_links(docs, DOCS / "catalog.md"), ""]
        (api / f"{slug}.md").write_text("\n".join(lines))
        pages.append(slug)
    toc = ["# API reference", "",
           "One page per section of the [API catalog](../../catalog.md), with the docstrings of every "
           "name exported by `jaxpwa`.", "", "```{toctree}", ":maxdepth: 1", "",
           *pages, "```", ""]
    (api / "index.md").write_text("\n".join(toc))


def _source_read(app, docname, source):
    if docname.startswith("_generated/"):
        return
    source[0] = rewrite_links(source[0], DOCS / f"{docname}.md")


def setup(app):
    generate_pages()
    app.connect("source-read", _source_read)
