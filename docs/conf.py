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
    "sphinx_design",
    "sphinx_copybutton",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
myst_enable_extensions = ["dollarmath", "colon_fence", "attrs_inline", "deflist"]
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
html_logo = "_static/logo.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
copybutton_prompt_text = r"\$ |>>> |\.\.\. "
copybutton_prompt_is_regexp = True
html_theme_options = {
    "source_repository": "https://github.com/JuanBSLeite/Jax-PWA/",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view", "edit"],
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/JuanBSLeite/Jax-PWA",
            "html": "<svg stroke='currentColor' fill='currentColor' stroke-width='0' viewBox='0 0 16 16'><path fill-rule='evenodd' d='M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z'></path></svg>",
            "class": "",
        },
    ],
}


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
