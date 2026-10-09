"""Server-side markdown rendering for report sections shown in the browser.

Reports are authored as markdown; the CLI renders them with rich, the web UI
renders them here (once, server-side) so the browser never needs a markdown
library or a CDN dependency.
"""

from __future__ import annotations

from markdown_it import MarkdownIt

# commonmark + GFM tables, which the reports use for rating/score grids.
# linkify stays off: it would need linkify-it-py, and bare URLs are rare here.
# Raw HTML is OFF: report text is LLM output shaped by fetched news and social
# posts, so an injected <img onerror=…> must render as text, never as markup
# in a page that can store keys and place orders.
_MD = MarkdownIt("commonmark", {"html": False}).enable("table")


def render_markdown(text: str) -> str:
    """Render markdown to a HTML fragment. Never raises on odd input."""
    if not text:
        return ""
    if not isinstance(text, str):
        text = "\n".join(str(item) for item in text) if isinstance(text, list) else str(text)
    return _MD.render(text)
