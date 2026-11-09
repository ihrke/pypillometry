"""Jinja templates for the HTML displays and reports of pypillometry.

:func:`render` renders one of the templates in this folder (internal)."""
import functools


@functools.lru_cache(maxsize=None)
def _environment():
    """Jinja environment for the templates in this folder (autoescaped, so that
    names and labels with "<" or "&" are shown as text)."""
    import jinja2
    from ..quality import _bounds_text
    env = jinja2.Environment(loader=jinja2.PackageLoader("pypillometry", "templates"),
                             autoescape=True, trim_blocks=True, lstrip_blocks=True)
    env.filters["bounds"] = _bounds_text
    return env


def render(template: str, **context) -> str:
    """Render the template `template` (a file name in this folder) with
    `context`."""
    return _environment().get_template(template).render(**context)
