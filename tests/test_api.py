"""The public surface: every advertised name imports and is documented."""

import importlib
import pkgutil

import pytest

import multipgs


def test_every_exported_name_resolves():
    for name in multipgs.__all__:
        assert getattr(multipgs, name) is not None


def test_no_export_collides_with_its_own_module_name():
    """A module named after one of its exports resolves to whichever was
    imported last. It is the one packaging mistake this layout can make."""
    modules = set(multipgs._EXPORTS)
    names = set(multipgs._NAME_TO_MODULE)
    assert modules.isdisjoint(names), (
        f"module name(s) {sorted(modules & names)} shadow an exported name")


def test_importing_a_submodule_first_does_not_break_the_export():
    import multipgs.metrics                      # noqa: F401
    fresh = importlib.reload(multipgs)
    assert callable(fresh.evaluate)


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        multipgs.nope


def test_dir_lists_the_public_names():
    listed = dir(multipgs)
    for name in ("multi_pgs_fit", "meta_pgs", "panel_from_catalog",
                 "evaluate", "screen"):
        assert name in listed


def test_every_module_and_public_function_has_a_docstring():
    for info in pkgutil.iter_modules(multipgs.__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"multipgs.{info.name}")
        assert module.__doc__, f"multipgs.{info.name} has no module docstring"
        for name in getattr(module, "__all__", []):
            obj = getattr(module, name)
            assert obj.__doc__, f"{info.name}.{name} has no docstring"


def test_version_is_a_release_string():
    parts = multipgs.__version__.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts)


def test_readme_examples_name_real_functions():
    """Guard against the README drifting away from the API."""
    import pathlib
    import re
    readme = pathlib.Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")
    for match in re.findall(r"from multipgs import ([^\n]+)", text):
        for name in (n.strip() for n in match.split(",")):
            if name and name.isidentifier():
                assert hasattr(multipgs, name), \
                    f"README imports multipgs.{name}, which does not exist"
