"""Pytest plugin: make the en_US.UTF-8 setlocale call in tests/conftest.py a no-op.

This container only ships C.UTF-8, but the GE test configuration assumes en_US.
Number formatting expectations in our tests do not depend on the locale.
"""
import locale as _locale

_real_setlocale = _locale.setlocale


def _patched_setlocale(category, loc=None):
    if loc == "en_US.UTF-8":
        return _real_setlocale(category, "C.UTF-8")
    return _real_setlocale(category, loc)


_locale.setlocale = _patched_setlocale
