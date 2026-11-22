"""Process-local one-shot bookkeeping for dataframe-chunk sources.

A dataframe-chunk source is only guaranteed to be consumable once.  We must reject a
second validation over the *same source object* even though the ``Batch`` is re-fetched
from the asset between ``Batch.validate``/``ValidationDefinition.run`` calls (each fetch
builds a fresh transient ``StreamingBatchData`` over the same source).

The registry deliberately lives outside any ``BatchDefinition``/suite/asset model so no
run state is ever persisted.  Entries are keyed by object identity and hold a weak
reference: a claimed, exhausted generator can be garbage-collected normally, while a live
source object keeps its claim (generators stay alive while referenced, so the weak ref is
non-empty for as long as the caller can possibly re-submit it).  If the object is collected
its id may be reused; a freshly created source in that slot is a genuinely new object and
is allowed its own single scan.
"""

from __future__ import annotations

import threading
import weakref
from typing import Any, Dict, Optional

from great_expectations.exceptions import StreamSourceReuseError

_claimed_sources: Dict[int, weakref.ReferenceType] = {}
_lock = threading.Lock()


def _lookup(key: int, source: Any) -> Optional[weakref.ReferenceType]:
    ref = _claimed_sources.get(key)
    if ref is None:
        return None
    # If the original object has been collected, the id may have been recycled; only treat
    # it as the same source when the weak reference still resolves to ``source``.
    if ref() is not source:
        _claimed_sources.pop(key, None)
        return None
    return ref


def claim_source(source: Any) -> None:
    """Claim ``source`` for its one allowed scan.

    Raises:
        StreamSourceReuseError: if this exact object already produced a scan.
    """
    key = id(source)
    with _lock:
        if _lookup(key, source) is not None:
            raise StreamSourceReuseError()
        try:
            _claimed_sources[key] = weakref.ref(source)
        except TypeError:
            # Not weak-referenceable: fall back to a strong reference so the claim is still
            # enforced for the lifetime of the process.
            _claimed_sources[key] = _StrongRef(source)


class _StrongRef:
    __slots__ = ("_obj",)

    def __init__(self, obj: Any) -> None:
        self._obj = obj

    def __call__(self) -> Any:
        return self._obj


def is_claimed(source: Any) -> bool:
    key = id(source)
    with _lock:
        return _lookup(key, source) is not None


def reset_registry() -> None:  # pragma: no cover - test helper
    with _lock:
        _claimed_sources.clear()


__all__ = ["claim_source", "is_claimed", "reset_registry"]
