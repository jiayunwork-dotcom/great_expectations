"""Shared harness for the dataframe-chunk streaming validation tests."""

from __future__ import annotations

from typing import Any, Iterable, List, Tuple

import pandas as pd

from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.core.validation_definition import ValidationDefinition


class CountingSource:
    """A one-shot synchronous source that records iterations and closes.

    ``chunks`` may be a concrete list or a factory returning an iterable; the source hands
    out a fresh iterator per ``__iter__`` call and counts each call so tests can assert the
    engine obtains exactly one iterator.
    """

    def __init__(self, chunks: Iterable[pd.DataFrame]) -> None:
        self._chunks = list(chunks)
        self.iter_count = 0
        self.close_count = 0
        self.yield_count = 0

    def __iter__(self):
        self.iter_count += 1
        source = self

        class _Iterator:
            def __init__(self) -> None:
                self._inner = iter(source._chunks)

            def __iter__(self):
                return self

            def __next__(self):
                chunk = next(self._inner)
                source.yield_count += 1
                return chunk

            def close(self) -> None:
                source.close_count += 1

        return _Iterator()


class GeneratorSource:
    """A true one-shot generator whose ``close`` is observable."""

    def __init__(self, chunks: Iterable[pd.DataFrame]) -> None:
        self._chunks = list(chunks)
        self.iter_count = 0
        self.closed = 0

    def __iter__(self):
        self.iter_count += 1
        chunks = self._chunks
        source = self

        def _gen():
            try:
                for chunk in chunks:
                    yield chunk
            finally:
                source.closed += 1

        return _gen()


def make_asset_batch_definition(context: Any, name: str = "stream_ds") -> Tuple[Any, Any, Any]:
    datasource = context.data_sources.add_pandas(name=name)
    asset = datasource.add_dataframe_stream_asset(name=f"{name}_asset")
    batch_definition = asset.add_batch_definition_whole_dataframe_stream(
        name=f"{name}_bd"
    )
    return datasource, asset, batch_definition


def build_stream_validation_definition(
    context: Any,
    suite: ExpectationSuite,
    name: str = "stream_vd",
) -> ValidationDefinition:
    _, _, batch_definition = make_asset_batch_definition(context, name=f"{name}_ds")
    suite = context.suites.add(suite)
    return context.validation_definitions.add(
        ValidationDefinition(name=name, data=batch_definition, suite=suite)
    )


def chunk(full: pd.DataFrame, sizes: List[int]) -> List[pd.DataFrame]:
    """Split ``full`` positionally into chunks of the given sizes.

    ``sizes`` must sum to ``len(full)`` unless the last entry is omitted; an empty list
    size ``0`` produces a zero-row chunk that preserves schema.
    """
    out: List[pd.DataFrame] = []
    start = 0
    for size in sizes:
        out.append(full.iloc[start : start + size])
        start += size
    if start < len(full):
        out.append(full.iloc[start:])
    return out


def per_row(full: pd.DataFrame) -> List[pd.DataFrame]:
    return [full.iloc[i : i + 1] for i in range(len(full))]


def unequal_chunks(full: pd.DataFrame) -> List[pd.DataFrame]:
    sizes = [1, 0, 3, 2, 5, 1]
    return chunk(full, sizes)


__all__ = [
    "CountingSource",
    "GeneratorSource",
    "build_stream_validation_definition",
    "chunk",
    "make_asset_batch_definition",
    "per_row",
    "unequal_chunks",
]
