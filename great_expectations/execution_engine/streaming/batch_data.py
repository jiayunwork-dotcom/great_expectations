from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Iterator, Optional

from great_expectations.core.batch import BatchData
from great_expectations.exceptions import StreamSourceReuseError

if TYPE_CHECKING:
    from great_expectations.core.batch_spec import DataframeChunksBatchSpec
    from great_expectations.execution_engine.execution_engine import ExecutionEngine


class StreamingBatchData(BatchData):
    """BatchData placeholder for a one-shot dataframe chunk stream.

    Creating this object (and the enclosing ``Batch``) never reads the source.  The source
    is only touched when :meth:`acquire_iterator` is called by a validation run, and that
    path is guarded so at most one scan ever consumes a given source object -- across every
    ``Batch``/``StreamingBatchData`` that wraps it.
    """

    def __init__(
        self,
        execution_engine: ExecutionEngine,
        dataframe_chunks: Iterable[Any],
        batch_spec: Optional[DataframeChunksBatchSpec] = None,
    ) -> None:
        super().__init__(execution_engine=execution_engine)
        self._dataframe_chunks = dataframe_chunks
        self._batch_spec = batch_spec
        self._iterated = False

    @property
    def batch_spec(self) -> Optional[DataframeChunksBatchSpec]:
        return self._batch_spec

    @property
    def dataframe_chunks(self) -> Iterable[Any]:
        return self._dataframe_chunks

    @property
    def consumed(self) -> bool:
        from great_expectations.execution_engine.streaming import source_registry

        return self._iterated or source_registry.is_claimed(self._dataframe_chunks)

    def acquire_iterator(self) -> Iterator[Any]:
        """Claim the source and return its single allowed iterator.

        Raises:
            StreamSourceReuseError: if this source object was already claimed by an earlier
                scan (whether that scan exhausted it, failed, or was aborted).
        """
        from great_expectations.execution_engine.streaming import source_registry

        if self._iterated:
            raise StreamSourceReuseError()
        # Claim *before* obtaining the iterator so a later Batch wrapping the same
        # exhausted/failed source is rejected even if the source's own ``__iter__`` would
        # silently yield nothing.
        source_registry.claim_source(self._dataframe_chunks)
        self._iterated = True
        return iter(self._dataframe_chunks)

