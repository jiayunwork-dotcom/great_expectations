from great_expectations.constants import (
    DATAFRAME_CHUNKS_REPLACEMENT_STR,
    DATAFRAME_REPLACEMENT_STR,
)
from great_expectations.core import IDDict


def make_batch_identifier(identifer_dict: dict) -> IDDict:
    batch_id: IDDict
    if "dataframe" in identifer_dict:
        batch_id = IDDict(identifer_dict, dataframe=DATAFRAME_REPLACEMENT_STR)
    elif "dataframe_chunks" in identifer_dict:
        batch_id = IDDict(identifer_dict, dataframe_chunks=DATAFRAME_CHUNKS_REPLACEMENT_STR)
    else:
        batch_id = IDDict(identifer_dict)
    return batch_id
