
from pathlib import Path

from pydantic import BaseModel


class StatDirectoryStructure(BaseModel):
    """
    The structure of the directory containing the statistics for the tables.
    This is used to ensure that the statistics are stored in a consistent way, and to make it easy to read the statistics when generating plans.
    """
    stats_file: Path
    mapping_file: Path
    sources_file: Path
