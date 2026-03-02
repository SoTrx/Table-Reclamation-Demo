"""
This is a facade that makes it easy to use the library without having to understand the internal workings of the library.
The core can be extended without affecting the facade, and the facade can be extended without affecting the core. 
"""
import logging
from json import load
from pathlib import Path
from typing import Any, Dict, List

from pandas import read_parquet

from table_reclamation.core import (
    build_sql_plan,
    gen_ap_order,
    parse_nl_to_ur,
)
from table_reclamation.core.generate_stats import generate_stats_from_folder
from table_reclamation.domain.sql_operation import SqlOperation
from table_reclamation.domain.stats_dir_structure import StatDirectoryStructure

_DEFAULT_LEXICON = Path(__file__).parent.parent / "assets" / "lexicon.json"

logger = logging.getLogger(__file__)


class AccessPlanner:
    """
    From a given set of tables and a natural language query, generates a SQL plan to retrieve the relevant rows from the tables.
     The SQL plan is a list of SQL operations, each operation specifies the source table to query
    """
    # Tables histograms
    _index: Dict[str, Any]
    _lexicon: Dict[str, Any]
    _tables_path: Path

    def __init__(self, tables_path: Path, lexicon_path: Path = _DEFAULT_LEXICON) -> None:
        """

        """
        self._tables_path = tables_path

        with open(lexicon_path) as f:
            self._lexicon = load(f)

    def __read_stats(self, stats_path: Path) -> Dict[str, Any]:
        """
        Read the mapping of the distribution of values in the tables.
        Arguments:
            tables_path: The path to the directory containing the value index and stats parquet files.
        Returns:
            A dictionary containing the value index, source vectors, and source file names.
        """
        vi_path = stats_path / "value_index.json"
        pq_path = stats_path / "stats.parquet"
        sf_path = stats_path / "source_files.json"

        with open(vi_path) as f:
            value_index = load(f)

        source_vectors = read_parquet(pq_path).values

        source_files = None
        if sf_path.exists():
            with open(sf_path) as f:
                source_files = load(f)

        return {
            "value_index": value_index,
            "source_vectors": source_vectors,
            "source_files": source_files,
        }

    def _patch_ur(self, ur: Dict[str, Any]) -> Dict[str, Any]:
        """
        Patches the UR to ensure that it is in the correct format for generating a plan. This includes:
        - Resolving topic_name to topic ID using platform__topic.csv
        - Resolving subtopic_name to subtopic ID using platform__subtopic.csv
        - Normalising newLevel to newlevel

        """
        from pandas import read_csv

        resolved: Dict[str, Any] = {}

        # topic_name → topic (ID lookup via platform__topic.csv)
        if "topic_name" in ur:
            topic_file = self._tables_path / "platform__topic.csv"
            if topic_file.exists():
                topics = read_csv(topic_file)
                name_to_id = dict(zip(topics["name"], topics["id"]))
                ids = [name_to_id[n]
                       for n in ur["topic_name"] if n in name_to_id]
                if ids:
                    resolved["topic"] = ids
            else:
                resolved["topic_name"] = ur["topic_name"]

        # subtopic_name → subtopic (ID mapping lookup via platform__subtopic.csv)
        if "subtopic_name" in ur:
            subtopic_file = self._tables_path / "platform__subtopic.csv"
            if subtopic_file.exists():
                subtopics = read_csv(subtopic_file)
                name_to_id = dict(
                    zip(subtopics["name"], subtopics["id"].astype(float)))
                ids = [name_to_id[n]
                       for n in ur["subtopic_name"] if n in name_to_id]
                if ids:
                    resolved["subtopic"] = ids
            else:
                resolved["subtopic_name"] = ur["subtopic_name"]

        # newLevel → newlevel
        if "newLevel" in ur:
            resolved["newlevel"] = ur["newLevel"]

        # Pass through any other keys unchanged
        for key, val in ur.items():
            if key not in ("topic_name", "subtopic_name", "newLevel"):
                resolved[key] = val

        return resolved

    def generate_plan(self, query: str) -> List[SqlOperation]:
        """
        Generates a new SQL plan for a given natural language query.
        Arguments:
            query: The natural language query to generate a plan for.
        Returns:
            A list of SqlOperation objects representing the SQL plan.
        """
        try:
            self._index = self.__read_stats(self._tables_path)
        except FileNotFoundError:
            raise ValueError("Statistics not generated for tables path.")

        ur = parse_nl_to_ur(query, self._lexicon)

        # TODO: Remove this patching step by step as the UR generation is improved to produce the correct format directly.
        ur = self._patch_ur(ur)
        order = gen_ap_order(ur, self._index)
        raw_plan = build_sql_plan(ur, order, self._index)
        plan: List[SqlOperation] = []
        for step in raw_plan:
            op = SqlOperation(**step)
            plan.append(op)

        return plan

    def generate_stats(self) -> None:
        """
        Generates the statistics for the tables and saves them to the tables directory. 
        This should be run before generating any plans, and should be re-run whenever the tables are updated.
        """
        value_index, vectors = generate_stats_from_folder(self._tables_path)
        if value_index is None and vectors is None:
            logger.info("Statistics already generated")

        return StatDirectoryStructure(
            stats_file=self._tables_path / "stats.parquet",
            mapping_file=self._tables_path / "value_index.json",
            sources_file=self._tables_path / "source_files.json",
        )
