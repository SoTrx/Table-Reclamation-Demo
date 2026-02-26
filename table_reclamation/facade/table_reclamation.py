"""
This is a facade that makes it easy to use the library without having to understand the internal workings of the library.
The core can be extended without affecting the facade, and the facade can be extended without affecting the core. 
"""
from json import load
from pathlib import Path
from typing import Any, Dict, List

from pandas import read_parquet

from table_reclamation.core import (
    build_sql_plan,
    gen_ap_order,
    parse_nl_to_ur,
)
from table_reclamation.domain.sql_operation import SqlOperation


class AccessPlanner:
    """
    From a given set of tables and a natural language query, generates a SQL plan to retrieve the relevant rows from the tables.
     The SQL plan is a list of SQL operations, each operation specifies the source table to query
    """
    # Tables histograms
    _index: Dict[str, Any]
    _lexicon: Dict[str, Any]

    def __init__(self, tables_path: Path, lexicon_path: Path = Path("assets/lexicon.json")) -> None:
        """

        """
        with open(lexicon_path) as f:
            self._lexicon = load(f)

        self._index = self.__read_reverse_index(tables_path)

    def __read_reverse_index(self, tables_path: Path) -> Dict[str, Any]:
        """
        Read the mapping of the distribution of values in the tables.
        Arguments:
            tables_path: The path to the directory containing the value index and stats parquet files.
        Returns:
            A dictionary containing the value index and source vectors.
        """
        vi_path = tables_path / "value_index.json"
        pq_path = tables_path / "stats.parquet"
        with open(vi_path) as f:
            value_index = load(f)

        source_vectors = read_parquet(pq_path).values
        return {
            "value_index": value_index,
            "source_vectors": source_vectors,
        }

    def generate_plan(self, query: str) -> List[SqlOperation]:
        """
        Generates a new SQL plan for a given natural language query.
        Arguments:
            query: The natural language query to generate a plan for.
        Returns:
            A list of SqlOperation objects representing the SQL plan.
        """
        ur = parse_nl_to_ur(query, self._lexicon)
        order = gen_ap_order(ur, self._index)
        return build_sql_plan(ur, order, self._index)
