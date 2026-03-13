from typing import Literal

from pydantic import BaseModel


class SqlOperation(BaseModel):
    # Operation order in the SQL plan
    src_idx: int
    # The source table to query
    table: str
    # The SQL query to execute on the source table to retrieve the relevant rows
    sql: str
    type: Literal['Document', 'Query']
