from os.path import abspath, dirname
from pathlib import Path

import pytest

from table_reclamation.facade.table_reclamation import AccessPlanner


@pytest.fixture(scope="module")
def project_root() -> Path:
    return Path(dirname(dirname(abspath(__file__))))


@pytest.fixture(scope="module")
def tr(project_root: Path) -> AccessPlanner:
    split_path = project_root / "data" / "MATHE_random_100"
    lexicon_path = project_root / "table_reclamation" / "assets" / "lexicon.json"
    return AccessPlanner(tables_path=split_path, lexicon_path=lexicon_path)
