from os.path import abspath, dirname
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from table_reclamation.facade.table_reclamation import AccessPlanner


@pytest.fixture(scope="module")
def project_root() -> Path:
    return Path(dirname(dirname(abspath(__file__))))


@pytest.fixture(scope="module")
def planner_random(project_root: Path) -> AccessPlanner:
    split_path = project_root / "data" / "MATHE_random_100"
    lexicon_path = project_root / "table_reclamation" / "assets" / "lexicon.json"
    return AccessPlanner(tables_path=split_path, lexicon_path=lexicon_path)


@pytest.fixture(scope="module")
def planner_mathe(project_root: Path) -> AccessPlanner:
    split_path = project_root / "data" / "real_mathe"
    lexicon_path = project_root / "table_reclamation" / "assets" / "lexicon.json"
    return AccessPlanner(tables_path=split_path, lexicon_path=lexicon_path)


@pytest.fixture(scope="module")
def planner_mathe_split(project_root: Path) -> AccessPlanner:
    split_path = project_root / "data" / "mathe_splitted"
    lexicon_path = project_root / "table_reclamation" / "assets" / "lexicon.json"
    return AccessPlanner(tables_path=split_path, lexicon_path=lexicon_path)
