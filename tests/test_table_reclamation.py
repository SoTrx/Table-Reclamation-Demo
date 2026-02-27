from table_reclamation.facade.table_reclamation import AccessPlanner


def test_generate_mathe_plan(tr: AccessPlanner):
    tr.generate_stats()
    ap = tr.generate_plan("Discrete Mathematics level 2 with Equations")
    print(ap)
