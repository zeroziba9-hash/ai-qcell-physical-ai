from qcell.simulation import LineSimulator


def test_zero_defect_rate_always_passes() -> None:
    simulator = LineSimulator(defect_rate=0, seed=1)
    events = [simulator.inspect_next() for _ in range(20)]

    assert all(event.result == "PASS" for event in events)
    assert all(event.action == "PASS_THROUGH" for event in events)


def test_full_defect_rate_always_rejects() -> None:
    simulator = LineSimulator(defect_rate=1, seed=1)
    events = [simulator.inspect_next() for _ in range(20)]

    assert all(event.result == "DEFECT" for event in events)
    assert all(event.action == "REJECT" for event in events)


def test_product_ids_are_sequential() -> None:
    simulator = LineSimulator(seed=1)

    assert simulator.inspect_next().product_id == "Q-00001"
    assert simulator.inspect_next().product_id == "Q-00002"

