import pytest

from qcell.digital_twin import twin_frame, twin_timeline


def test_pass_product_stays_on_main_lane():
    frames = twin_timeline("PASS")
    assert all(frame.product_y == 50.0 for frame in frames)
    assert frames[-1].state == "COMPLETE"


def test_reject_product_reaches_bin_and_actuator_completes():
    frames = twin_timeline("REJECT")
    assert min(frame.gate_angle for frame in frames) == -55.0
    assert frames[-1].product_y == 85.0
    assert frames[-1].actuator_progress == 100.0
    assert frames[-1].state == "REJECT_BIN"


def test_twin_rejects_invalid_decision():
    with pytest.raises(ValueError):
        twin_frame("UNKNOWN", 0.5)
