from qcell.vision import generate_demo_pair, inspect_against_reference


def test_normal_pair_passes() -> None:
    reference, target = generate_demo_pair("normal")
    result = inspect_against_reference(reference, target)

    assert result.is_defect is False
    assert result.defect_ratio == 0


def test_scratch_pair_is_rejected() -> None:
    reference, target = generate_demo_pair("scratch")
    result = inspect_against_reference(reference, target)

    assert result.is_defect is True
    assert result.defect_ratio > 0.002
    assert result.overlay.size == target.size


def test_missing_part_pair_is_rejected() -> None:
    reference, target = generate_demo_pair("missing_part")
    result = inspect_against_reference(reference, target)

    assert result.is_defect is True


def test_invalid_threshold_is_rejected() -> None:
    reference, target = generate_demo_pair("normal")

    try:
        inspect_against_reference(reference, target, pixel_threshold=0)
    except ValueError:
        return
    raise AssertionError("invalid threshold should raise ValueError")
