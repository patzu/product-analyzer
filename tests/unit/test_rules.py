from dataclasses import replace
import pytest
from vision_sorter.domain.models import Detection, Status, Classification
from vision_sorter.classification.rules import RuleConfig, RuleClassifier, aggregate

BASE = Detection(100, 40, 0, 0, 10, 10, 5, 5)


@pytest.mark.parametrize("field,value,reason", [("area", 99, "area below"), ("area", 201, "area above"),
    ("width", 9, "width below"), ("width", 21, "width above"), ("height", 9, "height below"), ("height", 21, "height above")])
def test_dimension_failures(field, value, reason):
    rules = RuleConfig(100, 200, 10, 20, 10, 20)
    result = RuleClassifier(rules).classify(replace(BASE, **{field: value}))
    assert result.status == Status.DEFECTIVE
    assert reason in result.reason


def test_boundaries_unknown_and_color():
    assert RuleClassifier().classify(BASE).status == Status.GOOD
    classifier = RuleClassifier(RuleConfig(minimum_confidence=0.8))
    assert classifier.classify(BASE).status == Status.UNKNOWN
    assert classifier.classify(replace(BASE, confidence=0.7)).status == Status.UNKNOWN
    assert classifier.classify(replace(BASE, confidence=0.8)).status == Status.GOOD
    color = RuleClassifier(RuleConfig(hsv_lower=(0, 0, 0), hsv_upper=(10, 255, 255), minimum_color_fraction=0.8))
    assert color.classify(BASE).status == Status.UNKNOWN
    assert color.classify(replace(BASE, color_fraction=0.5)).status == Status.DEFECTIVE
    assert color.classify(replace(BASE, color_fraction=0.8)).status == Status.GOOD
    assert RuleClassifier().classify(replace(BASE, area=float("nan"))).status == Status.UNKNOWN


def test_aggregate_and_invalid_configuration():
    assert aggregate([]).status == Status.UNKNOWN
    assert aggregate([Classification(Status.GOOD, "ok"), Classification(Status.DEFECTIVE, "small")]).status == Status.DEFECTIVE
    with pytest.raises(ValueError):
        RuleConfig(minimum_area=200, maximum_area=100)
    with pytest.raises(ValueError):
        RuleConfig(minimum_color_fraction=0.8)
