from eval.test_target_normalizer import normalize_target


def test_normalize_django_style_target():
    raw = "test_eq (test_exceptions.test_validation_error.TestValidationError)"
    out = normalize_target(raw)
    assert out == "test_exceptions/test_validation_error.py::TestValidationError::test_eq"
