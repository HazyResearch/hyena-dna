import rust_dl


def test_sum_as_string():
    assert rust_dl.sum_as_string(1, 1) == "2"
