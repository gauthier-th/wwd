from __future__ import annotations

import pytest

from server.quick import folds_of


@pytest.mark.parametrize("n", range(3, 13))
def test_folds_hold_out_every_recording_once_and_keep_two_to_fit(n):
    folds = folds_of(n)
    assert sorted(i for f in folds for i in f) == list(range(n))
    assert all(n - len(f) >= 2 for f in folds)


def test_folds_of_ten_are_five_pairs():
    assert folds_of(10) == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    assert folds_of(5) == [[0, 1], [2, 3], [4]]
    assert folds_of(3) == [[0], [1], [2]]
