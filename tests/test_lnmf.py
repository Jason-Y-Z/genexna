import numpy as np
import pytest
from genexna import lnmf


def compare_segments(segments_a, segments_b):
    for _, (seg_a, seg_b) in enumerate(zip(segments_a, segments_b)):
        test_dist = np.linalg.norm(seg_a - seg_b)
        assert pytest.approx(0) == test_dist


def test_ranks_unique_valued_list():
    test_list = [10, 5, 7, 9, 23, -23, 1.5, -9.2]
    result_list = [6, 3, 4, 5, 7, 0, 2, 1]
    assert result_list == lnmf.ranks(test_list)


def test_ranks_list_with_duplicates():
    test_list = [10, 5, 7, 9, 23, -23, 1.5, -9.2, 1.5, 7, 10]
    result_list = [6, 3, 4, 5, 7, 0, 2, 1, 2, 4, 6]
    assert result_list == lnmf.ranks(test_list)


def test_ranks_empty_list():
    test_list = []
    result_list = []
    assert result_list == lnmf.ranks(test_list)


def test_segment_single_class():
    test_x = np.random.rand(6, 10)
    test_y = [0] * 6
    result_x = [test_x]
    compare_segments(result_x, lnmf.segment(test_x, test_y))


def test_segment_two_classes():
    test_x = np.random.rand(6, 10)
    test_y = [0, 1, 1, 0, 0, 1]
    result_x = [test_x[[0, 3, 4], :], test_x[[1, 2, 5], :]]
    compare_segments(result_x, lnmf.segment(test_x, test_y))


def test_segment_five_classes():
    test_x = np.random.rand(10, 20)
    test_y = [3, 4, 9, 20, 3, 9, 20, 2, 2, 2]
    result_x = [test_x[[7, 8, 9], :],
                test_x[[0, 4], :],
                test_x[[1], :],
                test_x[[2, 5], :],
                test_x[[3, 6], :]]
    compare_segments(result_x, lnmf.segment(test_x, test_y))


def test_segment_empty_input():
    test_x = np.array([])
    test_y = []
    assert [] == lnmf.segment(test_x, test_y)


def test_segment_unmatched_input():
    test_x = np.random.rand(10, 20)
    test_y = [3, 4, 9, 20, 3, 9, 20]
    with pytest.raises(RuntimeError):
        lnmf.segment(test_x, test_y)


def test_unsegment_single_class():
    target_x = np.random.rand(6, 10)
    test_x = target_x
    test_y = [0] * 6
    result_x = target_x
    compare_segments(result_x, lnmf.unsegment(test_x, test_y))


def test_unsegment_two_classes():
    target_x = np.random.rand(6, 10)
    test_x = np.vstack([target_x[[0, 3, 4], :], target_x[[1, 2, 5], :]])
    test_y = [0, 1, 1, 0, 0, 1]
    result_x = target_x
    compare_segments(result_x, lnmf.unsegment(test_x, test_y))


def test_unsegment_five_classes():
    target_x = np.random.rand(10, 5)
    test_x = np.vstack([target_x[[7, 8, 9], :],
                        target_x[[0, 4], :],
                        target_x[[1], :],
                        target_x[[2, 5], :],
                        target_x[[3, 6], :]])
    test_y = [3, 4, 9, 20, 3, 9, 20, 2, 2, 2]
    result_x = target_x
    compare_segments(result_x, lnmf.unsegment(test_x, test_y))


def test_unsegment_empty_input():
    test_x = np.array([])
    test_y = []
    compare_segments(test_x, lnmf.unsegment(test_x, test_y))


def test_unsegment_unmatched_input():
    test_x = np.vstack([np.random.rand(5, 20), np.random.rand(5, 20)])
    test_y = [3, 4, 9, 20, 3, 9, 20]
    with pytest.raises(RuntimeError):
        lnmf.unsegment(test_x, test_y)


def test_non_negative_matrix():
    test_w = np.abs(np.random.rand(20, 10))
    test_h = np.abs(np.random.rand(10, 40))
    test_v = test_w.dot(test_h)
    test_v_norm = np.linalg.norm(test_v)
    test_y = np.random.randint(0, 10, 20)
    calc_w, calc_h = lnmf.factorize(test_v, test_y, n_components=10, return_w=True)
    test_dist = np.linalg.norm(test_v - calc_w.dot(calc_h))
    assert test_dist / test_v_norm < 5e-2


def test_negative_matrix():
    test_v = np.abs(np.random.rand(15, 45))
    test_v[3, 5] = -1
    test_y = np.random.randint(0, 10, 20)
    with pytest.raises(RuntimeError):
        lnmf.factorize(test_v, test_y, n_components=10)
