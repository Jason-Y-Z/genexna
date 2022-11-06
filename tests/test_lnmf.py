from genexna import lnmf
import numpy as np
import pytest


def compare_segments(segments_a, segments_b):
    for i in range(len(segments_a)):
        test_dist = np.linalg.norm(segments_a[i] - segments_b[i])
        assert pytest.approx(0) == test_dist


def test_ranks_unique_valued_list():
    TEST_LIST = [10, 5, 7, 9, 23, -23, 1.5, -9.2]
    RESULT_LIST = [6, 3, 4, 5, 7, 0, 2, 1]
    assert RESULT_LIST == lnmf.ranks(TEST_LIST)


def test_ranks_list_with_duplicates():
    TEST_LIST = [10, 5, 7, 9, 23, -23, 1.5, -9.2, 1.5, 7, 10]
    RESULT_LIST = [6, 3, 4, 5, 7, 0, 2, 1, 2, 4, 6]
    assert RESULT_LIST == lnmf.ranks(TEST_LIST)


def test_ranks_empty_list():
    TEST_LIST = []
    RESULT_LIST = []
    assert RESULT_LIST == lnmf.ranks(TEST_LIST)


def test_segment_single_class():
    TEST_X = np.random.rand(6, 10)
    TEST_Y = [0] * 6
    RESULT_X = [TEST_X]
    compare_segments(RESULT_X, lnmf.segment(TEST_X, TEST_Y))


def test_segment_two_classes():
    TEST_X = np.random.rand(6, 10)
    TEST_Y = [0, 1, 1, 0, 0, 1]
    RESULT_X = [TEST_X[[0, 3, 4], :], TEST_X[[1, 2, 5], :]]
    compare_segments(RESULT_X, lnmf.segment(TEST_X, TEST_Y))


def test_segment_five_classes():
    TEST_X = np.random.rand(10, 20)
    TEST_Y = [3, 4, 9, 20, 3, 9, 20, 2, 2, 2]
    RESULT_X = [TEST_X[[7, 8, 9], :],
                TEST_X[[0, 4], :],
                TEST_X[[1], :],
                TEST_X[[2, 5], :],
                TEST_X[[3, 6], :]]
    compare_segments(RESULT_X, lnmf.segment(TEST_X, TEST_Y))


def test_segment_empty_input():
    TEST_X = np.array([])
    TEST_Y = []
    assert [] == lnmf.segment(TEST_X, TEST_Y)


def test_segment_unmatched_input():
    TEST_X = np.random.rand(10, 20)
    TEST_Y = [3, 4, 9, 20, 3, 9, 20]
    with pytest.raises(RuntimeError):
        lnmf.segment(TEST_X, TEST_Y)


def test_unsegment_single_class():
    TARGET_X = np.random.rand(6, 10)
    TEST_X = TARGET_X
    TEST_Y = [0] * 6
    RESULT_X = TARGET_X
    compare_segments(RESULT_X, lnmf.unsegment(TEST_X, TEST_Y))


def test_unsegment_two_classes():
    TARGET_X = np.random.rand(6, 10)
    TEST_X = np.vstack([TARGET_X[[0, 3, 4], :], TARGET_X[[1, 2, 5], :]])
    TEST_Y = [0, 1, 1, 0, 0, 1]
    RESULT_X = TARGET_X
    compare_segments(RESULT_X, lnmf.unsegment(TEST_X, TEST_Y))


def test_unsegment_five_classes():
    TARGET_X = np.random.rand(10, 5)
    TEST_X = np.vstack([TARGET_X[[7, 8, 9], :],
                        TARGET_X[[0, 4], :],
                        TARGET_X[[1], :],
                        TARGET_X[[2, 5], :],
                        TARGET_X[[3, 6], :]])
    TEST_Y = [3, 4, 9, 20, 3, 9, 20, 2, 2, 2]
    RESULT_X = TARGET_X
    compare_segments(RESULT_X, lnmf.unsegment(TEST_X, TEST_Y))


def test_unsegment_empty_input():
    TEST_X = np.array([])
    TEST_Y = []
    compare_segments(TEST_X, lnmf.unsegment(TEST_X, TEST_Y))


def test_unsegment_unmatched_input():
    TEST_X = np.vstack([np.random.rand(5, 20), np.random.rand(5, 20)])
    TEST_Y = [3, 4, 9, 20, 3, 9, 20]
    with pytest.raises(RuntimeError):
        lnmf.unsegment(TEST_X, TEST_Y)


def test_non_negative_matrix():
    factor = lnmf.LNMF(n_components=10)
    TEST_W = np.abs(np.random.rand(20, 10))
    TEST_H = np.abs(np.random.rand(10, 40))
    TEST_V = TEST_W.dot(TEST_H)
    TEST_V_NORM = np.linalg.norm(TEST_V)
    TEST_Y = np.random.randint(0, 10, 20)
    calc_w = factor.fit_transform(TEST_V, TEST_Y)
    calc_h = factor.components_
    test_dist = np.linalg.norm(TEST_V - calc_w.dot(calc_h))
    assert test_dist / TEST_V_NORM < 5e-2


def test_negative_matrix():
    factor = lnmf.LNMF(n_components=10)
    TEST_V = np.abs(np.random.rand(15, 45))
    TEST_V[3, 5] = -1
    TEST_Y = np.random.randint(0, 10, 20)
    with pytest.raises(RuntimeError):
        factor.fit_transform(TEST_V, TEST_Y)
