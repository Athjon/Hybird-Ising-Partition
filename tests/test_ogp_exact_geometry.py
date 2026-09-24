"""Independent checks of exhaustive native-energy and overlap diagnostics."""

import itertools
import math

import numpy as np
import pytest

from benchmarks.hypergraph.ogp_exact_geometry import (
    balanced_masks,
    brute_force_pair_histogram,
    complete_pair_histogram,
    make_instance,
    native_costs,
    pair_witness,
    support_description,
)


@pytest.mark.parametrize('n', [8, 12, 16])
def test_strict_balance_and_flip_quotient_are_complete(n):
    masks = balanced_masks(n)
    assert len(masks) == math.comb(n, n // 2) // 2
    assert len(set(masks.tolist())) == len(masks)
    assert all(int(mask) & 1 == 0 for mask in masks)
    assert all(bin(int(mask)).count('1') == n // 2 for mask in masks)
    mask_set = set(masks.tolist())
    full_bits = (1 << n) - 1
    assert all(int(mask) ^ full_bits not in mask_set for mask in masks)


def test_native_energy_matches_explicit_connectivity_and_ignores_labels():
    n = 8
    masks = balanced_masks(n)
    edges = [[0, 1, 2, 3], [0, 1, 2, 3], [2, 4, 6, 7], [0, 7], [5]]
    expected = [sum(len({(int(mask) >> v) & 1 for v in edge}) - 1 for edge in edges)
                for mask in masks]
    np.testing.assert_array_equal(native_costs(masks, edges), expected)
    np.testing.assert_array_equal(native_costs(masks ^ ((1 << n) - 1), edges), expected)


@pytest.mark.parametrize('block_entries', [1, 17, 100000])
def test_every_pair_matches_independent_spin_dot_product(block_entries):
    n = 8
    masks = balanced_masks(n)
    hist = complete_pair_histogram(masks, n, max_block_entries=block_entries)
    np.testing.assert_array_equal(hist, brute_force_pair_histogram(masks, n))
    assert hist.sum() == math.comb(len(masks), 2)
    assert np.flatnonzero(hist).tolist() == [0, 4]


def test_singleton_and_missing_endpoint_are_not_interior_gaps():
    n = 12
    feasible_hist = complete_pair_histogram(balanced_masks(n), n)
    mask = balanced_masks(n)[:1]
    desc = support_description(complete_pair_histogram(mask, n), feasible_hist, n, 1)
    assert not desc['has_interior_gap']
    assert not desc['has_nontrivial_including_self_interior_gap']
    assert not desc['has_nontrivial_gap_with_self_upper_boundary']
    assert desc['distinct_pair_support_numerators'] == []
    assert desc['including_self_support_numerators'] == [n]
    endpoint_missing = np.zeros(n + 1, dtype=np.int64)
    endpoint_missing[8] = 1
    assert not support_description(endpoint_missing, feasible_hist, n, 2)['has_interior_gap']


def test_interior_gap_has_distinct_pair_witnesses_on_both_sides():
    n = 12
    masks = np.asarray([sum(1 << v for v in group) for group in
                        ([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 7],
                         [1, 2, 3, 8, 9, 10])], dtype=np.uint32)
    feasible = complete_pair_histogram(balanced_masks(n), n)
    hist = complete_pair_histogram(masks, n)
    desc = support_description(hist, feasible, n, len(masks))
    np.testing.assert_array_equal(hist, brute_force_pair_histogram(masks, n))
    assert desc['distinct_pair_support_numerators'] == [0, 8]
    assert desc['interior_gaps_bounded_by_distinct_pair_support'] == [{
        'lower_supported_numerator': 0, 'upper_supported_numerator': 8,
        'missing_feasible_numerators': [4],
    }]
    assert pair_witness(masks, n, 0) is not None
    assert pair_witness(masks, n, 8) is not None
    assert pair_witness(masks, n, 4) is None


def test_two_far_apart_classes_can_have_gap_bounded_above_by_self_pair():
    n = 12
    masks = np.asarray([sum(1 << v for v in group) for group in
                        ([1, 2, 3, 4, 5, 6], [1, 2, 3, 8, 9, 10])], dtype=np.uint32)
    feasible = complete_pair_histogram(balanced_masks(n), n)
    hist = complete_pair_histogram(masks, n)
    desc = support_description(hist, feasible, n, len(masks))
    assert desc['distinct_pair_support_numerators'] == [0]
    assert desc['including_self_support_numerators'] == [0, 12]
    assert not desc['has_distinct_bounded_interior_gap']
    assert desc['has_nontrivial_including_self_interior_gap']
    assert desc['has_nontrivial_gap_with_self_upper_boundary']
    assert desc['interior_gaps_including_self'] == [{
        'lower_supported_numerator': 0,
        'upper_supported_numerator': 12,
        'missing_feasible_numerators': [4, 8],
        'upper_boundary_is_self_overlap': True,
    }]
    np.testing.assert_array_equal(hist, brute_force_pair_histogram(masks, n))


def test_complete_feasible_set_has_no_gaps_in_either_definition():
    n = 12
    masks = balanced_masks(n)
    hist = complete_pair_histogram(masks, n)
    desc = support_description(hist, hist, n, len(masks))
    assert not desc['has_distinct_bounded_interior_gap']
    assert not desc['has_nontrivial_including_self_interior_gap']


def test_generated_instances_and_threshold_pairs_are_reproducible():
    n = 8
    masks = balanced_masks(n)
    for family in ('random', 'planted'):
        edges = make_instance(family, n, 941)
        assert edges == make_instance(family, n, 941)
        assert len(edges) == 2 * n
        assert all(len(set(edge)) == 4 for edge in edges)
        costs = native_costs(masks, edges)
        for epsilon in (0, 1, 2):
            near = masks[costs <= costs.min() + epsilon]
            np.testing.assert_array_equal(complete_pair_histogram(near, n),
                                          brute_force_pair_histogram(near, n))


def test_guard_against_unbounded_enumeration():
    with pytest.raises(ValueError, match='between 4 and 16'):
        balanced_masks(18)
