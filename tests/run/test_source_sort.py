"""Independent C transcription vectors from .NET 9 ArraySortHelper introsort."""
from sts2sim.engine.rng import RunRng, capture_random_state, random_stream, stable_shuffle
from sts2sim.mechanics.source_sort import dotnet_sorted


def test_duplicate_comparisons_keep_insertion_order_only_through_sixteen_items():
    assert dotnet_sorted(list(range(16)), key=lambda _: 0) == list(range(16))
    assert dotnet_sorted(list(range(17)), key=lambda _: 0) == [
        0, 14, 13, 12, 11, 10, 9, 15, 8, 6, 5, 4, 3, 2, 1, 7, 16]
    assert dotnet_sorted(list(range(33)), key=lambda _: 0) == [
        0, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 31, 16,
        14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 15, 32]


def test_three_duplicate_groups_match_c_median_partition_order():
    assert dotnet_sorted(list(range(17)), key=lambda value: value % 3) == [
        0, 12, 3, 9, 6, 15, 13, 10, 16, 4, 1, 7, 5, 11, 2, 14, 8]
    assert dotnet_sorted(list(range(33)), key=lambda value: value % 3) == [
        0, 30, 3, 27, 6, 24, 9, 21, 12, 18, 15, 28, 25, 22, 19, 31, 16,
        13, 10, 7, 4, 1, 17, 11, 20, 8, 23, 5, 26, 2, 29, 14, 32]


def test_stable_shuffle_keeps_dotnet_equal_key_permutation_before_fisher_yates():
    bank = RunRng('TEST')
    selected = stable_shuffle(list(range(33)), random_stream(bank, 'niche'),
                              key=lambda value: value % 3)
    assert selected[:6] == [30, 27, 16, 12, 6, 14]
    assert capture_random_state(bank).streams['niche'].draws == 32


def test_small_and_reverse_unique_arrays_sort_without_extra_randomness():
    for count in (0, 1, 2, 3, 16, 17, 33, 100, 1024):
        values = list(range(count, 0, -1))
        assert dotnet_sorted(values) == list(range(1, count + 1))
        assert values == list(range(count, 0, -1))
