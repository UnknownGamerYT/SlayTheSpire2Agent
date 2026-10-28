"""The .NET 9 List.Sort ordering used before the game's StableShuffle.

The stable game targets netcoreapp9.0. Reference: dotnet/runtime v9.0.0,
System.Private.CoreLib ArraySortHelper.cs and Array.IntrosortSizeThreshold.
Equal keys are deliberately not stabilized: duplicate card instances may have
identical ModelId/upgrade comparisons while retaining different enchantments.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

T = TypeVar('T')


def dotnet_sorted(items: Sequence[T], *, key: Callable[[T], Any] | None = None) -> list[T]:
    result = list(items)
    keys: list[Any] = [value if key is None else key(value) for value in result]

    def swap(left: int, right: int) -> None:
        result[left], result[right] = result[right], result[left]
        keys[left], keys[right] = keys[right], keys[left]

    def swap_greater(left: int, right: int) -> None:
        if keys[left] > keys[right]:
            swap(left, right)

    def down_heap(start: int, index: int, length: int) -> None:
        value, comparison = result[start + index - 1], keys[start + index - 1]
        while index <= length >> 1:
            child = 2 * index
            if child < length and keys[start + child - 1] < keys[start + child]:
                child += 1
            if not comparison < keys[start + child - 1]:
                break
            result[start + index - 1] = result[start + child - 1]
            keys[start + index - 1] = keys[start + child - 1]
            index = child
        result[start + index - 1], keys[start + index - 1] = value, comparison

    def intro(start: int, length: int, depth: int) -> None:
        while length > 1:
            if length <= 16:
                if length == 2:
                    swap_greater(start, start + 1)
                elif length == 3:
                    swap_greater(start, start + 1)
                    swap_greater(start, start + 2)
                    swap_greater(start + 1, start + 2)
                else:
                    for index in range(start, start + length - 1):
                        value, comparison = result[index + 1], keys[index + 1]
                        previous = index
                        while previous >= start and comparison < keys[previous]:
                            result[previous + 1] = result[previous]
                            keys[previous + 1] = keys[previous]
                            previous -= 1
                        result[previous + 1], keys[previous + 1] = value, comparison
                return
            if depth == 0:
                for index in range(length >> 1, 0, -1):
                    down_heap(start, index, length)
                for index in range(length, 1, -1):
                    swap(start, start + index - 1)
                    down_heap(start, 1, index - 1)
                return
            depth -= 1
            last = start + length - 1
            middle = start + ((length - 1) >> 1)
            swap_greater(start, middle)
            swap_greater(start, last)
            swap_greater(middle, last)
            pivot = keys[middle]
            swap(middle, last - 1)
            left, right = start, last - 1
            while left < right:
                left += 1
                while keys[left] < pivot:
                    left += 1
                right -= 1
                while pivot < keys[right]:
                    right -= 1
                if left >= right:
                    break
                swap(left, right)
            if left != last - 1:
                swap(left, last - 1)
            intro(left + 1, last - left, depth)
            length = left - start

    if len(result) > 1:
        intro(0, len(result), 2 * len(result).bit_length())
    return result
