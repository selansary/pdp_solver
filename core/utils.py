import itertools
from typing import Any, List


def permute(input: List[Any]) -> List[List[Any]]:
    tupled = list(itertools.permutations(input))
    listed = [list(tup) for tup in tupled]
    return listed


def slice_and_insert(input_list: List[int], sth: int, idx: int) -> List[int]:
    return input_list[:idx] + [sth] + input_list[idx:]
