from dataclasses import dataclass, field
from typing import List


@dataclass
class Item:
    length: int  # demand


@dataclass
class TwoDItem(Item):
    length: int
    width: int


@dataclass
class Compartment:
    depth: int  # capacity
    _stack: List[Item] = field(default_factory=list)

    @property
    def capacity(self) -> int:
        return self.depth

    @property
    def remaining_capacity(self) -> int:
        return self.depth - sum(item.length for item in self._stack)


@dataclass
class Vehicle:
    compartments: List[Compartment]
