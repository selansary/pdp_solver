from dataclasses import dataclass
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

    @property
    def capacity(self) -> int:
        return self.depth


@dataclass
class TwoDCompartment(Compartment):
    length: int


@dataclass
class Vehicle:
    compartments: List[Compartment]
