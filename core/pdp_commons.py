from dataclasses import dataclass
from enum import Enum, auto
from typing import List


@dataclass
class Item:
    length: int  # demand


@dataclass
class TwoDimensionalItem(Item):
    width: int


@dataclass
class Compartment:
    depth: int  # capacity

    @property
    def capacity(self) -> int:
        return self.depth

    def is_item_compatible(self, item: Item) -> bool:
        return item.length <= self.depth

    def demand_for_item(self, item: Item) -> int:
        if not self.is_item_compatible(item):
            raise ValueError("Compartment cannot accomodate item demand.")
        return item.length


@dataclass
class TwoDimensionalCompartment(Compartment):
    length: int

    def is_item_compatible(self, item: Item) -> bool:
        if not isinstance(item, TwoDimensionalItem):
            raise ValueError("Incompatible 1D Item with 2D Compartment.")

        if item.length == self.length:
            return True
        if item.width == self.length:
            # item is compatible when rotated
            return True
        return False

    def demand_for_item(self, item: Item) -> int:
        if not self.is_item_compatible(item):
            raise ValueError("Compartment cannot accomodate item demand.")
        if item.length == self.length:
            return item.width
        # item is rotated
        return item.length


@dataclass
class Vehicle:
    compartments: List[Compartment]


class OptimizationObjective(Enum):
    CHEAPEST_ROUTE = auto()
    LEAST_ROTATION = auto()
    CHEAPEST_ROUTE_LEAST_ROTATION = auto()
