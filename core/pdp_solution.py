from copy import copy
from typing import Dict, List, Tuple

from . import Item


class Solution:
    def __init__(
        self,
        items: Dict[int, Item],
        order: List[int],
        stack_assignment: List[List[int]],
        is_partial: bool = False,
    ) -> None:
        # Items representing the customers
        # items[pickup_vertex] = some Item
        self.items = items
        # Order of visiting vertices, represented by vertices indices
        self.order = order

        # Assignment of items to stacks:
        # stack_assignment[stack_idx] = [assigned pickup indices]
        self.stack_assignment = stack_assignment
        # defined only for pickup vertices (undefined for depot)
        # item_assignment[pickup_vertex] = assigned stack idx
        self.item_assignment: Dict[int, int] = dict()
        for stack_idx, stack_assign in enumerate(stack_assignment):
            for vertex in stack_assign:
                self.item_assignment[vertex] = stack_idx

        self.is_partial = is_partial

        # A snapshot per vertex / step for all stacks
        # vertex_stacks_snapshots[vertex][stack_idx] = capacity after leaving vertex
        # step_stacks_snapshots[ith step / ith vertex][stack_idx] = capacity after step
        self.vertex_stacks_snapshots, self.step_stacks_snapshots = (
            self.init_stack_snapshots()
        )

    def init_stack_snapshots(self) -> Tuple[List[List[int]], List[List[int]]]:
        nb_stacks = len(self.stack_assignment)
        nb_vertices = len(self.items) * 2 + 1
        vertices_snapshots: List[List[int]] = [[0] * nb_stacks] * nb_vertices
        steps_snapshots: List[List[int]] = [[0] * nb_stacks] * nb_vertices

        def is_pickup_vertex(v: int) -> bool:
            return v <= len(self.items)

        def corresponding_pickup_vertex(v: int) -> int:
            if is_pickup_vertex(v):
                return v
            return v - len(self.items)

        # We always start with empty stacks at the depot
        state = [0 for _ in range(nb_stacks)]
        for step, vertex in enumerate(self.order):
            if step != 0:
                state = copy(state)
                pick_vertex = corresponding_pickup_vertex(vertex)
                stack_idx = self.item_assignment[pick_vertex]
                item = self.items[pick_vertex]
                demand = item.length if is_pickup_vertex(vertex) else -item.length
                # update state with demand picked / delivered at the vertex
                state[stack_idx] += demand

            vertices_snapshots[vertex] = state
            steps_snapshots[step] = state

        return vertices_snapshots, steps_snapshots

    def __repr__(self) -> str:
        sol_str = "Solution(\n"
        order_str = f"\torder={', '.join(str(v) for v in self.order)}, \n"

        stacks_str = ""
        for step, state in enumerate(self.step_stacks_snapshots):
            stacks_str += f"\tstep: {step} vertex: {self.order[step]} \t"
            stacks_str += ", ".join(str(s) for s in state) + "\n"

        sol_str += order_str + stacks_str + ")"
        return sol_str
