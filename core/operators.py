import multiprocessing as mp
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from re import I
from typing import Dict, List, Tuple

import numpy as np

from . import OptimizationObjective, Problem, Solution, permute, slice_and_insert


class DestroyOperator(ABC):
    """Abstract base class for a destroy operator."""

    def __init__(
        self, problem: Problem, solution: Solution, destruction_degree: int
    ) -> None:
        self.problem = problem
        self.solution = solution
        self.nb_requests_to_remove = destruction_degree

    @abstractmethod
    def destroy(self) -> Solution:
        "Main method of the DestroyOperator. Returns a parital solution."


class RandomDestroyOperator(DestroyOperator):
    def destroy(self) -> Solution:
        N = len(self.problem.items)
        pickup_vertices = self.problem.P
        requests_to_remove = np.random.choice(
            pickup_vertices, size=self.nb_requests_to_remove, replace=False
        )

        def removable_vertex(vertex: int) -> bool:
            pick_vertex = vertex if vertex <= N else vertex - N
            return pick_vertex in requests_to_remove

        partial_order = [v for v in self.solution.order if not removable_vertex(v)]
        partial_stack_assignment = [
            [v for v in assign if not removable_vertex(v)]
            for assign in self.solution.stack_assignment
        ]

        return Solution(
            self.problem.items,
            partial_order,
            partial_stack_assignment,
            self.problem.vehicle,
            is_partial=True,
        )


class HighestCostDestroyOperator(DestroyOperator):
    """Destroy a solution by removing the highest cost edges of the route."""

    def _destroy_highest_cost_edges(self) -> Solution:
        """
        Destroy based on highest cost edges, regardless of them being towards
        pickup or delivery vertices.
        """
        C = self.problem.C
        N = len(self.problem.items)
        order = self.solution.order
        stack_assignment = self.solution.stack_assignment

        is_pickup = lambda v: v <= N
        associated_vertex = lambda v: v + N if is_pickup(v) else v - N

        prev = 0
        edges_costs: List[Tuple[int, int]] = []
        for vertex in order[1:]:
            cost = C[prev][vertex]
            edges_costs.append((cost, vertex))
            prev = vertex

        # Sort by highest cost
        edges_costs = sorted(edges_costs, key=lambda t: t[0], reverse=True)

        # Remove R highest-cost edges with their associated edges
        # (pickup-delivery / delivery-pickup)
        R = self.nb_requests_to_remove
        vertices_to_remove: List[int] = []
        for _, vertex in edges_costs:
            if not associated_vertex(vertex) in vertices_to_remove:
                vertices_to_remove.append(vertex)
                if len(vertices_to_remove) == R:
                    break
        # Add associated vertices
        associated_vertices = [
            associated_vertex(vertex) for vertex in vertices_to_remove
        ]
        vertices_to_remove.extend(associated_vertices)
        # Remove vertices from total order of solution
        reduced_order = [vertex for vertex in order if vertex not in vertices_to_remove]
        # Adapt stack assignment to removed vertices
        reduced_stack_assignment = [
            [
                pick_vertex
                for pick_vertex in stack
                if not pick_vertex in vertices_to_remove
            ]
            for stack in stack_assignment
        ]

        return Solution(
            self.problem.items,
            reduced_order,
            reduced_stack_assignment,
            self.problem.vehicle,
            is_partial=True,
        )

    def _destroy_highest_cost_requests(self) -> Solution:
        """
        Destroy based on highest cost requests. Highest cost requests are those with
        highest sum of edge costs of their pickup and delivery vertices.
        """
        C = self.problem.C
        N = len(self.problem.items)
        order = self.solution.order
        stack_assignment = self.solution.stack_assignment

        is_pickup = lambda v: v <= N

        prev = 0
        # a mapping from the pickup vertex to total request cost (sum pick/del edges)
        request_costs: Dict[int, int] = defaultdict(int)
        for vertex in order[1:]:
            cost = C[prev][vertex]
            pickup_vertex = vertex if is_pickup(vertex) else vertex - N
            request_costs[pickup_vertex] += cost
            prev = vertex

        # Sort requests by highest cost
        request_costs = {
            req: cost
            for req, cost in sorted(
                request_costs.items(), key=lambda item: item[1], reverse=True
            )
        }

        # Remove R highest-cost requests
        R = self.nb_requests_to_remove
        vertices_to_remove: List[int] = []
        for pick_vertex, _ in request_costs.items():
            vertices_to_remove.append(pick_vertex)
            vertices_to_remove.append(pick_vertex + N)
            if len(vertices_to_remove) / 2 == R:
                break
        # Remove vertices from total order of solution
        reduced_order = [vertex for vertex in order if vertex not in vertices_to_remove]
        # Adapt stack assignment to removed vertices
        reduced_stack_assignment = [
            [
                pick_vertex
                for pick_vertex in stack
                if not pick_vertex in vertices_to_remove
            ]
            for stack in stack_assignment
        ]

        return Solution(
            self.problem.items,
            reduced_order,
            reduced_stack_assignment,
            self.problem.vehicle,
            is_partial=True,
        )

    def destroy(self) -> Solution:
        return self._destroy_highest_cost_requests()


class RepairOperator(ABC):
    """Abstract base class for a repair operator."""

    def __init__(
        self,
        problem: Problem,
        destroyed_solution: Solution,
        optimization_objective: OptimizationObjective = (
            OptimizationObjective.CHEAPEST_ROUTE
        ),
    ) -> None:
        self.problem = problem
        self.destroyed_solution = destroyed_solution
        self.stack_assignment = destroyed_solution.stack_assignment
        self.optimization_objective = optimization_objective

        P = self.problem.P
        self.removed_pickups = [v for v in P if v not in destroyed_solution.order]

        self.insertion_orders = permute(self.removed_pickups)

    @property
    def items(self):
        return self.problem.items

    @property
    def partial_order(self) -> List[int]:
        return [v for v in self.destroyed_solution.order]

    def pickup_vertex(self, v: int) -> int:
        N = self.problem.N
        return v if v <= N else v - N

    def evaluate_solution(self, solution: Solution) -> int:
        return self.problem.evaluate_solution(solution, self.optimization_objective)

    @abstractmethod
    def repair(self) -> Solution:
        """Main method of a RepairOperator. Returns a full, feasible solution."""


class LeastCostRepairOperator(RepairOperator, ABC):
    """Abstract Least-cost repair operator"""

    def single_order_repair(self, insertion_order: List[int]) -> Solution:
        """Performs least cost insertion for the `insertion_order`."""
        N, _, _, _, _, M = self.problem.problem_data()
        items = self.items
        compartments = self.problem.vehicle.compartments

        # stack assingment to be extended after every request inserted
        stack_assignment = deepcopy(self.stack_assignment)
        # order to be updated after every request inserted
        extended_partial_order = self.partial_order

        # Try to insert the pickup vertex anywhere on any
        # stack with capacity then try to insert the delivery vertex
        # in a feasible location (without violating LIFO constraints)
        for vertex in insertion_order:
            item = items[vertex]
            # best cost and order for the current vertex
            best_extended_cost = None
            best_extended_partial_order = None
            best_extended_stack_assignment = None

            # Generic support for 1D and 2D items
            possible_stack_indices = [
                i for i in M if compartments[i].is_item_compatible(item)
            ]
            # Try a stack assignment
            for stack_idx in possible_stack_indices:
                # Try a pos in the visiting order for the pickup vertex
                for pick_pos in range(1, len(extended_partial_order) + 1):
                    # extend the extended order with the pick
                    ep_order = slice_and_insert(
                        extended_partial_order, vertex, pick_pos
                    )

                    # Try a pos for the delivery vertex
                    del_pos = pick_pos + 1
                    while del_pos < len(ep_order) + 1:

                        feasible = False
                        if del_pos == pick_pos + 1:
                            # delivery right after pickup
                            feasible = True

                        else:
                            # current pos is after a vertex for another request
                            # if the vertex is related to a different stack, it's
                            # possible to deliver the request after this vertex
                            other_vertex = ep_order[del_pos - 1]
                            curr_stack_assignment = stack_assignment[stack_idx]
                            if (
                                self.pickup_vertex(other_vertex)
                                not in curr_stack_assignment
                            ):
                                feasible = True

                            # else: other request assigned to the same stack
                            elif self.pickup_vertex(other_vertex) == other_vertex:
                                # if the vertex before is a pickup vertex for
                                # another request, we cannot deliver the to-be-inserted
                                # request since its item is not on top of the stack!
                                #  --> jump till the delivery pos of the other request
                                del_pos = ep_order.index(other_vertex + N) + 1
                                continue
                            else:
                                # the other vertex is a delivery vertex
                                # since this after delivering the other request,
                                # it's only feasible if the other request has been
                                # picked after the to-be-inserted one
                                if ep_order.index(other_vertex) > pick_pos:
                                    feasible = True

                        if feasible:
                            # extend the extended pick order with the delivery
                            ed_order = slice_and_insert(ep_order, vertex + N, del_pos)

                            # sanity check for stack capacity
                            ep_stack_assign = deepcopy(stack_assignment)
                            ep_stack_assign[stack_idx].append(vertex)

                            related_vertices = ep_stack_assign[stack_idx]
                            corresponding_deliveres = [i + N for i in related_vertices]

                            stack = compartments[stack_idx]
                            collected_demand = 0
                            for v in ed_order[1:]:
                                # Generic support for 1D and 2D items
                                demand = stack.demand_for_item(item)
                                if v in related_vertices:
                                    collected_demand += demand
                                elif v in corresponding_deliveres:
                                    collected_demand -= demand

                                if (
                                    collected_demand < 0
                                    or collected_demand > stack.capacity
                                ):
                                    # not a valid delivery position
                                    feasible = False
                                    break

                        if feasible:
                            # feasible partial solution found!
                            # check detour and consider least cost detour
                            solution = Solution(
                                self.items,
                                ed_order,
                                ep_stack_assign,
                                self.problem.vehicle,
                                is_partial=True,
                            )
                            e_cost = self.evaluate_solution(solution)
                            if (
                                best_extended_cost is None
                                or best_extended_cost > e_cost
                            ):
                                best_extended_cost = e_cost
                                # least cost detour found, update extended order
                                # with new request
                                best_extended_partial_order = ed_order
                                best_extended_stack_assignment = ep_stack_assign

                        del_pos += 1

            assert best_extended_partial_order, "A solution must be repairable!"
            # Update the partial order and the stack assignment for the
            # newly-inserted request
            extended_partial_order = best_extended_partial_order
            stack_assignment = best_extended_stack_assignment

        return Solution(
            self.items, extended_partial_order, stack_assignment, self.problem.vehicle
        )

    def optimal_all_orders_repair(self) -> Solution:
        """Brute force for an optimally-repaired solution. Single process repair.

        Note: Too slow, a single iteration may exceed time limit for the search.
        For optimal repair, use a multiple-process repair instead.
        """
        removed_pickups = [
            v for v in self.problem.P if v not in self.destroyed_solution.order
        ]

        # Try all possible insertion orders for the removed vertices
        # and choose the order that results in the least cost solution
        insertion_orders = permute(removed_pickups)
        best_solution = None
        best_total_cost = None
        for insertion_order in insertion_orders:
            solution = self.single_order_repair(insertion_order)
            # Check if this insertion order contributes to the lowest cost total order
            total_cost = self.evaluate_solution(solution)
            if best_solution is None or best_total_cost > total_cost:
                best_total_cost = total_cost
                best_solution = solution

        return best_solution


class SingleOrderLeastCostRepairOperator(LeastCostRepairOperator):
    """
    Least Cost repair operator based on a single insertion order. In case an insertion
    order is specified upon initialization, the order is used. Otherwise, an
    arbitrary insertion order is selected.
    """

    def __init__(
        self,
        problem: Problem,
        destroyed_solution: Solution,
        insertion_order: List[int] = [],
        optimization_objective: OptimizationObjective = (
            OptimizationObjective.CHEAPEST_ROUTE
        ),
    ) -> None:
        super().__init__(problem, destroyed_solution, optimization_objective)
        self.insertion_order = insertion_order or random.choice(self.insertion_orders)

    def repair(self) -> Solution:
        return self.single_order_repair(self.insertion_order)


class ParallelOptimalLeastCostRepairOperator(LeastCostRepairOperator):
    """
    Optimal Least Cost repair operator based on multiple parallel insertion orders.
    Each possible insertion order is solved independently in a process and the best
    solution with the best insertion cost is selected.
    """

    def repair(self) -> Solution:
        q: mp.Queue = mp.Queue()
        best_solution = None
        best_objective = None

        processes = []  # operator is not reusable, no pool needed (for now at least)
        for order in self.insertion_orders:
            process = mp.Process(target=self.single_order_repair, args=(order, q))
            processes.append(process)
            process.start()

        for process in processes:
            sol = q.get()
            obj = self.evaluate_solution(sol)
            if not best_solution or obj < best_objective:
                best_solution = sol
                best_objective = obj
            process.join()

        return best_solution

    def single_order_repair(self, insertion_order, q):
        solution = super().single_order_repair(insertion_order)
        q.put(solution)


class GreedyLeastCostInsertRepairOperator(LeastCostRepairOperator):
    def _insert_one_request(
        self, pickup_vertex: int, partial_solution: Solution
    ) -> Solution:
        N = self.problem.N
        item = self.items[pickup_vertex]
        compartments = self.problem.vehicle.compartments
        possible_stack_indices = [
            i for i in self.problem.M if compartments[i].is_item_compatible(item)
        ]

        # stack assingment to be extended after insertion
        stack_assignment = partial_solution.stack_assignment
        # order to be updated after insertion
        extended_partial_order = partial_solution.order

        # Try to insert along the route in any compatible stack, and chose the
        # least cost insertion
        best_cost = None
        best_solution = None

        # Try a stack assignment
        for stack_idx in possible_stack_indices:
            # Try a pos in the visiting order for the pickup vertex
            for pick_pos in range(1, len(extended_partial_order) + 1):
                # extend the extended order with the pick
                ep_order = slice_and_insert(
                    extended_partial_order, pickup_vertex, pick_pos
                )

                # Try a pos for the delivery vertex
                del_pos = pick_pos + 1
                while del_pos < len(ep_order) + 1:

                    feasible = False
                    if del_pos == pick_pos + 1:
                        # delivery right after pickup
                        feasible = True

                    else:
                        # current pos is after a vertex for another request
                        # if the vertex is related to a different stack, it's
                        # possible to deliver the request after this vertex
                        other_vertex = ep_order[del_pos - 1]
                        curr_stack_assignment = stack_assignment[stack_idx]
                        if (
                            self.pickup_vertex(other_vertex)
                            not in curr_stack_assignment
                        ):
                            feasible = True

                        # else: other request assigned to the same stack
                        elif self.pickup_vertex(other_vertex) == other_vertex:
                            # if the vertex before is a pickup vertex for
                            # another request, we cannot deliver the to-be-inserted
                            # request since its item is not on top of the stack!
                            #  --> jump till the delivery pos of the other request
                            del_pos = ep_order.index(other_vertex + N) + 1
                            continue
                        else:
                            # the other vertex is a delivery vertex
                            # since this after delivering the other request,
                            # it's only feasible if the other request has been
                            # picked after the to-be-inserted one
                            if ep_order.index(other_vertex) > pick_pos:
                                feasible = True

                    if feasible:
                        # extend the extended pick order with the delivery
                        ed_order = slice_and_insert(ep_order, pickup_vertex + N, del_pos)

                        # sanity check for stack capacity
                        ep_stack_assign = deepcopy(stack_assignment)
                        ep_stack_assign[stack_idx].append(pickup_vertex)

                        related_vertices = ep_stack_assign[stack_idx]
                        corresponding_deliveres = [i + N for i in related_vertices]

                        stack = compartments[stack_idx]
                        collected_demand = 0
                        for v in ed_order[1:]:
                            # Generic support for 1D and 2D items
                            demand = stack.demand_for_item(item)
                            if v in related_vertices:
                                collected_demand += demand
                            elif v in corresponding_deliveres:
                                collected_demand -= demand

                            if (
                                collected_demand < 0
                                or collected_demand > stack.capacity
                            ):
                                # not a valid delivery position
                                feasible = False
                                break

                    if feasible:
                        # feasible partial solution found!
                        # check detour and consider least cost detour
                        solution = Solution(
                            self.items,
                            ed_order,
                            ep_stack_assign,
                            self.problem.vehicle,
                            is_partial=True,
                        )
                        e_cost = self.evaluate_solution(solution)
                        if (
                            best_cost is None
                            or best_cost > e_cost
                        ):
                            best_cost = e_cost
                            best_solution = solution

                    del_pos += 1

        assert best_solution
        return best_solution

    def repair(self) -> Solution:
        partial_solution = self.destroyed_solution
        remaining_requests = set(self.removed_pickups)

        while remaining_requests:
            least_cost = None
            chosen_request = None
            least_insertion_solution = None

            for pick_vertex in remaining_requests:
                # insert this request in the least cost position
                solution = self._insert_one_request(pick_vertex, partial_solution)
                cost = self.evaluate_solution(solution)

                if not least_cost or cost < least_cost:
                    least_cost = cost
                    chosen_request = pick_vertex
                    least_insertion_solution = solution

            # remove the chosen request
            assert chosen_request
            remaining_requests.remove(chosen_request)
            partial_solution = least_insertion_solution

        return partial_solution
