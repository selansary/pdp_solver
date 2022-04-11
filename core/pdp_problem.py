import random
from typing import Any, Dict, List, Tuple

import gurobipy as gp

from . import (
    Compartment,
    Item,
    OptimizationObjective,
    Solution,
    TwoDimensionalItem,
    Vehicle,
)


class Problem:
    """
    The base Single Vehicle Pickup and Delivery Problem with Multiple Compartments.
    The generic base problem, independent of the Model and its constraints.
    """

    def __init__(
        self,
        items: List[Item],
        vehicle: Vehicle,
        distance_matrix: List[List[int]] = [],
        name: str = "Base-PDP",
    ) -> None:
        # A mapping from the pickup vertex to the Item
        self.items = {i + 1: item for i, item in enumerate(items)}
        # Number of Items
        self.N = len(items)
        # All vertices in the PDP
        self.V = [i for i in range(self.N * 2 + 1)]
        # Pickup vertices
        self.P = self.V[1 : self.N + 1]
        # Delivery vertices
        self.D = self.V[self.N + 1 :]
        # Vertices with Pickup and Delivery: vertices without the depot
        self.PUD = self.P + self.D
        # Vehicle with, possibly inhomogeneous, compartments having capacities
        self.vehicle = vehicle
        # Compartment indices
        self.M = list(range(len(self.vehicle.compartments)))

        # Time-based Cost Matrix. Defaults to a randomly initialized matrix.
        # Distance from / to depot is 0
        random.seed(0)
        self.C = distance_matrix or [
            [
                random.randint(1, 100) if i != 0 and j != 0 and i != j else 0
                for i in self.V
            ]
            for j in self.V
        ]

        self.name = name

    def problem_data(self):
        """Return problem data: N, V, P, D, C, M."""
        return self.N, self.V, self.P, self.D, self.C, self.M

    def evaluate_solution(
        self,
        solution: Solution,
        optimization_objective: OptimizationObjective = (
            OptimizationObjective.CHEAPEST_ROUTE
        ),
    ) -> int:
        """
        Return the objective value of the solution. The default objective minimizes
        the route cost.

        Applicable objectives for the base problem:
            - `OptimizationObjective.CHEAPEST_ROUTE`
        """
        if optimization_objective != OptimizationObjective.CHEAPEST_ROUTE:
            raise ValueError("This objective is not applicable to this problem.")

        cost = 0
        prev = 0  # we always start at the depot
        order = solution.order
        for vertex in order[1:]:
            cost += self.C[prev][vertex]
            prev = vertex
        return cost


class TwoDimensionalProblem(Problem):
    """
    The base Single Vehicle Pickup and Delivery Problem with Multiple 2D Compartments.
    The generic extended 2D base problem, independent of the Model and its constraints.
    """

    def __init__(
        self,
        items: List[TwoDimensionalItem],
        vehicle: Vehicle,
        distance_matrix: List[List[int]] = [],
        name: str = "Base-2D-PDP",
    ) -> None:
        super().__init__(items, vehicle, distance_matrix, name)

    def evaluate_solution(
        self,
        solution: Solution,
        optimization_objective: OptimizationObjective = (
            OptimizationObjective.CHEAPEST_ROUTE
        ),
    ) -> int:
        """
        Return the objective value of the solution. The default objective minimizes
        the route cost.

        Applicable objectives for the problem:
            - `OptimizationObjective.CHEAPEST_ROUTE`
            - `OptimizationObjective.LEAST_ROTATION`
            - `OptimizationObjective.CHEAPEST_ROUTE_LEAST_ROTATION`
        """

        def is_item_rotated(item_vertex: int, stack_idx: int) -> bool:
            item = self.items[item_vertex]
            stack = self.vehicle.compartments[stack_idx]
            return item.width == stack.length

        def evaluate_rotation_cost() -> int:
            item_assignemt = solution.item_assignment
            rotated = [is_item_rotated(i, item_assignemt[i]) for i in self.P]
            return sum(rotated)

        objective_val = 0
        if optimization_objective == OptimizationObjective.CHEAPEST_ROUTE:
            objective_val = super().evaluate_solution(solution)
        elif optimization_objective == OptimizationObjective.LEAST_ROTATION:
            objective_val = evaluate_rotation_cost()
        elif (
            optimization_objective
            == OptimizationObjective.CHEAPEST_ROUTE_LEAST_ROTATION
        ):
            objective_val = (
                super().evaluate_solution(solution) + evaluate_rotation_cost()
            )
        else:
            raise ValueError("This objective is not applicable to this problem.")

        return objective_val


class ModelledBaseProblem(Problem):
    """
    The modelled base Single Vehicle Pickup and Delivery Problem with Multiple
    Compartments.

    Extends the base `Problem` with the Gurobi model and its base constraints.

    The model for this base problem doesn't include loading constraints;
    for PDPs with stacking constraints, use `ModelledProblem` for 1D items or
    `ModelledTwoDimensionalProblem` for
    2D items.
    """

    def __init__(
        self,
        items: List[Item],
        vehicle: Vehicle,
        distance_matrix: List[List[int]] = [],
        name: str = "Base-PDP-Model",
    ) -> None:
        super().__init__(items, vehicle, distance_matrix, name)
        # Extend with model and decision variables
        self.model, self.data = self.create_model()

    def create_model(self) -> Tuple[gp.Model, Dict[str, Dict[Tuple[Any, ...], gp.Var]]]:
        """Initialize a gurobi model for the problem."""
        model = gp.Model(self.name)
        V = self.V
        M = self.M
        # edges between the vertices
        x = model.addVars(V, V, vtype=gp.GRB.BINARY, name="x")
        # assignment of an item / pickup vertex to a compartment
        y = model.addVars(V, M, vtype=gp.GRB.BINARY, name="y")
        # total ordering of the vertices
        u = model.addVars(V, vtype=gp.GRB.INTEGER, name="u")
        # compartment states after visiting vertices
        s = model.addVars(V, M, vtype=gp.GRB.INTEGER, name="s")

        # Set the bounds of the integer vars
        for i in u:
            u[i].setAttr(gp.GRB.Attr.LB, 0)
            u[i].setAttr(gp.GRB.Attr.UB, len(V) - 1)

        for i, k in s:
            s[i, k].setAttr(gp.GRB.Attr.LB, 0)
            s[i, k].setAttr(gp.GRB.Attr.UB, self.vehicle.compartments[k].capacity)

        # Apply the bounds to the var domains
        model.update()

        data = dict(x=x, y=y, u=u, s=s)
        return model, data

    def model_vars(self):
        """Return model data: x, y, u, s."""
        return self.data["x"], self.data["y"], self.data["u"], self.data["s"]

    def _apply_base_constraints(self):
        """
        Apply the base constraints to the gurobi model.

        The base constraints include the constraints of the base problem without
        the capacity constraints of the compartments.
        """
        N, V, P, D, _, M = self.problem_data()
        x, y, u, s = self.model_vars()

        # 2. Every vertex is left once.
        self.model.addConstrs(
            (sum(x[i, j] for j in V) == 1 for i in V), name="out_vertex_constr"
        )
        # 3. Every vertex is entered once.
        self.model.addConstrs(
            (sum(x[j, i] for j in V) == 1 for i in V), name="in_vertex_constr"
        )
        # 4. Demand of every item must be fulfilled.
        self.model.addConstrs(
            ((sum(y[i, k] for k in M) == 1) for i in P), name="fulfill_demands_constr"
        )
        # 5. The route is constituted by the order of visiting vertices.
        self.model.addConstrs(
            (u[j] >= u[i] + 1 - (2 * N * (1 - x[i, j])) for i in V for j in P + D),
            name="visiting_order_constr",
        )
        # 6. Every item must be picked before it's delivered.
        self.model.addConstrs(
            (u[N + i] >= u[i] + 1 for i in P), name="pickup_before_delivery_constr"
        )

        # 10. We always start at the depot
        self.model.addConstr(u[0] == 0, name="start_at_depot_constr")
        # 11. Vertices are order along the route.
        # constraint implicit in the domain of u
        # 12. We always begin with empty stacks at the depot.
        self.model.addConstrs(
            (s[0, k] == 0 for k in M), name="empty_stacks_at_depot_constr"
        )
        # 13. The stack capacity is always maintained for all stacks.
        # constraint implicit in the domain of s

    def _apply_compartment_constraints(self):
        """Apply compartment constraints for the base problem.

        Assumptions:
            - Compartments are not necessarily stacks for the base problem, so only the
            capacity constraints apply.
            - Base problem handles 1D items.

        Overriden, in extended problem definitions: `Problem` and
        `TwoDimensionalProblem` to include loading constraints and consider orientation
        for the latter problem.
        """
        N, V, P, _, _, M = self.problem_data()
        x, y, _, s = self.model_vars()

        # 7 Compartment capacity after pickup
        self.model.addConstrs(
            (
                s[j, k]
                >= s[i, k]
                + self.items[j].length * y[j, k]
                - self.vehicle.compartments[k].capacity * (1 - x[i, j])
                for i in V
                for j in P
                for k in M
            ),
            name="comp_after_pickup_constr",
        )
        # 8 Compartment capacity after delivery
        # items[N + j] with -items[j]
        self.model.addConstrs(
            (
                s[N + j, k]
                >= s[i, k]
                - self.items[j].length * y[j, k]
                - self.vehicle.compartments[k].capacity * (1 - x[i, N + j])
                for i in V
                for j in P
                for k in M
            ),
            name="comp_after_delivery_constr",
        )

    def apply_constraints(self):
        """Apply the problem constraints to the gurobi model."""
        self._apply_base_constraints()
        self._apply_compartment_constraints()

    def set_model_objective(
        self,
        optimization_objective: OptimizationObjective = (
            OptimizationObjective.CHEAPEST_ROUTE
        ),
    ):
        """
        Set the objective for the gurobi model. The default objective minimizes
        the route cost.

        Applicable objectives for the base problem:
            - `OptimizationObjective.CHEAPEST_ROUTE`
        """
        if optimization_objective != OptimizationObjective.CHEAPEST_ROUTE:
            raise ValueError("This objective is not applicable to this problem.")

        _, V, _, _, C, _ = self.problem_data()
        x = self.data["x"]
        self.model.setObjective(
            (sum(C[i][j] * x[i, j] for i in V for j in V)), gp.GRB.MINIMIZE
        )

    def solve(self, time_limit: float = 3.0, write_model: bool = True):
        """Run the gurobi solver on the problem model with a time limit."""
        if not self.model:
            return

        if write_model:
            self.model.write("./lp_models/" + self.name + ".lp")

        self.model.Params.TimeLimit = time_limit
        self.model.optimize()

    def extract_solution(self) -> Solution:
        """Extract a solution from the gurobi-solved model."""
        _, V, P, _, _, M = self.problem_data()
        _, y, u, _ = self.model_vars()

        u_list = [int(u[i].x) for i in V]
        order = [u_list.index(i) for i in range(len(V))]
        stack_assignment = [[i for i in P if abs(y[i, k].x) > 1e-6] for k in M]

        return Solution(self.items, order, stack_assignment, self.vehicle)


class ModelledOneDimensionalProblem(ModelledBaseProblem):
    """
    The modelled Single Vehicle Pickup and Delivery Problem with Multiple Stacked
    Compartments.

    The problem extends the `ModelledBaseProblem` with loading consraints and
    supports only 1D Items.
    """

    def __init__(
        self,
        items: List[Item],
        vehicle: Vehicle,
        distance_matrix: List[List[int]] = [],
        name: str = "PDPMS-Model",
    ) -> None:
        super().__init__(items, vehicle, distance_matrix, name)

    def _apply_compartment_constraints(self):
        """
        Apply the capacity and the LIFO loading constraints for the vehicle
        compartments.
        """
        # Apply the capacity constraints from the base problem.
        super()._apply_compartment_constraints()

        # Add the LIFO constraint
        N, _, P, _, _, M = self.problem_data()
        _, y, _, s = self.model_vars()
        self.model.addConstrs(
            (
                s[N + j, k]
                >= s[j, k]
                - self.items[j].length * y[j, k]
                - self.vehicle.compartments[k].capacity * (1 - y[j, k])
                for j in P
                for k in M
            ),
            name="stack_LIFO_constr",
        )


class ModelledTwoDimensionalProblem(ModelledBaseProblem, TwoDimensionalProblem):
    """
    The modelled Single Vehicle 2D Pickup and Delivery Problem with Multiple Stacked
    Compartments.

    The problem extends the `ModelledBaseProblem` with loading consraints and
    supports 2D items.
    """

    def __init__(
        self,
        items: List[TwoDimensionalItem],
        vehicle: Vehicle,
        distance_matrix: List[List[int]] = [],
        name: str = "2DPDPMS-Model",
    ) -> None:
        super().__init__(items, vehicle, distance_matrix, name)
        # Extend with compartment capabilities to suport 2D items and compartments
        self.compartment_capabilities = self.init_compatibility_matrix()

    def init_compatibility_matrix(self) -> List[Dict[int, int]]:
        def item_fits_in_compartment(item: Item, compartment: Compartment) -> int:
            if item is None:
                return 0
            if item.length == compartment.length and item.width <= compartment.depth:
                return 1
            if item.width == compartment.length and item.length <= compartment.depth:
                return 1
            return 0

        return [
            {p: item_fits_in_compartment(item, comp) for p, item in self.items.items()}
            for comp in self.vehicle.compartments
        ]

    def create_model(self) -> Tuple[gp.Model, Dict[str, Dict[Tuple[Any, ...], gp.Var]]]:
        model, data = super().create_model()

        _, _, P, _, _, M = self.problem_data()
        # Extend the model with rotation decision variables
        r = model.addVars(P, vtype=gp.GRB.BINARY, name="r")
        # combination of rotation and assingment decision variables
        # z[i, k] = 1 then item i is assigned to compartment k and is rotated
        z = model.addVars(P, M, vtype=gp.GRB.BINARY, name="z")

        data["r"] = r
        data["z"] = z
        return model, data

    def _apply_compartment_constraints(self):
        """
        Apply the capacity and the LIFO loading constraints for the 2D problem.
        Overrides the base problem compartment constraints with support for 2D items
        and loading constraints.
        """
        N, V, P, _, _, M = self.problem_data()
        x, y, _, s = self.model_vars()
        z = self.data["z"]

        I = self.items
        Capacities = [compartment.capacity for compartment in self.vehicle.compartments]

        # Apply the capacity constraints for 2D rotatable items
        load_expr = (
            lambda j_, k_: I[j_].length * z[j_, k_]
            + I[j_].width * y[j_, k_]
            - I[j_].width * z[j_, k_]
        )
        # 7.1
        self.model.addConstrs(
            (
                s[j, k] >= s[i, k] + load_expr(j, k) - Capacities[k] * (1 - x[i, j])
                for i in V
                for j in P
                for k in M
            ),
            name="stack_after_pickup_constr",
        )
        # 7.2
        self.model.addConstrs(
            (
                s[j, k] <= s[i, k] + load_expr(j, k) + Capacities[k] * (1 - x[i, j])
                for i in V
                for j in P
                for k in M
            ),
            name="stack_after_pickup_constr",
        )

        unload_expr = lambda j_, k_: -1 * load_expr(j_, k_)
        # 8.1
        self.model.addConstrs(
            (
                s[N + j, k]
                >= s[i, k] + unload_expr(j, k) - Capacities[k] * (1 - x[i, N + j])
                for i in V
                for j in P
                for k in M
            ),
            name="stack_after_delivery_constr",
        )
        # 8.2
        self.model.addConstrs(
            (
                s[N + j, k]
                <= s[i, k] + unload_expr(j, k) + Capacities[k] * (1 - x[i, N + j])
                for i in V
                for j in P
                for k in M
            ),
            name="stack_after_delivery_constr",
        )

        # Add the LIFO constraint for 2D rotatable items
        # 9.1
        self.model.addConstrs(
            (
                s[N + j, k]
                >= s[j, k] + unload_expr(j, k) - Capacities[k] * (1 - y[j, k])
                for j in P
                for k in M
            ),
            name="stack_LIFO_constr",
        )
        # 9.2
        self.model.addConstrs(
            (
                s[N + j, k]
                <= s[j, k] + unload_expr(j, k) + Capacities[k] * (1 - y[j, k])
                for j in P
                for k in M
            ),
            name="stack_LIFO_constr",
        )

        # Add the Item-Compartment compatibility constraints
        Capabilities = self.compartment_capabilities
        # 17
        self.model.addConstrs(
            (y[i, k] <= Capabilities[k][i] for i in P for k in M),
            name="item_comp_constr",
        )

    def _apply_rotation_constraints(self):
        _, _, P, _, _, M = self.problem_data()
        I = self.items
        y = self.data["y"]
        r = self.data["r"]
        z = self.data["z"]

        # 14 z[i, k] = r[i] & y[i, k] (1/3)
        self.model.addConstrs(
            (z[i, k] <= r[i] for i in P for k in M), name="z_r_constr"
        )

        # 15 z[i, k] = r[i] & y[i, k] (2/3)
        self.model.addConstrs(
            (z[i, k] <= y[i, k] for i in P for k in M), name="z_y_constr"
        )

        # 16 z[i, k] = r[i] & y[i, k] (3/3)
        self.model.addConstrs(
            (z[i, k] >= r[i] + y[i, k] - 1 for i in P for k in M), name="z_r_y_constr"
        )

        # 18 If an item has the same length of the compartment it's assigned to, then
        # the item is not rotate.
        same_length = (
            lambda i_, k_: I[i_].length == self.vehicle.compartments[k_].length
        )
        self.model.addConstrs(
            (y[i, k] <= 1 - r[i] for i in P for k in M if same_length(i, k)),
            name="item_comp_not_rot_constr",
        )
        # 19 If an item doesn't have the same length of the compartment it's assigned
        # to, then the item is not rotate.
        self.model.addConstrs(
            (y[i, k] <= r[i] for i in P for k in M if not same_length(i, k)),
            name="item_comp_rot_constr",
        )

    def apply_constraints(self):
        # Apply base constraints
        self._apply_base_constraints()
        # Apply compartment constraints (capacity & loading & compatibility)
        self._apply_compartment_constraints()
        # Apply rotation constraints
        self._apply_rotation_constraints()
        # Update model with constraints
        self.model.update()

    def set_model_objective(
        self,
        optimization_objective: OptimizationObjective = (
            OptimizationObjective.CHEAPEST_ROUTE
        ),
    ):
        """
        Set the objective for the gurobi model. The default objective minimizes
        the route cost.

        Applicable objectives for the problem:
            - `OptimizationObjective.CHEAPEST_ROUTE`
            - `OptimizationObjective.LEAST_ROTATION`
            - `OptimizationObjective.CHEAPEST_ROUTE_LEAST_ROTATION`
        """
        _, V, P, _, C, _ = self.problem_data()
        x, _, _, _ = self.model_vars()
        r = self.data["r"]

        route_cost_expr = sum(C[i][j] * x[i, j] for i in V for j in V)
        rotation_cost_expr = sum(r[i] for i in P)

        if optimization_objective == OptimizationObjective.CHEAPEST_ROUTE:
            objective_expr = route_cost_expr
        elif optimization_objective == OptimizationObjective.LEAST_ROTATION:
            objective_expr = rotation_cost_expr
        elif (
            optimization_objective
            == OptimizationObjective.CHEAPEST_ROUTE_LEAST_ROTATION
        ):
            objective_expr = route_cost_expr + rotation_cost_expr
        else:
            raise ValueError("This objective is not applicable to this problem.")

        self.model.setObjective(objective_expr)
