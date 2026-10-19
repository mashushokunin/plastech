from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from plastech.inverse_design import (  # noqa: E402
    DesignBrief,
    ForwardModelSet,
    TargetSpecification,
    add_design_brief_status,
    add_formulation_material_count,
    add_preference_status,
    score_design_preferences,
    select_diverse_candidates,
)


class DesignBriefRoleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.objective = TargetSpecification("objective", "最適化物性", -40.0, 5.0, "upper_bound")
        self.management = TargetSpecification("management", "管理物性", 70.0, 4.0)
        self.constraint = TargetSpecification(
            "constraint",
            "制約物性",
            370.0,
            40.0,
            "lower_bound",
        )
        self.monitor = TargetSpecification("monitor", "監視物性", 1.34, 0.06)
        self.brief = DesignBrief(
            name="test",
            description="test brief",
            preferences=(self.objective,),
            management_targets=(self.management,),
            constraints=(self.constraint,),
            monitors=(self.monitor,),
        )

    def test_property_cannot_have_multiple_roles(self) -> None:
        with self.assertRaisesRegex(ValueError, "only one role"):
            DesignBrief(
                name="invalid",
                description="duplicate property",
                preferences=(self.objective,),
                monitors=(self.objective,),
            )

    def test_monitor_does_not_change_design_score(self) -> None:
        candidates = pd.DataFrame(
            {
                "predicted__objective": [-42.0, -42.0],
                "predicted__monitor": [1.20, 1.60],
            }
        )
        scored = score_design_preferences(candidates, self.brief.objectives)
        self.assertEqual(scored.loc[0, "design_score"], scored.loc[1, "design_score"])

    def test_minimize_objective_rewards_improvement_beyond_reference(self) -> None:
        minimize = TargetSpecification(
            "temperature",
            "脆化温度",
            -41.0,
            4.0,
            "minimize",
        )
        candidates = pd.DataFrame({"predicted__temperature": [-41.0, -45.0]})
        scored = score_design_preferences(candidates, (minimize,))
        self.assertEqual(scored.iloc[0]["predicted__temperature"], -45.0)
        self.assertLess(scored.iloc[0]["design_score"], scored.iloc[1]["design_score"])

    def test_role_status_is_reported_separately(self) -> None:
        candidates = pd.DataFrame(
            {
                "predicted__objective": [-42.0, -38.0],
                "predicted__management": [70.0, 80.0],
                "predicted__constraint": [390.0, 350.0],
                "predicted__monitor": [1.50, 1.34],
            }
        )
        result = add_design_brief_status(candidates, self.brief)
        self.assertTrue(bool(result.loc[0, "predicted_meets_objectives"]))
        self.assertTrue(bool(result.loc[0, "predicted_meets_requirements"]))
        self.assertFalse(bool(result.loc[0, "predicted_meets_monitors"]))
        self.assertFalse(bool(result.loc[1, "predicted_meets_requirements"]))
        self.assertEqual(result.loc[1, "predicted_requirement_violation_count"], 2)

    def test_decimal_boundary_is_inclusive(self) -> None:
        gravity = TargetSpecification("gravity", "比重", 1.34, 0.05)
        candidates = pd.DataFrame({"predicted__gravity": [1.29, 1.39]})
        result = add_preference_status(candidates, (gravity,))
        self.assertTrue(result["predicted_meets_preferences"].all())

    def test_candidate_selection_prioritizes_requirements(self) -> None:
        candidates = pd.DataFrame(
            {
                "row_id": ["fail-best", "pass-a", "pass-b", "fail-c"],
                "primary_pvc": ["PVC0", "PVC1", "PVC2", "PVC3"],
                "primary_plasticizer": ["P0", "P1", "P2", "P3"],
                "design_score": [0.0, 0.1, 0.2, 0.3],
                "pareto_optimal": [True, True, True, True],
                "predicted_meets_requirements": [False, True, True, False],
                "predicted_requirement_violation_count": [1, 0, 0, 1],
            }
        )
        selected = select_diverse_candidates(candidates, top_n=2)
        self.assertEqual(selected["row_id"].tolist(), ["pass-a", "pass-b"])

    def test_candidate_selection_fills_requested_count(self) -> None:
        candidates = pd.DataFrame(
            {
                "row_id": ["a", "b", "c"],
                "primary_pvc": ["PVC", "PVC", "PVC"],
                "primary_plasticizer": ["P", "P", "P"],
                "design_score": [0.0, 0.1, 0.2],
                "pareto_optimal": [True, False, False],
            }
        )
        selected = select_diverse_candidates(candidates, top_n=3)
        self.assertEqual(selected["row_id"].tolist(), ["a", "b", "c"])

    def test_formulation_material_count_uses_nonzero_features(self) -> None:
        candidates = pd.DataFrame({"a": [1.0, 0.0], "b": [0.2, 0.0], "ignored": [5, 5]})
        models = ForwardModelSet(
            specifications=(),
            models={},
            prepared={"dummy": SimpleNamespace(feature_columns=["a", "b"])},
        )
        result = add_formulation_material_count(candidates, models)
        self.assertEqual(result["formulation_material_count"].tolist(), [2, 0])


if __name__ == "__main__":
    unittest.main()
