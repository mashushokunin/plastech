from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from plastech.candidate_constraints import (  # noqa: E402
    CandidateConstraints,
    add_prediction_intervals,
    apply_candidate_constraints,
    load_prediction_errors,
)
from plastech.inverse_design import ForwardModelSet, TargetSpecification  # noqa: E402


class CandidateConstraintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.models = ForwardModelSet(
            specifications=(),
            models={},
            prepared={"dummy": SimpleNamespace(feature_columns=["a", "b", "c"])},
        )
        self.candidates = pd.DataFrame(
            {
                "a": [10.0, 10.0],
                "b": [5.0, 0.0],
                "c": [1.0, 0.0],
                "primary_pvc": ["PVC-A", "(no PVC)"],
                "primary_plasticizer": ["P-A", "P-B"],
            }
        )

    def test_constraints_keep_candidates_and_record_reasons(self) -> None:
        constraints = CandidateConstraints(
            max_material_count=2,
            min_total_formulation_amount=11.0,
            forbidden_columns=("c",),
        )
        result = apply_candidate_constraints(self.candidates, self.models, constraints)
        self.assertEqual(len(result), 2)
        self.assertFalse(bool(result.loc[0, "practical_constraint_pass"]))
        self.assertIn("材料数>2", result.loc[0, "practical_constraint_violations"])
        self.assertIn("使用禁止原料", result.loc[0, "practical_constraint_violations"])
        self.assertIn("総配合量<11", result.loc[1, "practical_constraint_violations"])
        self.assertIn("主材料", result.loc[1, "practical_constraint_violations"])

    def test_prediction_interval_uses_conservative_edge(self) -> None:
        specifications = (
            TargetSpecification("hardness", "硬度", 70.0, 4.0),
            TargetSpecification("elongation", "伸び", 370.0, 40.0, "lower_bound"),
            TargetSpecification("brittleness", "脆化", -41.0, 4.0, "minimize"),
        )
        candidates = pd.DataFrame(
            {
                "predicted__hardness": [70.0],
                "predicted__elongation": [430.0],
                "predicted__brittleness": [-55.0],
            }
        )
        result = add_prediction_intervals(
            candidates,
            specifications,
            {"hardness": 7.8, "elongation": 50.8, "brittleness": 6.72},
        )
        self.assertFalse(bool(result.loc[0, "p90_robust_pass__hardness"]))
        self.assertTrue(bool(result.loc[0, "p90_robust_pass__elongation"]))
        self.assertTrue(bool(result.loc[0, "p90_robust_pass__brittleness"]))
        self.assertFalse(bool(result.loc[0, "p90_robust_pass_all"]))

    def test_json_loaders(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            constraints_path = root / "constraints.json"
            error_path = root / "errors.json"
            constraints_path.write_text(
                json.dumps({"max_material_count": 7, "forbidden_columns": ["x"]}),
                encoding="utf-8",
            )
            error_path.write_text(json.dumps({"errors": {"x": 1.25}}), encoding="utf-8")
            constraints = CandidateConstraints.from_json(constraints_path)
            self.assertEqual(constraints.max_material_count, 7)
            self.assertEqual(constraints.forbidden_columns, ("x",))
            self.assertEqual(load_prediction_errors(error_path), {"x": 1.25})

    def test_invalid_total_range_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CandidateConstraints(
                min_total_formulation_amount=100.0,
                max_total_formulation_amount=90.0,
            )


if __name__ == "__main__":
    unittest.main()
