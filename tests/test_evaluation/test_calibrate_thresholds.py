"""Tests for the threshold calibration script.

The live-pipeline path is exercised by `scripts/calibrate_thresholds.py`
against real services; here we just validate the math + persistence
helpers in isolation so they stay regression-safe.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from scripts.calibrate_thresholds import (
    _band_to_label,
    _calibrate,
    _from_results,
    _lexical_baseline,
    _persist,
    _sweep_thresholds,
)


class TestBandToLabel:
    """Band -> binary positive/negative label."""

    def test_high_is_positive(self):
        assert _band_to_label("high") == 1

    def test_medium_is_positive(self):
        assert _band_to_label("medium") == 1

    def test_low_is_negative(self):
        assert _band_to_label("low") == 0

    def test_unknown_band_is_negative(self):
        # Conservative: treat noise as negative rather than positive
        assert _band_to_label("garbage") == 0
        assert _band_to_label("") == 0
        assert _band_to_label(None) == 0  # type: ignore[arg-type]

    def test_case_insensitive(self):
        assert _band_to_label("HIGH") == 1
        assert _band_to_label("Low") == 0


class TestSweepThresholds:
    """Threshold sweep + Youden's J selection."""

    def test_perfect_separation_picks_optimal(self):
        # Positives all above 0.8, negatives all below 0.4 -> any threshold
        # in (0.4, 0.8] gives J=1.0. Picker returns one of them.
        pairs = [
            (0.9, 1),
            (0.85, 1),
            (0.95, 1),
            (0.1, 0),
            (0.2, 0),
            (0.35, 0),
        ]
        out = _sweep_thresholds(pairs)
        assert 0.4 <= out["chosen_threshold"] <= 0.85
        assert out["chosen_metrics"]["j"] == pytest.approx(1.0, abs=1e-6)
        assert out["chosen_metrics"]["precision"] == pytest.approx(1.0)
        assert out["chosen_metrics"]["recall"] == pytest.approx(1.0)
        assert out["n_pos"] == 3
        assert out["n_neg"] == 3

    def test_curve_covers_full_range(self):
        pairs = [(0.5, 1), (0.4, 0)]
        out = _sweep_thresholds(pairs)
        # Sweep step 0.05 -> 21 points from 0.0 to 1.0 inclusive
        assert len(out["curve"]) == 21
        thresholds = [p["threshold"] for p in out["curve"]]
        assert thresholds[0] == pytest.approx(0.0)
        assert thresholds[-1] == pytest.approx(1.0)

    def test_all_negative_returns_above_max_score(self):
        # No positives -> J = TPR - FPR = 0 - FPR. The best (least negative)
        # value is at thresholds high enough that every prediction is "no",
        # making FPR=0 and J=0. Picker should land at or past the max score
        # in the negative pool.
        pairs = [(0.1, 0), (0.2, 0), (0.3, 0)]
        out = _sweep_thresholds(pairs)
        assert out["n_pos"] == 0
        assert out["chosen_threshold"] >= 0.3
        assert out["chosen_metrics"]["fpr"] == pytest.approx(0.0)

    def test_all_positive_returns_zero_threshold(self):
        # No negatives -> J trivially 1.0 everywhere recall=1.
        pairs = [(0.5, 1), (0.7, 1), (0.9, 1)]
        out = _sweep_thresholds(pairs)
        assert out["n_neg"] == 0
        assert out["chosen_metrics"]["recall"] == pytest.approx(1.0)


class TestCalibrate:
    """End-to-end calibration over a synthetic results list."""

    def _row(
        self,
        rid: str,
        conf: float,
        faith: float,
        conf_band: str,
        faith_band: str,
        blocked: bool = False,
    ) -> dict:
        return {
            "id": rid,
            "category": "test",
            "expected_confidence_band": conf_band,
            "expected_faithfulness_band": faith_band,
            "expected_outcome": "answer" if not blocked else "block",
            "confidence_score": conf,
            "faithfulness_ratio": faith,
            "blocked": blocked,
            "answer": "" if blocked else "an answer",
            "ground_truth": "an answer",
            "contexts": ["context"] if not blocked else [],
            "latency_ms": 10.0,
        }

    def test_separates_signals_independently(self):
        results = [
            self._row("a", conf=0.9, faith=0.95, conf_band="high", faith_band="high"),
            self._row("b", conf=0.85, faith=0.9, conf_band="high", faith_band="high"),
            self._row("c", conf=0.1, faith=0.2, conf_band="low", faith_band="low"),
            self._row("d", conf=0.2, faith=0.3, conf_band="low", faith_band="low"),
        ]
        calib = _calibrate(results)
        assert calib["n_rows_total"] == 4
        assert calib["n_rows_usable"] == 4
        # Both sweeps should land in the obvious gap between positives + negatives
        assert 0.25 <= calib["confidence"]["chosen_threshold"] <= 0.85
        assert 0.35 <= calib["faithfulness"]["chosen_threshold"] <= 0.9

    def test_drops_blocked_rows_from_faithfulness_sweep(self):
        # Blocked rows have faithfulness_ratio defaulting to 1.0 -> would skew
        # the negative band. Calibration must exclude them.
        results = [
            self._row("a", conf=0.9, faith=0.9, conf_band="high", faith_band="high"),
            self._row(
                "b",
                conf=0.0,
                faith=1.0,
                conf_band="low",
                faith_band="low",
                blocked=True,
            ),
        ]
        calib = _calibrate(results)
        # Only one row enters faithfulness sweep -> n_total=1 not 2
        assert calib["faithfulness"]["n_total"] == 1
        # Confidence sweep still sees both rows (blocked still has confidence
        # signal — 0.0 for the blocked one).
        assert calib["confidence"]["n_total"] == 2

    def test_excludes_errored_rows(self):
        rows = [
            self._row("a", conf=0.9, faith=0.9, conf_band="high", faith_band="high"),
            {**self._row("b", 0.0, 0.0, "low", "low"), "error": "boom"},
        ]
        calib = _calibrate(rows)
        assert calib["n_rows_total"] == 2
        assert calib["n_rows_usable"] == 1


class TestLexicalBaseline:
    """Lexical Ragas fallback metrics."""

    def test_perfect_overlap(self):
        out = _lexical_baseline(
            [
                {
                    "answer": "alpha beta gamma",
                    "contexts": ["alpha beta gamma delta"],
                    "ground_truth": "alpha beta gamma",
                    "blocked": False,
                },
            ]
        )
        assert out["faithfulness"] == pytest.approx(1.0)
        assert out["context_precision"] == pytest.approx(1.0)
        assert out["answer_relevancy"] == pytest.approx(1.0)

    def test_skips_blocked_rows(self):
        out = _lexical_baseline(
            [
                {"answer": "", "contexts": [], "ground_truth": "x", "blocked": True},
            ]
        )
        assert out == {
            "faithfulness": 0.0,
            "context_precision": 0.0,
            "answer_relevancy": 0.0,
        }


class TestPersist:
    """Filesystem round-trip + JSON shape."""

    def test_writes_calibration_baseline_and_snapshot(self, tmp_path, monkeypatch):
        calib = {
            "timestamp": "2026-05-23T00:00:00Z",
            "confidence": {"chosen_threshold": 0.65},
            "faithfulness": {"chosen_threshold": 0.72},
            "n_rows_total": 1,
            "n_rows_usable": 1,
        }
        baseline = {"faithfulness": 0.9, "context_precision": 0.8, "answer_relevancy": 0.85}
        results = [{"id": "x", "confidence_score": 0.9}]

        monkeypatch.setattr("scripts.calibrate_thresholds.CALIB_PATH", tmp_path / "calib.json")
        monkeypatch.setattr(
            "scripts.calibrate_thresholds.BASELINE_PATH", tmp_path / "baseline.json"
        )
        monkeypatch.setattr("scripts.calibrate_thresholds.RESULTS_DIR", tmp_path / "results")

        _persist(calib, baseline, results)

        loaded_calib = json.loads((tmp_path / "calib.json").read_text(encoding="utf-8"))
        assert loaded_calib["confidence"]["chosen_threshold"] == 0.65

        loaded_baseline = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
        assert loaded_baseline["faithfulness"] == 0.9
        assert "_calibrated_at" in loaded_baseline

        snapshots = list((tmp_path / "results").glob("calibration_*.json"))
        assert len(snapshots) == 1
        snapshot = json.loads(snapshots[0].read_text(encoding="utf-8"))
        assert snapshot["per_row"] == results

    def test_preserves_prior_baseline_notes(self, tmp_path, monkeypatch):
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(
            json.dumps({"faithfulness": 0.5, "_note": "legacy hand-set"}), encoding="utf-8"
        )

        monkeypatch.setattr("scripts.calibrate_thresholds.CALIB_PATH", tmp_path / "calib.json")
        monkeypatch.setattr("scripts.calibrate_thresholds.BASELINE_PATH", baseline_path)
        monkeypatch.setattr("scripts.calibrate_thresholds.RESULTS_DIR", tmp_path / "results")

        _persist(
            {
                "timestamp": "t",
                "confidence": {"chosen_threshold": 0.5},
                "faithfulness": {"chosen_threshold": 0.5},
            },
            {"faithfulness": 0.91},
            [],
        )

        loaded = json.loads(baseline_path.read_text(encoding="utf-8"))
        assert loaded["faithfulness"] == 0.91  # updated
        assert loaded["_note"] == "legacy hand-set"  # preserved


class TestFromResults:
    """Re-derive calibration from a previous snapshot JSON without live runs."""

    def test_round_trip(self, tmp_path):
        snapshot = {
            "calibration": {"confidence": {"chosen_threshold": 0.55}},
            "per_row": [{"id": "a", "confidence_score": 0.8}],
        }
        p = tmp_path / "snap.json"
        p.write_text(json.dumps(snapshot), encoding="utf-8")
        rows, calib = _from_results(p)
        assert rows[0]["id"] == "a"
        assert calib["confidence"]["chosen_threshold"] == 0.55


class TestSettingsCalibrationOverride:
    """Settings ``_apply_calibration`` honours the JSON when env is unset."""

    def test_loads_thresholds_when_env_unset(self, tmp_path, monkeypatch):
        from config.settings import Settings, _apply_calibration

        monkeypatch.delenv("SAR_CONFIDENCE_THRESHOLD", raising=False)
        monkeypatch.delenv("SAR_FAITHFULNESS_THRESHOLD", raising=False)

        # Point the loader at a temp calibration.json by patching the module
        # constants the loader uses.
        calib_file = tmp_path / "calibration.json"
        calib_file.write_text(
            json.dumps(
                {
                    "confidence": {
                        "chosen_threshold": 0.42,
                        "n_pos": 10,
                        "n_neg": 10,
                    },
                    "faithfulness": {
                        "chosen_threshold": 0.66,
                        "n_pos": 8,
                        "n_neg": 6,
                    },
                }
            ),
            encoding="utf-8",
        )

        s = Settings()
        # Patch the path resolution inside _apply_calibration via the Path call
        with patch("config.settings.Path") as mock_path_cls:
            # Make any Path(...) chain ultimately resolve to our temp file
            mock_path_cls.return_value.resolve.return_value.parent.parent.__truediv__.return_value.__truediv__.return_value = calib_file
            _apply_calibration(s)

        assert s.confidence_threshold == pytest.approx(0.42)
        assert s.faithfulness_threshold == pytest.approx(0.66)

    def test_env_var_wins_over_calibration(self, tmp_path, monkeypatch):
        from config.settings import Settings, _apply_calibration

        monkeypatch.setenv("SAR_CONFIDENCE_THRESHOLD", "0.91")
        monkeypatch.delenv("SAR_FAITHFULNESS_THRESHOLD", raising=False)

        calib_file = tmp_path / "calibration.json"
        calib_file.write_text(
            json.dumps(
                {
                    "confidence": {
                        "chosen_threshold": 0.42,
                        "n_pos": 10,
                        "n_neg": 10,
                    },
                    "faithfulness": {
                        "chosen_threshold": 0.66,
                        "n_pos": 8,
                        "n_neg": 6,
                    },
                }
            ),
            encoding="utf-8",
        )

        s = Settings()
        # Env-set value is honoured at construction
        assert s.confidence_threshold == pytest.approx(0.91)

        with patch("config.settings.Path") as mock_path_cls:
            mock_path_cls.return_value.resolve.return_value.parent.parent.__truediv__.return_value.__truediv__.return_value = calib_file
            _apply_calibration(s)

        # Env var wins over the calibration file
        assert s.confidence_threshold == pytest.approx(0.91)
        # Faithfulness was unset in env -> calibration wins
        assert s.faithfulness_threshold == pytest.approx(0.66)

    def test_missing_calibration_file_is_no_op(self, tmp_path, monkeypatch):
        from config.settings import Settings, _apply_calibration

        monkeypatch.delenv("SAR_CONFIDENCE_THRESHOLD", raising=False)
        s = Settings()
        original = s.confidence_threshold

        with patch("config.settings.Path") as mock_path_cls:
            # Point at a file that doesn't exist
            mock_path_cls.return_value.resolve.return_value.parent.parent.__truediv__.return_value.__truediv__.return_value = (
                tmp_path / "nonexistent.json"
            )
            _apply_calibration(s)

        assert s.confidence_threshold == original

    def test_degenerate_sweep_is_ignored(self, tmp_path, monkeypatch):
        """When ``n_pos=0`` or ``n_neg=0`` the sweep has no statistical meaning;
        the loader must keep the prior defaults rather than apply a 0.0 cut-off.
        """
        from config.settings import Settings, _apply_calibration

        monkeypatch.delenv("SAR_CONFIDENCE_THRESHOLD", raising=False)
        monkeypatch.delenv("SAR_FAITHFULNESS_THRESHOLD", raising=False)

        # Smoke-style degenerate output: positives only, threshold defaults to 0
        bad_file = tmp_path / "calibration.json"
        bad_file.write_text(
            json.dumps(
                {
                    "confidence": {
                        "chosen_threshold": 0.0,
                        "n_pos": 3,
                        "n_neg": 0,
                    },
                    "faithfulness": {
                        "chosen_threshold": 0.0,
                        "n_pos": 1,
                        "n_neg": 0,
                    },
                }
            ),
            encoding="utf-8",
        )

        s = Settings()
        original_conf = s.confidence_threshold
        original_faith = s.faithfulness_threshold

        with patch("config.settings.Path") as mock_path_cls:
            mock_path_cls.return_value.resolve.return_value.parent.parent.__truediv__.return_value.__truediv__.return_value = bad_file
            _apply_calibration(s)

        assert s.confidence_threshold == original_conf
        assert s.faithfulness_threshold == original_faith

    def test_malformed_calibration_file_is_no_op(self, tmp_path, monkeypatch):
        from config.settings import Settings, _apply_calibration

        monkeypatch.delenv("SAR_CONFIDENCE_THRESHOLD", raising=False)
        bad_file = tmp_path / "calibration.json"
        bad_file.write_text("not valid json {{{", encoding="utf-8")

        s = Settings()
        original = s.confidence_threshold

        with patch("config.settings.Path") as mock_path_cls:
            mock_path_cls.return_value.resolve.return_value.parent.parent.__truediv__.return_value.__truediv__.return_value = bad_file
            _apply_calibration(s)

        assert s.confidence_threshold == original
