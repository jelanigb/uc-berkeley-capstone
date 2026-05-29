"""
SegmentAuditor — evaluate trained models on per-segment subsets of the validation set.

Segments are broken out by vertical and tier independently (not crossed). This reveals
whether a global model has blind spots across content domains or channel sizes, and
informs whether segment-specific models are worth training.

Requires df_val (pre-feature-engineering split, with 'vertical' and 'tier' string columns)
aligned by row position with X_val / y_val.

Usage
-----
    from evaluation.segment_auditor import SegmentAuditor

    auditor = SegmentAuditor(
        models=run.models,          # dict keyed by model name; values are run.models entries
        X_val=run.X_val,            # scaled validation feature matrix (DataFrame or ndarray)
        y_val=run.y_val,            # binary validation labels (above_baseline)
        df_val=run.df_val,          # pre-FE split — must have 'vertical' and 'tier' columns
        threshold=0.58,             # optimized threshold from threshold sweep
    )

    segment_results = auditor.audit()
    auditor.print_summary(segment_results)
    segment_results.to_csv('outputs/segment_audit.csv', index=False)
"""

import warnings
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

_MIN_SEGMENT_SAMPLES = 30


class SegmentAuditor:
    """
    Evaluates trained models on per-segment subsets of the validation set.

    Segments are broken out by vertical and tier independently (not crossed).
    Requires df_val (pre-feature-engineering split) to extract string segment labels,
    aligned by row position with X_val / y_val.
    """

    def __init__(
        self,
        models: dict,
        X_val: Union[np.ndarray, pd.DataFrame],
        y_val: pd.Series,
        df_val: pd.DataFrame,
        threshold: float = 0.5,
    ):
        self.models = models
        self.X_val = X_val
        self.y_val = y_val
        self.df_val = df_val
        self.threshold = threshold

    def audit(self) -> pd.DataFrame:
        """
        Run per-segment evaluation for all models.
        Returns a tidy DataFrame with one row per (model, segment_type, segment_value).
        """
        x_rows = self.X_val.shape[0]
        assert len(self.df_val) == x_rows, (
            f"Row count mismatch: df_val={len(self.df_val)}, X_val={x_rows}"
        )

        for col in ("vertical", "tier"):
            if col not in self.df_val.columns:
                raise KeyError(f"df_val is missing required column: '{col}'")

        vertical_labels = self.df_val["vertical"].reset_index(drop=True)
        tier_labels = self.df_val["tier"].reset_index(drop=True)

        SEGMENT_DIMENSIONS = {
            "vertical": vertical_labels,
            "tier": tier_labels,
        }

        records = []
        for model_name, model_entry in self.models.items():
            model_obj = self._extract_model(model_name, model_entry)

            for dimension_name, label_series in SEGMENT_DIMENSIONS.items():
                for segment_value in label_series.unique():
                    mask = (label_series == segment_value).values
                    X_seg = self.X_val[mask]
                    y_seg = self.y_val.values[mask]

                    if len(y_seg) < _MIN_SEGMENT_SAMPLES:
                        warnings.warn(
                            f"[SegmentAuditor] Skipping '{dimension_name}={segment_value}' "
                            f"for model '{model_name}': only {len(y_seg)} samples "
                            f"(minimum is {_MIN_SEGMENT_SAMPLES}).",
                            UserWarning,
                            stacklevel=2,
                        )
                        continue

                    y_prob = model_obj.predict_proba(X_seg)[:, 1]
                    y_pred = (y_prob >= self.threshold).astype(int)

                    try:
                        roc_auc = roc_auc_score(y_seg, y_prob)
                    except ValueError:
                        roc_auc = np.nan

                    records.append({
                        "model": model_name,
                        "segment_type": dimension_name,
                        "segment_value": str(segment_value),
                        "n_samples": len(y_seg),
                        "pct_positive": float(y_seg.mean()),
                        "roc_auc": roc_auc,
                        "accuracy": accuracy_score(y_seg, y_pred),
                        # Global (macro) metrics — unweighted mean across both
                        # classes, listed before the per-class breakdown.
                        "precision_macro": precision_score(y_seg, y_pred, average="macro", zero_division=0),
                        "recall_macro": recall_score(y_seg, y_pred, average="macro", zero_division=0),
                        "f1_macro": f1_score(y_seg, y_pred, average="macro", zero_division=0),
                        "precision_above": precision_score(y_seg, y_pred, zero_division=0),
                        "recall_above": recall_score(y_seg, y_pred, zero_division=0),
                        "f1_above": f1_score(y_seg, y_pred, zero_division=0),
                        "recall_below": recall_score(1 - y_seg, 1 - y_pred, zero_division=0),
                    })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values(
                ["segment_type", "model", "segment_value"]
            ).reset_index(drop=True)
        return df

    def print_summary(self, results: pd.DataFrame) -> None:
        """Print a formatted summary table grouped by segment_type."""
        try:
            from tabulate import tabulate as _tabulate
        except ImportError:
            _tabulate = None

        for dim in ("vertical", "tier"):
            block = results[results["segment_type"] == dim]
            if block.empty:
                continue

            print(f"\n{'=' * 64}")
            print(f"  Segment audit — {dim.upper()}")
            print(f"{'=' * 64}")

            global_rows = []
            for model_name, model_entry in self.models.items():
                model_obj = self._extract_model(model_name, model_entry)
                g = self._global_metrics_(model_name, model_obj)
                global_rows.append(g)
            global_df = pd.DataFrame(global_rows).set_index("model")

            pivot_auc = block.pivot_table(
                index="segment_value", columns="model", values="roc_auc"
            )
            pivot_acc = block.pivot_table(
                index="segment_value", columns="model", values="accuracy"
            )

            # Append the ALL (global) reference row
            available = [m for m in global_df.index if m in pivot_auc.columns]
            pivot_auc.loc["ALL"] = {m: global_df.loc[m, "roc_auc"] for m in available}
            pivot_acc.loc["ALL"] = {m: global_df.loc[m, "accuracy"] for m in available}

            combined = pd.concat(
                [pivot_auc.add_suffix("  auc"), pivot_acc.add_suffix("  acc")],
                axis=1,
            ).round(4)
            combined.index.name = dim

            if _tabulate is not None:
                print(_tabulate(combined, headers="keys", tablefmt="simple", floatfmt=".4f"))
            else:
                print(combined.to_string())

            self._print_blind_spots_(block, global_df, dim)

    # ── private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_model(model_name: str, model_entry):
        """Return the raw sklearn model from a run.models entry or plain model object."""
        if isinstance(model_entry, dict):
            if "model" not in model_entry:
                raise AttributeError(
                    f"Model entry '{model_name}' is a dict but has no 'model' key. "
                    "Expected run.models entry format: {{'model': ..., 'scaler': ..., ...}}"
                )
            model_obj = model_entry["model"]
        else:
            model_obj = model_entry

        if not hasattr(model_obj, "predict_proba"):
            raise AttributeError(
                f"Model '{model_name}' does not implement predict_proba. "
                "SegmentAuditor requires probability estimates."
            )
        return model_obj

    def _global_metrics_(self, model_name: str, model_obj) -> dict:
        """Compute full-validation-set roc_auc and accuracy for one model."""
        y_prob = model_obj.predict_proba(self.X_val)[:, 1]
        y_pred = (y_prob >= self.threshold).astype(int)
        y_true = self.y_val.values
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            roc_auc = np.nan
        return {
            "model": model_name,
            "roc_auc": roc_auc,
            "accuracy": accuracy_score(y_true, y_pred),
        }

    @staticmethod
    def _print_blind_spots_(
        block: pd.DataFrame, global_df: pd.DataFrame, dim: str, drop_threshold: float = 0.03
    ) -> None:
        """Flag segments where roc_auc drops more than drop_threshold below global."""
        print(f"\n  Blind spots (roc_auc > {drop_threshold:.2f} below global):")
        found_any = False
        for model_name in block["model"].unique():
            if model_name not in global_df.index:
                continue
            global_auc = global_df.loc[model_name, "roc_auc"]
            model_block = block[block["model"] == model_name]
            for _, row in model_block.iterrows():
                if pd.isna(row["roc_auc"]):
                    continue
                drop = global_auc - row["roc_auc"]
                if drop > drop_threshold:
                    print(
                        f"    [{model_name}] {dim}={row['segment_value']:<12} "
                        f"auc={row['roc_auc']:.4f}  (global={global_auc:.4f}, "
                        f"drop={drop:.4f})"
                    )
                    found_any = True
        if not found_any:
            print("    None — model performs uniformly across all segments.")
