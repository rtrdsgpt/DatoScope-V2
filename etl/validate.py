"""
Data quality validation stage — Great Expectations checks on a DataFrame
before it's allowed to reach the processed zone / warehouse. Fails loudly
(raises DataQualityError) on violation instead of silently passing bad data
downstream.
"""

from __future__ import annotations

import great_expectations as gx
import pandas as pd

from etl.config import get_settings


class DataQualityError(Exception):
    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


def _build_suite(
    context,
    suite_name: str,
    columns: list[str],
    required_columns: list[str] | None,
    max_null_fraction: float,
    column_bounds: dict[str, dict[str, float]] | None,
):
    suite = context.suites.add(gx.ExpectationSuite(name=suite_name))
    suite.add_expectation(gx.expectations.ExpectTableRowCountToBeBetween(min_value=1))

    for col in required_columns or []:
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column=col))

    for col in columns:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=col, mostly=1 - max_null_fraction)
        )

    for col, bounds in (column_bounds or {}).items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=col, min_value=bounds.get("min"), max_value=bounds.get("max")
            )
        )

    return suite


def validate_dataframe(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    required_columns: list[str] | None = None,
    max_null_fraction: float | None = None,
    column_bounds: dict[str, dict[str, float]] | None = None,
    raise_on_failure: bool = True,
) -> dict:
    """
    Run schema/null/range checks against `df`. Raises DataQualityError
    (containing the full failure report) when validation fails and
    `raise_on_failure` is True.
    """
    settings = get_settings()
    if max_null_fraction is None:
        max_null_fraction = settings.max_null_fraction

    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("pandas")
    asset = data_source.add_dataframe_asset(name="asset")
    batch_definition = asset.add_batch_definition_whole_dataframe("batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = _build_suite(
        context,
        suite_name=f"{dataset_name}_suite",
        columns=df.columns.tolist(),
        required_columns=required_columns,
        max_null_fraction=max_null_fraction,
        column_bounds=column_bounds,
    )
    result = batch.validate(suite)

    failed = [
        {"expectation": r.expectation_config.type, "kwargs": dict(r.expectation_config.kwargs), "result": r.result}
        for r in result.results
        if not r.success
    ]

    report = {
        "dataset_name": dataset_name,
        "success": bool(result.success),
        "n_expectations": len(result.results),
        "n_failed": len(failed),
        "failed_expectations": failed,
    }

    if not result.success and raise_on_failure:
        raise DataQualityError(
            f"Data quality validation failed for '{dataset_name}': {len(failed)} of "
            f"{len(result.results)} expectation(s) failed",
            report,
        )

    return report
