import numpy as np
import pandas as pd
import pytest

from mar_missing import MAR


def _base_df(rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    return pd.DataFrame(
        {
            "x1": rng.normal(size=rows),
            "x2": rng.normal(size=rows),
            "x3": rng.normal(size=rows),
            "x4": rng.normal(size=rows),
        }
    )


def test_raises_for_missing_values_in_input_dataset() -> None:
    df = _base_df()
    df.loc[0, "x1"] = np.nan

    with pytest.raises(ValueError, match="contains missing values"):
        MAR(
            dataset=df,
            missing_rate=0.2,
            target_vars=["x1", "x2"],
            deter_vars=["x3", "x4"],
            weights=[0.5, -0.2],
        )


def test_raises_for_invalid_missing_rate() -> None:
    df = _base_df()

    with pytest.raises(ValueError, match="missing_rate"):
        MAR(
            dataset=df,
            missing_rate=1.2,
            target_vars=["x1", "x2"],
            deter_vars=["x3", "x4"],
            weights=[0.5, -0.2],
        )


def test_raises_for_overlap_between_target_and_deter_vars() -> None:
    df = _base_df()

    with pytest.raises(ValueError, match="must not overlap"):
        MAR(
            dataset=df,
            missing_rate=0.2,
            target_vars=["x1", "x2"],
            deter_vars=["x2", "x3"],
            weights=[0.5, -0.2],
        )


def test_raises_when_split_provided_without_d_deter() -> None:
    df = _base_df()

    with pytest.raises(
        ValueError,
        match="split should be provided together with d_deter|Either d_deter or target_vars/deter_vars must be provided",
    ):
        MAR(
            dataset=df,
            missing_rate=0.2,
            split=2,
            weights_range=(-1.0, 1.0),
        )
