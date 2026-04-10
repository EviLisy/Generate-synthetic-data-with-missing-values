import numpy as np
import pandas as pd

from mar_missing import MAR

# NOTE ABOUT CURRENT RNG SEMANTICS:
# `MAR` currently uses NumPy global RNG (`np.random.seed(seed)` in `__init__`).
# Therefore, reproducibility is guaranteed only when the call sequence is controlled.
# In practice, construct-and-apply in the same order for each run you compare.


def _df(rows: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "x1": rng.normal(size=rows),
            "x2": rng.normal(size=rows),
            "x3": rng.normal(size=rows),
            "x4": rng.normal(size=rows),
        }
    )


def test_same_seed_gives_identical_missing_pattern() -> None:
    df = _df()

    # Controlled sequence: initialize with seed, then apply immediately.
    mar1 = MAR(
        dataset=df.copy(),
        missing_rate=0.3,
        target_vars=["x1", "x2"],
        deter_vars=["x3", "x4"],
        weights=[0.8, -0.4],
        seed=123,
    )
    out1 = mar1.apply(get_statistics=False)

    mar2 = MAR(
        dataset=df.copy(),
        missing_rate=0.3,
        target_vars=["x1", "x2"],
        deter_vars=["x3", "x4"],
        weights=[0.8, -0.4],
        seed=123,
    )
    out2 = mar2.apply(get_statistics=False)

    assert out1.isna().equals(out2.isna())
    assert mar1.final_missing_rate == mar2.final_missing_rate


def test_same_seed_split_mode_is_reproducible() -> None:
    df = _df(120)

    # Controlled sequence in split mode as well.
    mar1 = MAR(
        dataset=df.copy(),
        missing_rate=0.25,
        d_deter=2,
        split=3,
        weights_range=(-1.0, 1.0),
        seed=2024,
    )
    out1 = mar1.apply(get_statistics=False)

    mar2 = MAR(
        dataset=df.copy(),
        missing_rate=0.25,
        d_deter=2,
        split=3,
        weights_range=(-1.0, 1.0),
        seed=2024,
    )
    out2 = mar2.apply(get_statistics=False)

    assert out1.isna().equals(out2.isna())
