import numpy as np
import pandas as pd

from mar_missing import MAR


def _df(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "a": rng.normal(size=rows),
            "b": rng.normal(size=rows),
            "c": rng.normal(size=rows),
            "d": rng.normal(size=rows),
            "e": rng.normal(size=rows),
            "f": rng.normal(size=rows),
        }
    )


def _model(x: pd.DataFrame) -> np.ndarray:
    logits = x.sum(axis=1).to_numpy(dtype=float)
    return 1.0 / (1.0 + np.exp(-logits))


def test_split_mode_keeps_shape_and_index_and_tracks_splits() -> None:
    df = _df()
    mar = MAR(
        dataset=df,
        missing_rate=0.2,
        same_miss_prob=True,
        d_deter=2,
        split=3,
        model=_model,
        seed=2026,
    )

    out = mar.apply(get_statistics=False)

    assert out.shape == df.shape
    assert out.index.equals(df.index)
    assert hasattr(mar, "processed_splits")
    assert len(mar.processed_splits) == 3
    assert 0 < out.isna().sum().sum() < out.size


def test_each_split_has_valid_variable_partition() -> None:
    df = _df()
    mar = MAR(
        dataset=df,
        missing_rate=0.2,
        same_miss_prob=True,
        d_deter=2,
        split=4,
        model=_model,
        seed=99,
    )

    mar.apply(get_statistics=False)

    all_cols = set(df.columns)
    for split_data, deter_names, target_names in mar.processed_splits:
        assert len(deter_names) == 2
        assert set(deter_names).isdisjoint(set(target_names))
        assert set(deter_names).union(set(target_names)) == all_cols
        assert set(split_data.columns) == all_cols
