import pandas as pd


def index(idx):
    if isinstance(idx, pd.MultiIndex):
        x = list(idx)
        return [x[0]] + [[" " if x[i][j] == z else z for j, z in enumerate(y)] for i, y in enumerate(x[1:])]
    return [[i] for i in idx]


def float_fmt(x, fmt):
    if not isinstance(x, float):
        return str(x)
    return format(x, fmt)


def table(df: pd.DataFrame, fmt="f"):
    idx = index(df.index)
    cols = index(df.columns)
    M = df.to_numpy(dtype=object)
    return (
        "|"
        + (
            "|\n|".join(
                "|".join(" " for _ in range(len(idx[0]))) + "|" + "|".join(c[i] for c in cols)
                for i in range(len(cols[0]))
            )
            + "|\n|"
            + "+".join("-" for _ in range(len(idx[0]) + len(M[0])))
            + "|\n|"
            + "|\n|".join("|".join(float_fmt(i, fmt) for j in z for i in j) for z in zip(idx, M))
        )
        + "|"
    )
