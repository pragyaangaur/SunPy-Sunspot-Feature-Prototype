"""
Does naive quadrature propagation reproduce SILSO's published sigma?

Run:  python scripts/quadrature_check.py

The question asked was whether
    sigma_month = sqrt(sum(sigma_day^2)) / N_days
matches SILSO's published monthly sigma, and if not, what effective correlation
between daily values the discrepancy implies.

This script computes, for every calendar month with daily coverage:

  A            = sum_d sigma_d^2
  B            = (sum_d sigma_d)^2
  N            = number of days with a value
  pooled       = sqrt(sum_d N_d sigma_d^2 / sum_d N_d)   <- SILSO's documented formula
  naive_disp   = sqrt(A) / N            <- the quadrature formula as asked, rho = 0
  naive_sem    = sqrt(sum_d sigma_d^2/N_d) / N
                                        <- same, but after the documented
                                           dispersion -> standard-error conversion
  rho_eff      = (N^2 sigma_pub^2 - A) / (B - A)
                                        <- the equicorrelation coefficient that
                                           would make naive quadrature reproduce
                                           the published value

rho_eff is a diagnostic, not a physical claim. A value outside [-1, 1] is
meaningful: it says the discrepancy cannot be explained by *any* correlation
structure, because the two numbers are not estimating the same quantity.

The same calculation is repeated for daily -> yearly, to confirm nothing here is
an artifact of one particular resampling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from silso_uncertainty import reader  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures"
OUTPUT = Path(__file__).resolve().parents[1] / "output"


def aggregate(daily: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Per-bin quadrature quantities from the daily series."""
    valid = daily.dropna(subset=["sunspot_number", "sigma", "n_observations"])
    sigma = valid["sigma"]
    n_obs = valid["n_observations"]

    grp = valid.resample(rule)
    out = pd.DataFrame({
        "mean_sn": grp["sunspot_number"].mean(),
        "n_days": grp["sunspot_number"].count(),
        "sum_n_obs": grp["n_observations"].sum(),
        "A": (sigma ** 2).resample(rule).sum(),
        "sum_sigma": sigma.resample(rule).sum(),
        "sum_n_sig2": (n_obs * sigma ** 2).resample(rule).sum(),
        "mean_n_obs": n_obs.resample(rule).mean(),
    })
    out["B"] = out["sum_sigma"] ** 2
    # SILSO's documented pooled formula.
    out["pooled"] = np.sqrt(out["sum_n_sig2"] / out["sum_n_obs"])
    # The quadrature formula exactly as posed, applied to the raw dispersions.
    out["naive_disp"] = np.sqrt(out["A"]) / out["n_days"]
    # The same, after converting each daily dispersion to a standard error.
    sem_sq = (sigma ** 2 / n_obs).resample(rule).sum()
    out["naive_sem"] = np.sqrt(sem_sq) / out["n_days"]
    return out[out["n_days"] > 0]


def rho_effective(published: pd.Series, A: pd.Series, B: pd.Series,
                  n: pd.Series) -> pd.Series:
    """Equicorrelation that would reconcile naive quadrature with ``published``."""
    denom = B - A
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (n ** 2 * published ** 2 - A) / denom
    return rho.where(denom > 0)


def describe(name: str, ratio: pd.Series) -> str:
    r = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    q = r.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return (f"{name:<34} n={len(r):>5}  median={r.median():8.4f}  "
            f"mean={r.mean():8.4f}  5%={q[0.05]:7.4f}  95%={q[0.95]:7.4f}")


def main() -> None:
    daily, _ = reader.read_daily_total()
    monthly, _ = reader.read_monthly_total()
    yearly, _ = reader.read_yearly_total()
    smoothed, _ = reader.read_monthly_smoothed()

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    # ---------------------------------------------------------------- monthly
    agg = aggregate(daily, "MS")
    joined = agg.join(monthly[["sigma", "n_observations"]].rename(
        columns={"sigma": "published", "n_observations": "published_n_obs"}), how="inner")
    joined = joined.dropna(subset=["published"])

    emit("=" * 78)
    emit("daily -> monthly")
    emit("=" * 78)
    emit(f"months compared: {len(joined)}  "
         f"({joined.index.min():%Y-%m} to {joined.index.max():%Y-%m})")
    emit()

    # Cross-check that we are reading the same rows SILSO is.
    nobs_match = (joined["sum_n_obs"] == joined["published_n_obs"]).mean()
    emit(f"sum of daily n_observations == published monthly n_observations: "
         f"{100 * nobs_match:.2f}% of months")
    pooled_err = (joined["pooled"] - joined["published"]).abs()
    emit(f"SILSO pooled formula reproduces published sigma: "
         f"max|diff| = {pooled_err.max():.4f}, "
         f"within file rounding (0.05) for {100 * (pooled_err <= 0.05).mean():.2f}% of months")
    emit()

    joined["ratio_disp"] = joined["naive_disp"] / joined["published"]
    joined["ratio_sem"] = joined["naive_sem"] / joined["published"]
    emit("ratio of naive quadrature to SILSO published monthly sigma")
    emit("-" * 78)
    emit(describe("naive (raw dispersions, rho=0)", joined["ratio_disp"]))
    emit(describe("naive (converted to SEM, rho=0)", joined["ratio_sem"]))
    emit()
    emit("expected if the discrepancy were purely 1/sqrt(N_days):")
    pred = 1 / np.sqrt(joined["n_days"])
    emit(describe("1/sqrt(N_days)", pred))
    emit(f"correlation of ratio_disp with 1/sqrt(N_days): "
         f"{joined['ratio_disp'].corr(pred):.4f}")
    emit()

    # ------------------------------------------------- effective correlation
    joined["rho_eff_disp"] = rho_effective(joined["published"], joined["A"],
                                           joined["B"], joined["n_days"])
    sem_A = (daily["sigma"] ** 2 / daily["n_observations"]).resample("MS").sum()
    sem_B = (daily["sigma"] / np.sqrt(daily["n_observations"])).resample("MS").sum() ** 2
    joined["rho_eff_sem"] = rho_effective(joined["published"],
                                          sem_A.reindex(joined.index),
                                          sem_B.reindex(joined.index),
                                          joined["n_days"])
    emit("implied effective correlation rho_eff")
    emit("-" * 78)
    for label, col in [("from raw dispersions", "rho_eff_disp"),
                       ("from standard errors", "rho_eff_sem")]:
        r = joined[col].replace([np.inf, -np.inf], np.nan).dropna()
        frac = (r.between(-1, 1)).mean()
        emit(f"{label:<24} median={r.median():9.4f}  "
             f"5%={r.quantile(0.05):9.4f}  95%={r.quantile(0.95):9.4f}  "
             f"fraction in [-1,1]: {100 * frac:.2f}%")
    emit()

    # ------------------------------------------------ stability of the effect
    emit("is the discrepancy stable, or does it track solar cycle / station count?")
    emit("-" * 78)
    sm = smoothed["sunspot_number_smoothed"].reindex(joined.index)
    for label, series in [("smoothed SN (cycle phase)", sm),
                          ("monthly mean SN", joined["mean_sn"]),
                          ("mean stations per day", joined["mean_n_obs"]),
                          ("days in month", joined["n_days"].astype(float))]:
        common = pd.concat([joined["ratio_disp"], series], axis=1).dropna()
        emit(f"  corr(ratio_disp, {label:<26}) = "
             f"{common.iloc[:, 0].corr(common.iloc[:, 1]):7.4f}")
    emit()
    era = joined.groupby(joined.index.year // 25 * 25)["ratio_disp"]
    emit("  ratio_disp by 25-year era:")
    for start, grp in era:
        emit(f"    {start}-{start + 24}: n={len(grp):>4}  median={grp.median():.4f}  "
             f"IQR=[{grp.quantile(.25):.4f}, {grp.quantile(.75):.4f}]")
    emit()

    # ----------------------------------------------------------- daily->yearly
    emit("=" * 78)
    emit("daily -> yearly (second aggregation, control)")
    emit("=" * 78)
    yagg = aggregate(daily, "YS")
    yj = yagg.join(yearly[["sigma"]].rename(columns={"sigma": "published"}), how="inner")
    yj = yj.dropna(subset=["published"])
    yj["ratio_disp"] = yj["naive_disp"] / yj["published"]
    yj["ratio_sem"] = yj["naive_sem"] / yj["published"]
    yj["rho_eff_disp"] = rho_effective(yj["published"], yj["A"], yj["B"], yj["n_days"])
    pooled_err_y = (yj["pooled"] - yj["published"]).abs()
    emit(f"years compared: {len(yj)} ({yj.index.min():%Y}-{yj.index.max():%Y})")
    emit(f"SILSO pooled formula reproduces published yearly sigma: "
         f"max|diff| = {pooled_err_y.max():.4f}, "
         f"within 0.05 for {100 * (pooled_err_y <= 0.05).mean():.2f}% of years")
    emit(describe("naive (raw dispersions, rho=0)", yj["ratio_disp"]))
    emit(describe("naive (converted to SEM, rho=0)", yj["ratio_sem"]))
    emit(describe("1/sqrt(N_days)", 1 / np.sqrt(yj["n_days"])))
    r = yj["rho_eff_disp"].replace([np.inf, -np.inf], np.nan).dropna()
    emit(f"rho_eff (raw dispersions): median={r.median():.4f}  "
         f"fraction in [-1,1]: {100 * r.between(-1, 1).mean():.2f}%")
    emit()

    _plot(joined, yj)
    emit(f"figure written to {FIGURES / 'quadrature_check.png'}")

    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "quadrature_check.txt").write_text("\n".join(lines) + "\n")
    joined.to_csv(OUTPUT / "monthly_comparison.csv")


def _plot(monthly: pd.DataFrame, yearly: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)

    split = pd.Timestamp("1981-01-01")

    def mark_1981(ax, y=0.03):
        ax.axvline(split, color="0.4", lw=0.8, ls="-.")
        ax.text(split, y, "  1981: single station -> network", transform=
                ax.get_xaxis_transform(), fontsize=7, color="0.3", rotation=90,
                va="bottom")

    ax = axes[0]
    ax.plot(monthly.index, monthly["published"], lw=2.2, color="0.75", solid_capstyle="butt",
            label="SILSO published monthly $\\sigma$")
    ax.plot(monthly.index, monthly["pooled"], lw=0.6, ls="--", color="tab:green",
            label="reproduced from SILSO's pooled formula")
    ax.plot(monthly.index, monthly["naive_disp"], lw=0.6, color="tab:red",
            label=r"naive $\sqrt{\Sigma\sigma_d^2}/N$  ($\rho=0$)")
    ax.plot(monthly.index, monthly["naive_sem"], lw=0.6, color="tab:blue",
            label=r"naive, after $\sigma_d/\sqrt{N_d}$  ($\rho=0$)")
    ax.set_yscale("log")
    ax.set_ylabel(r"monthly $\sigma$")
    ax.set_title("Monthly sunspot-number uncertainty: published vs naively propagated")
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax.grid(alpha=.3)
    mark_1981(ax)

    ax = axes[1]
    ax.plot(monthly.index, monthly["ratio_disp"], lw=0.5, color="tab:red",
            label="naive(dispersion) / published")
    ax.plot(monthly.index, monthly["ratio_sem"], lw=0.5, color="tab:blue",
            label="naive(SEM) / published")
    ax.plot(monthly.index, 1 / np.sqrt(monthly["n_days"]), lw=1.0, color="k",
            ls=":", label=r"$1/\sqrt{N_{days}}$  (predicted)")
    ax.axhline(1.0, color="k", lw=0.9)
    ax.text(monthly.index[40], 1.06, "agreement", fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("ratio to published")
    ax.set_title(r"The discrepancy is the $1/\sqrt{N_{days}}$ scaling, not scatter "
                 "and not correlation")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=.3)
    mark_1981(ax)

    ax = axes[2]
    r = yearly["ratio_disp"]
    ax.plot(yearly.index, r, lw=0.8, color="tab:red", label="yearly: naive(disp)/published")
    ax.plot(yearly.index, yearly["ratio_sem"], lw=0.8, color="tab:blue",
            label="yearly: naive(SEM)/published")
    ax.plot(yearly.index, 1 / np.sqrt(yearly["n_days"]), lw=1.0, color="k", ls=":",
            label=r"$1/\sqrt{N_{days}}$  (predicted)")
    ax.axhline(1.0, color="k", lw=0.9)
    mark_1981(ax)
    ax.set_yscale("log")
    ax.set_ylabel("ratio to published")
    ax.set_xlabel("year")
    ax.set_title("Same effect under a second aggregation (daily to yearly)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=.3)

    fig.savefig(FIGURES / "quadrature_check.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
