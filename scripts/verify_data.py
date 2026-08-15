"""
Verify the structure of the SILSO files against the actual bytes.

Run:  python scripts/verify_data.py

Every claim in section 1 of the README is produced by this script, so it can be
regenerated and re-checked against a fresh download rather than trusted. Nothing
here reads the info pages; it reads the files.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from silso_uncertainty import fetch  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "output"


def rows(key: str) -> list[list[str]]:
    with open(fetch.fetch(key), newline="") as fh:
        return [r for r in csv.reader(fh, delimiter=";")]


def main() -> None:
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("Structural verification of the SILSO v2.0 files")
    emit("=" * 78)
    emit()

    data = {}
    for key in fetch.FILES:
        r = rows(key)
        data[key] = r
        prov = fetch.provenance(key)
        widths = Counter(len(x) for x in r)
        emit(f"{key}  ({fetch.FILES[key]})")
        emit(f"   rows={len(r)}  field counts={dict(widths)}  "
             f"delimiter=';'  header=none  quoting=none")
        emit(f"   bytes={prov.get('bytes')}  sha256={str(prov.get('sha256'))[:16]}...")
        emit(f"   retrieved={prov.get('retrieved_utc')}  url={prov.get('url')}")
        emit(f"   first={r[0]}")
        emit(f"   last ={r[-1]}")
        emit()

    # -- sentinel semantics ------------------------------------------------
    emit("SENTINEL (-1) SEMANTICS")
    emit("-" * 78)
    d = data["daily_total"]
    m1 = sum(1 for x in d if float(x[4]) == -1 and float(x[5]) == -1)
    m2 = sum(1 for x in d if float(x[4]) == -1 and float(x[5]) != -1)
    m3 = sum(1 for x in d if float(x[4]) != -1 and float(x[5]) == -1)
    emit(f"daily_total: value=-1 & sigma=-1: {m1};  value=-1 only: {m2};  "
         f"sigma=-1 only: {m3}")
    emit(f"daily_total: n_observations where value=-1: "
         f"{sorted({int(x[6]) for x in d if float(x[4]) == -1})} "
         "(0, not -1: the count column uses a different sentinel)")
    emit(f"daily_total: genuine zero sunspot numbers: "
         f"{sum(1 for x in d if float(x[4]) == 0)} "
         "(distinct from missing, must not be conflated)")

    m = data["monthly_total"]
    emit(f"monthly_total: value=-1: {sum(1 for x in m if float(x[3]) == -1)};  "
         f"sigma=-1 while value is valid: "
         f"{sum(1 for x in m if float(x[4]) == -1 and float(x[3]) != -1)}")
    valid = [x for x in m if float(x[4]) != -1][0]
    emit(f"   first month with a sigma: {valid[0]}-{valid[1]} "
         "(sigma exists only once per-station daily data begins)")

    h = data["daily_hemispheric"]
    emit(f"daily_hemispheric: missing values or sigmas: "
         f"{sum(1 for x in h if any(float(x[i]) == -1 for i in range(4, 10)))}")
    first_hem = [x for x in h if float(x[11]) != -1][0]
    emit(f"   hemispheric station counts are -1 on "
         f"{sum(1 for x in h if float(x[11]) == -1)} rows, all before "
         f"{first_hem[0]}-{first_hem[1]}-{first_hem[2]}, on rows whose values are valid")

    s = data["monthly_smoothed"]
    bad = [f"{x[0]}-{x[1]}" for x in s if float(x[3]) == -1]
    emit(f"monthly_smoothed: value=-1 on {len(bad)} rows: {bad} "
         "(the +/-6 month edges of the 13-month filter)")
    emit()

    # -- the standard deviation column ------------------------------------
    emit("WHAT THE SIGMA COLUMN IS")
    emit("-" * 78)
    n1 = [x for x in d if int(x[6]) == 1 and float(x[4]) != -1]
    nz = sum(1 for x in n1 if float(x[5]) != 0.0)
    emit(f"days with exactly one contributing station: {len(n1)}")
    emit(f"   of those, sigma != 0 on {nz} ({100 * nz / len(n1):.1f}%)")
    emit("   -> sigma cannot be a plain dispersion across stations on those days;")
    emit("      before 1981 the count column is fixed at 1 by convention.")
    gt1 = [x for x in d if float(x[6]) > 1]
    emit(f"first day with more than one station: "
         f"{gt1[0][0]}-{gt1[0][1]}-{gt1[0][2]}")

    num, den = defaultdict(float), defaultdict(float)
    for x in d:
        if float(x[4]) == -1:
            continue
        num[(x[0], x[1])] += int(x[6]) * float(x[5]) ** 2
        den[(x[0], x[1])] += int(x[6])
    diffs = []
    for x in m:
        k = (x[0], x[1])
        if float(x[4]) == -1 or den[k] == 0:
            continue
        diffs.append(abs(sqrt(num[k] / den[k]) - float(x[4])))
    emit(f"monthly sigma reproduced as sqrt(SUM(N_d sigma_d^2)/SUM(N_d)): "
         f"n={len(diffs)}, max|diff|={max(diffs):.4f}, "
         f"within rounding for {100 * sum(1 for e in diffs if e <= 0.05) / len(diffs):.2f}%")
    emit("   -> the published monthly sigma is a pooled station dispersion,")
    emit("      NOT a standard error of the monthly mean.")
    emit()

    # -- provisional flag --------------------------------------------------
    emit("PROVISIONAL / DEFINITIVE FLAG")
    emit("-" * 78)
    prov = [x for x in d if x[7].strip() == "0"]
    emit(f"daily_total: flag=0 on {len(prov)} rows, "
         f"{prov[0][0]}-{prov[0][1]}-{prov[0][2]} to "
         f"{prov[-1][0]}-{prov[-1][1]}-{prov[-1][2]}  (0 = provisional, 1 = definitive)")
    emit()

    # -- monthly mean relationship ----------------------------------------
    emit("RELATIONSHIP BETWEEN PRODUCTS")
    emit("-" * 78)
    vals = defaultdict(list)
    obs = Counter()
    for x in d:
        if float(x[4]) != -1:
            vals[(x[0], x[1])].append(float(x[4]))
            obs[(x[0], x[1])] += int(x[6])
    same_mean = same_obs = tot = 0
    for x in m:
        k = (x[0], x[1])
        if not vals[k]:
            continue
        tot += 1
        same_mean += abs(sum(vals[k]) / len(vals[k]) - float(x[3])) <= 0.05
        same_obs += obs[k] == int(x[5])
    emit(f"monthly mean == arithmetic mean of daily values: "
         f"{100 * same_mean / tot:.2f}% of {tot} months (to file rounding)")
    emit(f"monthly n_observations == sum of daily n_observations: "
         f"{100 * same_obs / tot:.2f}% of {tot} months")

    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "verify_data.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
