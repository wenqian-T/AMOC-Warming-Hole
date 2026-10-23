from pathlib import Path
import os
_PROJECT_START = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
PROJECT_ROOT = next((p for p in [_PROJECT_START, *_PROJECT_START.parents] if (p / "PROJECT_ROOT.marker").is_file()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Run from the project folder or its notebooks/scripts subfolder.")



# ============================================================
# CMIP6 Historical AMOC | Figure 5, RAPID, Figure 7
#
# Input:
#   historical_analysis/annual/{model}_{member}_annual.nc
#   OBS/RAPID/RAPID_transports_v2024-2.nc
#
# Output:
#   Fig05_AMOC_12models.png / .pdf
#   Fig_RAPID_AMOC.png / .pdf
#   Fig07_AMOC_trends.png / .pdf
#   AMOC_member_annual.csv
#   AMOC_member_trends.csv
#   RAPID_annual.csv
# ============================================================

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.stats import linregress


# ============================================================
# 1. CONFIG
# ============================================================

ROOT = Path(str(PROJECT_ROOT / 'data'))

ANNUAL_DIR = ROOT / "historical_analysis/annual"

RAPID_FILE = (
    ROOT / "OBS/RAPID/RAPID_transports_v2024-2.nc"
)

OUT = Path(
    str(PROJECT_ROOT / 'results/figures')
)
OUT.mkdir(parents=True, exist_ok=True)

MODELS = [
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "CESM2",
    "CanESM5",
    "CanESM5-1",
    "FGOALS-g3",
    "ICON-ESM-LR",
    "MIROC6",
    "MPI-ESM1-2-HR",
    "MPI-ESM1-2-LR",
    "MRI-ESM2-0",
    "NorCPM1",
]

HIST_START = 1950
HIST_END = 2014

RAPID_START = 2005
RAPID_END = 2023

COMMON_START = 2005
COMMON_END = 2014

SMOOTH_WINDOW = 11
DPI = 300

# Figure colors
COLOR_MEAN = "#245B8F"
COLOR_SMOOTH = "#D47A23"
COLOR_TREND = "#C34B4B"
COLOR_SHADE = "#A9C5DC"
COLOR_RAPID = "#245B8F"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": DPI,
})


# ============================================================
# 2. AUTOMATICALLY FIND FIVE MEMBERS
# ============================================================

def member_key(member):
    match = re.fullmatch(
        r"r(\d+)i(\d+)p(\d+)f(\d+)", member
    )
    return tuple(map(int, match.groups()))


MEMBER_FILES = {}

for model in MODELS:
    found = {}

    for path in ANNUAL_DIR.glob(
        f"{model}_*_annual.nc"
    ):
        if path.name.startswith("._"):
            continue

        member = path.name[
            len(model) + 1:-len("_annual.nc")
        ]

        if re.fullmatch(
            r"r\d+i\d+p\d+f\d+", member
        ):
            found[member] = path

    if len(found) != 5:
        raise ValueError(
            f"{model}: expected 5 members, "
            f"found {len(found)}: {list(found)}"
        )

    MEMBER_FILES[model] = {
        m: found[m]
        for m in sorted(found, key=member_key)
    }


# ============================================================
# 3. READ ANNUAL AMOC INDEX
# ============================================================

def read_member_amoc(path):
    """
    Read a precomputed 1D annual AMOC index.
    No need to access raw msftmz.

    Supports usual variable names and either year/time
    coordinates. AMOC values must already be in Sv
    or have identifiable kg/s units.
    """

    candidate_names = (
        "amoc_annual",
        "amoc_26p5",
        "amoc_26_5",
        "amoc26",
        "amoc",
        "AMOC",
        "AMOC26",
        "amoc_strength",
        "amoc_26N",
    )

    with xr.open_dataset(path, decode_times=True) as ds:

        name = next(
            (n for n in candidate_names if n in ds),
            None
        )

        if name is None:
            possible = [
                n for n in ds.data_vars
                if "amoc" in n.lower()
            ]

            if len(possible) == 1:
                name = possible[0]
            else:
                raise KeyError(
                    f"{path.name}: cannot identify annual AMOC. "
                    f"Variables: {list(ds.data_vars)}"
                )

        da = ds[name].squeeze(drop=True)

        if "year" in da.dims:
            years = np.asarray(
                da["year"].values, dtype=int
            )
            tdim = "year"

        elif "time" in da.dims:
            years = np.asarray(
                da["time"].dt.year.values, dtype=int
            )
            tdim = "time"

        else:
            raise ValueError(
                f"{path.name}: AMOC has no year/time dimension: "
                f"{da.dims}"
            )

        if da.ndim != 1:
            raise ValueError(
                f"{path.name}: expected 1D annual AMOC, "
                f"got {da.dims} {da.shape}"
            )

        values = np.asarray(da.values, dtype=float)

        units = str(
            da.attrs.get("units", "")
        ).lower().replace(" ", "")

        # Only convert when units explicitly indicate kg/s.
        if (
            "kg" in units
            and (
                "s-1" in units
                or "s^-1" in units
                or "/s" in units
            )
        ):
            values = values / (1025.0 * 1e6)

        selected = (
            (years >= HIST_START)
            & (years <= HIST_END)
        )

        years = years[selected]
        values = values[selected]

        expected = np.arange(
            HIST_START, HIST_END + 1
        )

        if not np.array_equal(years, expected):
            raise ValueError(
                f"{path.name}: incomplete 1950–2014 coverage"
            )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{path.name}: non-finite AMOC values"
            )

        return years, values


# ============================================================
# 4. LOAD MEMBER ANNUAL SERIES AND COMPUTE TRENDS
# ============================================================

annual_rows = []
trend_rows = []
model_series = {}

for model in MODELS:

    years_list = []
    values_list = []

    print(f"Reading {model}", flush=True)

    for member, path in MEMBER_FILES[model].items():

        years, values = read_member_amoc(path)

        years_list.append(years)
        values_list.append(values)

        for year, value in zip(years, values):
            annual_rows.append({
                "model": model,
                "member": member,
                "year": int(year),
                "amoc_Sv": float(value),
            })

        for start, end in (
            (1950, 2014),
            (2005, 2014),
        ):
            selected = (
                (years >= start)
                & (years <= end)
            )

            fit = linregress(
                years[selected],
                values[selected]
            )

            trend_rows.append({
                "model": model,
                "member": member,
                "period": f"{start}-{end}",
                "trend_Sv_decade": fit.slope * 10,
                "pvalue": fit.pvalue,
            })

    arr = np.stack(values_list, axis=0)

    model_series[model] = {
        "year": years,
        "members": arr,
        "mean": arr.mean(axis=0),
        "sd": arr.std(axis=0, ddof=1),
    }

annual_df = pd.DataFrame(annual_rows)
trend_df = pd.DataFrame(trend_rows)

annual_df.to_csv(
    OUT / "AMOC_member_annual.csv",
    index=False
)
trend_df.to_csv(
    OUT / "AMOC_member_trends.csv",
    index=False
)


# ============================================================
# 5. FIGURE 5 — 12 MODELS (3 ROWS × 4 COLUMNS)
# ============================================================

fig, axes = plt.subplots(
    3, 4,
    figsize=(17, 10),
    sharex=True,
    sharey=True,
)

for i, model in enumerate(MODELS):

    ax = axes.flat[i]
    data = model_series[model]

    years = data["year"]
    mean = data["mean"]
    sd = data["sd"]

    # Member spread
    shade = ax.fill_between(
        years,
        mean - sd,
        mean + sd,
        color=COLOR_SHADE,
        alpha=0.45,
        linewidth=0,
        label="Member ±1 SD",
        zorder=1
    )

    # Annual ensemble mean
    line_mean, = ax.plot(
        years,
        mean,
        color=COLOR_MEAN,
        linewidth=1.0,
        alpha=0.75,
        label="Ensemble mean",
        zorder=3
    )

    # Centered 11-year running mean
    smooth = pd.Series(mean).rolling(
        SMOOTH_WINDOW,
        center=True,
        min_periods=SMOOTH_WINDOW
    ).mean().to_numpy()

    line_smooth, = ax.plot(
        years,
        smooth,
        color=COLOR_SMOOTH,
        linewidth=2.0,
        label="11-year running mean",
        zorder=4
    )

    # Linear trend in 1950–2014
    fit = linregress(years, mean)

    line_trend, = ax.plot(
        years,
        fit.intercept + fit.slope * years,
        color=COLOR_TREND,
        linewidth=1.3,
        linestyle="--",
        label="Linear trend",
        zorder=5
    )

    ax.set_title(
        f"({chr(97+i)}) {model}",
        loc="left",
        fontsize=10,
        fontweight="bold",
    )

    ax.set_xlim(1950, 2014)
    ax.set_xticks(
        np.arange(1950, 2011, 10)
    )

    # Do not draw a special 1980 reference line.
    ax.grid(
        axis="y",
        color="#DDDDDD",
        linewidth=0.5
    )

    if i % 4 == 0:
        ax.set_ylabel("AMOC strength (Sv)")

    if i >= 8:
        ax.set_xlabel("Year")

    # Trend magnitude shown inside each panel
    ax.text(
        0.03, 0.06,
        f"{fit.slope*10:+.2f} Sv/decade",
        transform=ax.transAxes,
        fontsize=8,
        color=COLOR_TREND
    )

# Shared legend outside the 12 panels
handles = [
    line_mean, shade, line_smooth, line_trend
]

labels = [
    "Annual ensemble mean",
    "Member spread (±1 SD)",
    "11-year running mean",
    "OLS linear trend",
]

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.025),
    ncol=4,
    frameon=False,
    fontsize=9
)

fig.suptitle(
    "Historical AMOC strength (1950–2014)",
    fontsize=15,
    y=0.985
)

fig.subplots_adjust(
    left=0.065,
    right=0.985,
    top=0.93,
    bottom=0.13,
    wspace=0.14,
    hspace=0.26
)

for ext in ("png", "pdf"):
    fig.savefig(
        OUT / f"Fig05_AMOC_12models.{ext}",
        dpi=DPI,
        bbox_inches="tight"
    )

plt.show()
plt.close(fig)

print("Figure 5 saved.", flush=True)


# ============================================================
# 6. RAPID ANNUAL AMOC
# ============================================================


def read_rapid_annual(path):
    """
    Read RAPID AMOC observations and calculate annual means.

    Time dimension: time_1m
    Variable: amoc_depth_1m
    Period: 2005–2023

    First aggregate to monthly means, then calculate
    day-weighted annual means.
    """

    with xr.open_dataset(path, decode_times=True) as ds:

        da = ds["amoc_depth_1m"].squeeze(drop=True)

        # RAPID 使用 time_1m，统一改名为 time
        if "time_1m" in da.dims:
            da = da.rename({"time_1m": "time"})

        elif "time" not in da.dims:
            raise ValueError(
                f"Unknown RAPID time dimension: {da.dims}"
            )

        # 按时间排序
        da = da.sortby("time")

        # 先计算逐月平均
        monthly = da.resample(time="MS").mean(
            skipna=True
        )

        years = []
        annual_values = []

        for year in range(RAPID_START, RAPID_END + 1):

            one_year = monthly.sel(
                time=slice(
                    f"{year}-01-01",
                    f"{year}-12-31"
                )
            )

            # 要求每年具有12个月
            if one_year.sizes["time"] != 12:
                warnings.warn(
                    f"RAPID {year}: incomplete year, skipped"
                )
                continue

            values = np.asarray(
                one_year.values,
                dtype=float
            )

            if not np.isfinite(values).all():
                warnings.warn(
                    f"RAPID {year}: missing monthly values"
                )
                continue

            # 按每个月的实际天数加权
            days = np.asarray(
                one_year.time.dt.days_in_month.values,
                dtype=float
            )

            annual_mean = np.average(
                values,
                weights=days
            )

            years.append(year)
            annual_values.append(annual_mean)

    return (
        np.asarray(years, dtype=int),
        np.asarray(annual_values, dtype=float)
    )



rapid_years, rapid_amoc = read_rapid_annual(
    RAPID_FILE
)

rapid_df = pd.DataFrame({
    "year": rapid_years,
    "amoc_Sv": rapid_amoc
})

rapid_df.to_csv(
    OUT / "RAPID_annual.csv",
    index=False
)

common = (
    (rapid_years >= COMMON_START)
    & (rapid_years <= COMMON_END)
)

if common.sum() != 10:
    raise ValueError(
        "RAPID 2005–2014 does not contain 10 valid annual means"
    )

rapid_common_mean = rapid_amoc[common].mean()


# ============================================================
# 7. RAPID INDEPENDENT FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 4.7)
)

ax.plot(
    rapid_years,
    rapid_amoc,
    "-o",
    color=COLOR_RAPID,
    linewidth=1.7,
    markersize=3.5,
    label="RAPID annual mean"
)

ax.axhline(
    rapid_common_mean,
    color=COLOR_TREND,
    linewidth=1.4,
    linestyle="--",
    label=(
        f"2005–2014 mean: "
        f"{rapid_common_mean:.2f} Sv"
    )
)

ax.set_xlim(2004.5, 2023.5)
ax.set_xticks(
    [2005, 2010, 2015, 2020, 2023]
)

ax.set_xlabel("Year")
ax.set_ylabel("AMOC strength (Sv)")

ax.set_title(
    "RAPID-observed AMOC strength (2005–2023)",
    fontsize=13
)

ax.grid(
    axis="y",
    alpha=0.3
)

ax.legend(
    loc="upper right",
    fontsize=9,
    frameon=False
)

fig.tight_layout()

for ext in ("png", "pdf"):
    fig.savefig(
        OUT / f"Fig_RAPID_AMOC.{ext}",
        dpi=DPI,
        bbox_inches="tight"
    )

plt.show()
plt.close(fig)

print("RAPID figure saved.", flush=True)



# ============================================================
# Figure 7 — AMOC trend comparison
# (a) 1950–2014
# (b) 2005–2014, including RAPID observations
# ============================================================

# 1. 计算 RAPID 2005–2014 年线性趋势
rapid_mask = (
    (rapid_years >= 2005) &
    (rapid_years <= 2014)
)

rapid_fit = linregress(
    rapid_years[rapid_mask],
    rapid_amoc[rapid_mask]
)

rapid_trend = rapid_fit.slope * 10  # Sv/decade

print(f"RAPID AMOC trend (2005–2014): {rapid_trend:+.3f} Sv/decade")
print(f"RAPID trend p-value: {rapid_fit.pvalue:.4f}")


# 2. 绘制 Figure 7
fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 7.8),
    sharey=True
)

periods = [
    ("1950-2014", "(a) 1950–2014"),
    ("2005-2014", "(b) 2005–2014")
]

for ax, (period, title) in zip(axes, periods):

    for j, model in enumerate(MODELS):

        vals = trend_df.loc[
            (trend_df["model"] == model) &
            (trend_df["period"] == period),
            "trend_Sv_decade"
        ].to_numpy(dtype=float)

        if len(vals) != 5:
            raise ValueError(
                f"{model}, {period}: expected 5 trends"
            )

        mean = vals.mean()
        sd = vals.std(ddof=1)

        # 五个初始条件成员
        ax.scatter(
            vals,
            j + np.linspace(-0.15, 0.15, 5),
            s=20,
            color="#999999",
            alpha=0.75,
            zorder=3
        )

        # 模式集合平均 ±1 SD
        ax.errorbar(
            mean,
            j,
            xerr=sd,
            fmt="o",
            markersize=6,
            color="#26384C",
            ecolor="#26384C",
            elinewidth=1.1,
            capsize=3,
            zorder=5
        )

    # 零趋势参考线
    ax.axvline(
        0,
        color="#666666",
        linewidth=0.9,
        linestyle="--",
        zorder=1
    )

    # 仅在 (b) 中加入 RAPID
    if period == "2005-2014":
        ax.axvline(
            rapid_trend,
            color="#D1495B",
            linewidth=1.8,
            linestyle="--",
            label=f"RAPID ({rapid_trend:+.2f} Sv/decade)",
            zorder=2
        )

        ax.legend(
            loc="upper right",
            fontsize=9,
            frameon=False
        )

    ax.set_title(
        title,
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xlabel("AMOC trend (Sv/decade)")

    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.invert_yaxis()

    ax.grid(
        axis="x",
        alpha=0.25,
        linewidth=0.6
    )

fig.suptitle(
    "AMOC trend",
    fontsize=15,
    y=0.99
)

fig.subplots_adjust(
    left=0.20,
    right=0.98,
    top=0.91,
    bottom=0.10,
    wspace=0.10
)

for ext in ("png", "pdf"):
    fig.savefig(
        OUT / f"Fig07_AMOC_trends.{ext}",
        dpi=DPI,
        bbox_inches="tight"
    )

plt.show()
plt.close(fig)


# ============================================================
# 6. RAPID ANNUAL AMOC
# ============================================================


def read_rapid_annual(path):
    """
    Read RAPID AMOC observations and calculate annual means.

    Time dimension: time_1m
    Variable: amoc_depth_1m
    Period: 2005–2023

    First aggregate to monthly means, then calculate
    day-weighted annual means.
    """

    with xr.open_dataset(path, decode_times=True) as ds:

        da = ds["amoc_depth_1m"].squeeze(drop=True)

        # RAPID 使用 time_1m，统一改名为 time
        if "time_1m" in da.dims:
            da = da.rename({"time_1m": "time"})

        elif "time" not in da.dims:
            raise ValueError(
                f"Unknown RAPID time dimension: {da.dims}"
            )

        # 按时间排序
        da = da.sortby("time")

        # 先计算逐月平均
        monthly = da.resample(time="MS").mean(
            skipna=True
        )

        years = []
        annual_values = []

        for year in range(RAPID_START, RAPID_END + 1):

            one_year = monthly.sel(
                time=slice(
                    f"{year}-01-01",
                    f"{year}-12-31"
                )
            )

            # 要求每年具有12个月
            if one_year.sizes["time"] != 12:
                warnings.warn(
                    f"RAPID {year}: incomplete year, skipped"
                )
                continue

            values = np.asarray(
                one_year.values,
                dtype=float
            )

            if not np.isfinite(values).all():
                warnings.warn(
                    f"RAPID {year}: missing monthly values"
                )
                continue

            # 按每个月的实际天数加权
            days = np.asarray(
                one_year.time.dt.days_in_month.values,
                dtype=float
            )

            annual_mean = np.average(
                values,
                weights=days
            )

            years.append(year)
            annual_values.append(annual_mean)

    return (
        np.asarray(years, dtype=int),
        np.asarray(annual_values, dtype=float)
    )



rapid_years, rapid_amoc = read_rapid_annual(
    RAPID_FILE
)

rapid_df = pd.DataFrame({
    "year": rapid_years,
    "amoc_Sv": rapid_amoc
})

rapid_df.to_csv(
    OUT / "RAPID_annual.csv",
    index=False
)

common = (
    (rapid_years >= COMMON_START)
    & (rapid_years <= COMMON_END)
)

if common.sum() != 10:
    raise ValueError(
        "RAPID 2005–2014 does not contain 10 valid annual means"
    )

rapid_common_mean = rapid_amoc[common].mean()







# ============================================================
# 5. FIGURE 5 — 12 MODELS (3 × 4)
# ============================================================

# 调整模式排列顺序
PLOT_MODELS = [
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "CESM2",
    "CanESM5",
    "CanESM5-1",
    "ICON-ESM-LR",
    "MIROC6",
    "MPI-ESM1-2-HR",
    "MPI-ESM1-2-LR",
    "MRI-ESM2-0",
    "FGOALS-g3",
    "NorCPM1",
]

# 纵坐标范围
YLIM_COMMON = (10, 24)
YLIM_FGOALS = (24, 35)
YLIM_NORCPM = (29, 36)

# 统一配色
COLOR_MEAN = "black"     # 黑色：年平均集合序列
COLOR_SMOOTH = "blue"   # 蓝色：11年滑动平均
COLOR_TREND = "red"    # 红色：线性趋势
COLOR_SHADE = "#BDBDBD"    # 灰色：成员离散度

fig, axes = plt.subplots(
    3, 4,
    figsize=(17, 10),
    sharex=True,
    sharey=False  # 允许各模式设置独立纵轴
)

for i, model in enumerate(PLOT_MODELS):

    ax = axes.flat[i]
    data = model_series[model]

    years = data["year"]
    mean = data["mean"]
    sd = data["sd"]

    # 1. 五成员 ±1 SD
    shade = ax.fill_between(
        years,
        mean - sd,
        mean + sd,
        color=COLOR_SHADE,
        alpha=0.45,
        linewidth=0,
        label="Member spread (±1 SD)",
        zorder=1
    )

    # 2. 年平均集合序列：黑色
    line_mean, = ax.plot(
        years,
        mean,
        color=COLOR_MEAN,
        linewidth=1.0,
        alpha=0.85,
        label="Annual ensemble mean",
        zorder=3
    )

    # 3. 11年滑动平均：蓝色
    smooth = pd.Series(mean).rolling(
        window=11,
        center=True,
        min_periods=11
    ).mean().to_numpy()

    line_smooth, = ax.plot(
        years,
        smooth,
        color=COLOR_SMOOTH,
        linewidth=2.0,
        label="11-year running mean",
        zorder=4
    )

    # 4. OLS线性趋势：红色虚线
    fit = linregress(years, mean)

    line_trend, = ax.plot(
        years,
        fit.intercept + fit.slope * years,
        color=COLOR_TREND,
        linewidth=1.3,
        linestyle="--",
        label="OLS linear trend",
        zorder=5
    )

    # 5. 设置纵坐标范围
    if model in ["FGOALS-g3", "NorCPM1"]:
        ax.set_ylim(24, 38)
        ax.set_yticks(np.arange(24, 39, 2))

    else:
        ax.set_ylim(10, 24)
        ax.set_yticks(np.arange(10, 25, 2))

    # 6. 面板标题
    ax.set_title(
        f"({chr(97+i)}) {model}",
        loc="left",
        fontsize=10,
        fontweight="bold"
    )

    # 7. 横坐标
    ax.set_xlim(1950, 2014)
    ax.set_xticks(np.arange(1950, 2011, 10))

    if i >= 8:
        ax.set_xlabel("Year")

    if i % 4 == 0:
        ax.set_ylabel("AMOC strength (Sv)")

    # 8. 趋势数值
    ax.text(
        0.03, 0.06,
        f"{fit.slope * 10:+.2f} Sv/decade",
        transform=ax.transAxes,
        fontsize=8,
        color=COLOR_TREND
    )
    # 10. 显示完整四边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#555555")
        spine.set_linewidth(0.7)

    ax.set_axisbelow(True)

# 统一图例
fig.legend(
    [line_mean, shade, line_smooth, line_trend],
    [
        "Annual ensemble mean",
        "Member spread (±1 SD)",
        "11-year running mean",
        "Linear trend",
    ],
    loc="lower center",
    bbox_to_anchor=(0.5, 0.025),
    ncol=4,
    frameon=False,
    fontsize=9
)

fig.suptitle(
    "CMIP6 AMOC strength and trends (1950–2014)",
    fontsize=20,
    y=0.985
)

fig.subplots_adjust(
    left=0.065,
    right=0.985,
    top=0.93,
    bottom=0.13,
    wspace=0.14,
    hspace=0.26
)

for ext in ("png", "pdf"):
    fig.savefig(
        OUT / f"Fig05_AMOC_12models.{ext}",
        dpi=300,
        bbox_inches="tight"
    )

plt.show()
plt.close(fig)



# ============================================================
# Figure 7 — Comparison of AMOC Trends Across CMIP6 Models
# (a) 1950–2014
# (b) 2005–2014 + RAPID observation
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLOR_POS = "red"    # 红色：AMOC增强
COLOR_NEG = "blue"    # 蓝色：AMOC减弱
COLOR_ZERO = "#333333"   # 黑色：零趋势参考线
COLOR_RAPID = "cyan"  # 深灰色：RAPID观测

fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 7.8),
    sharey=True
)

periods = [
    ("1950-2014", "(a) 1950–2014"),
    ("2005-2014", "(b) 2005–2014")
]

for ax, (period, title) in zip(axes, periods):

    for j, model in enumerate(MODELS):

        vals = trend_df.loc[
            (trend_df["model"] == model) &
            (trend_df["period"] == period),
            "trend_Sv_decade"
        ].to_numpy(dtype=float)

        if len(vals) != 5:
            raise ValueError(
                f"{model}, {period}: expected 5 trends"
            )

        mean = vals.mean()
        sd = vals.std(ddof=1)

        # 1. 五个初始条件成员：根据正负趋势着色
        member_colors = [
            COLOR_POS if v > 0 else COLOR_NEG
            for v in vals
        ]

        ax.scatter(
            vals,
            j + np.linspace(-0.15, 0.15, 5),
            s=20,
            c=member_colors,
            alpha=0.55,
            linewidths=0,
            zorder=3
        )

        # 2. 模式集合平均及 ±1 SD
        mean_color = (
            COLOR_POS if mean > 0 else COLOR_NEG
        )

        ax.errorbar(
            mean,
            j,
            xerr=sd,
            fmt="o",
            markersize=6,
            color=mean_color,
            ecolor=mean_color,
            elinewidth=1.2,
            capsize=3,
            zorder=5
        )

    # 3. 零趋势参考线
    ax.axvline(
        0,
        color=COLOR_ZERO,
        linewidth=1.0,
        linestyle="--",
        zorder=1
    )

    # 4. 仅在(b)中加入RAPID观测趋势
    if period == "2005-2014":

        ax.axvline(
            rapid_trend,
            color=COLOR_RAPID,
            linewidth=1.8,
            linestyle="--",
            zorder=2
        )

    # 5. 标题及坐标轴
    ax.set_title(
        title,
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xlabel(
        "AMOC trend (Sv/decade)",
        fontsize=11
    )

    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.invert_yaxis()

    # 6. 浅灰色竖直虚线网格
    ax.grid(
        axis="x",
        color="#DDDDDD",
        linestyle="--",
        linewidth=0.6,
        alpha=0.7
    )

    ax.set_axisbelow(True)

    # 7. 完整四边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#555555")
        spine.set_linewidth(0.7)


# 8. 自定义图例
legend_handles = [
    Line2D(
        [0], [0],
        marker="o",
        color=COLOR_POS,
        linestyle="None",
        markersize=6,
        label="Positive trend"
    ),
    Line2D(
        [0], [0],
        marker="o",
        color=COLOR_NEG,
        linestyle="None",
        markersize=6,
        label="Negative trend"
    ),
    Line2D(
        [0], [0],
        color=COLOR_RAPID,
        linestyle="--",
        linewidth=1.8,
        label=f"RAPID ({rapid_trend:+.2f} Sv/decade)"
    )
]

axes[1].legend(
    handles=legend_handles,
    loc="upper left",
    fontsize=10,
    frameon=False
)

# 9. 总标题：居中于两个子图的绘图区
fig.suptitle(
    "AMOC trends across CMIP6 models",
    fontsize=15,
    x=0.59,
    y=0.99,
    ha="center"
)

fig.subplots_adjust(
    left=0.20,
    right=0.98,
    top=0.91,
    bottom=0.10,
    wspace=0.10
)

# 10. 保存
for ext in ("png", "pdf"):
    fig.savefig(
        OUT / f"Fig07_AMOC_trends.{ext}",
        dpi=DPI,
        bbox_inches="tight"
    )

plt.show()
plt.close(fig)


