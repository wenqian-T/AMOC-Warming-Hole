from pathlib import Path
import os
_PROJECT_START = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
PROJECT_ROOT = next((p for p in [_PROJECT_START, *_PROJECT_START.parents] if (p / "PROJECT_ROOT.marker").is_file()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Run from the project folder or its notebooks/scripts subfolder.")


# ============================================================
# Figure 20: Absolute AMOC strength at 26.5°N and 45°N
# Historical + SSP5-8.5 | 1950–2100
#
# OLS trend: 2015–2100 ONLY
# Layout: 3 rows × 4 columns
#
# Black: 26.5°N
# Blue:  45°N
#
# Solid: 5-member annual mean
# Shading: mean ±1 SD
# Red dashed: OLS trend (2015–2100)
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ============================================================
# 1. Paths
# ============================================================

DATA_ROOT = Path(
    str(PROJECT_ROOT / 'data/ssp585_analysis')
)

DATA_FILE = (
    DATA_ROOT / "tables/annual_indices_1950_2100.csv"
)

OUT = Path(
    str(PROJECT_ROOT / 'results')
)

OUT.mkdir(parents=True, exist_ok=True)

ts = pd.read_csv(DATA_FILE)
ts = ts.loc[ts["region"] == "main"].copy()


# ============================================================
# 2. Settings
# ============================================================

MODELS = [
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "CanESM5",
    "CanESM5-1",
    "MPI-ESM1-2-LR",
]

LATITUDES = [
    ("amoc26", "26.5°N"),
    ("amoc45", "45°N"),
]

LAT_COLORS = {
    "amoc26": "black",
    "amoc45": "blue",
}

YEARS = np.arange(1950, 2101)

XTICKS = [1950, 2000, 2050, 2100]

TREND_START = 2015
TREND_END = 2100

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "savefig.dpi": 300,
})


# ============================================================
# 3. Calculate absolute AMOC mean and SD
# ============================================================

def absolute_amoc(model, variable):

    sub = ts.loc[
        (ts["model"] == model)
        & (ts["member"] != "ensemble")
    ]

    members = []

    for member, group in sub.groupby("member"):

        group = group.sort_values("year")

        years = group["year"].to_numpy(dtype=int)

        if not np.array_equal(years, YEARS):
            raise ValueError(
                f"Incomplete years: {model}/{member}"
            )

        values = group[variable].to_numpy(dtype=float)

        members.append(values)

    if len(members) != 5:
        raise ValueError(
            f"{model}: expected 5 members, "
            f"found {len(members)}"
        )

    a = np.stack(members)

    mean = np.mean(a, axis=0)
    sd = np.std(a, axis=0, ddof=1)

    return mean, sd


# ============================================================
# 4. OLS trend: 2015–2100
# ============================================================

def calculate_future_trend(years, values):

    mask = (
        (years >= TREND_START)
        & (years <= TREND_END)
        & np.isfinite(values)
    )

    x = years[mask]
    y = values[mask]

    if len(x) < 3:
        raise ValueError("Insufficient future data")

    slope, intercept = np.polyfit(x, y, 1)

    # Convert Sv/year to Sv/decade
    trend = slope * 10.0

    fitted = slope * x + intercept

    return trend, x, fitted


# ============================================================
# 5. Figure
# ============================================================

fig, axes = plt.subplots(
    3, 4,
    figsize=(14, 9),
    sharex=True,
    sharey=True
)

axes_flat = axes.flatten()

trend_records = []

for model_index, model in enumerate(MODELS):

    for lat_index, (variable, latitude) in enumerate(LATITUDES):

        panel_index = model_index * 2 + lat_index

        ax = axes_flat[panel_index]

        color = LAT_COLORS[variable]

        mean, sd = absolute_amoc(model, variable)

        # ----------------------------------------------------
        # 5.1 Ensemble ±1 SD
        # ----------------------------------------------------

        ax.fill_between(
            YEARS,
            mean - sd,
            mean + sd,
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=1
        )

        # ----------------------------------------------------
        # 5.2 Annual ensemble mean
        # ----------------------------------------------------

        ax.plot(
            YEARS,
            mean,
            color=color,
            linewidth=1.4,
            zorder=4
        )

        # ----------------------------------------------------
        # 5.3 Historical / SSP5-8.5 boundary
        # ----------------------------------------------------

        ax.axvline(
            2014.5,
            color="#777777",
            linestyle="--",
            linewidth=0.8,
            alpha=0.85,
            zorder=2
        )

        # ----------------------------------------------------
        # 5.4 Future trend: 2015–2100
        # ----------------------------------------------------

        trend, xfit, yfit = calculate_future_trend(
            YEARS, mean
        )

        ax.plot(
            xfit,
            yfit,
            color="#C44E52",
            linestyle="--",
            linewidth=1.2,
            zorder=5
        )

        trend_records.append({
            "model": model,
            "latitude": latitude,
            "period": "2015–2100",
            "trend_Sv_decade": trend
        })

        ax.text(
            0.04,
            0.07,
            f"Trend: {trend:+.2f} Sv/decade",
            transform=ax.transAxes,
            fontsize=8.5,
            color="red",
            ha="left",
            va="bottom",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.80,
                pad=1.5
            ),
            zorder=6
        )

        # ----------------------------------------------------
        # 5.5 Axes
        # ----------------------------------------------------

        ax.set_xlim(1950, 2100)
        ax.set_xticks(XTICKS)

        ax.set_title(
            f"({chr(97 + panel_index)}) "
            f"{model} ({latitude})",
            fontsize=10,
            pad=8
        )

        if panel_index % 4 == 0:
            ax.set_ylabel(
                "AMOC strength (Sv)",
                fontsize=10
            )

        if panel_index >= 6:
            ax.set_xlabel("Year", fontsize=10)
            ax.tick_params(axis="x", labelbottom=True)
        else:
            ax.tick_params(axis="x", labelbottom=False)

        ax.grid(
            linestyle="--",
            linewidth=0.5,
            color="#CCCCCC",
            alpha=0.65
        )

        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#555555")
            spine.set_linewidth(0.75)

        ax.tick_params(
            axis="both",
            labelsize=9,
            direction="out"
        )


# ============================================================
# 6. Legend
# ============================================================

legend_ax = axes_flat[10]
legend_ax.axis("off")

axes_flat[11].axis("off")

handles = [
    Line2D(
        [0], [0],
        color="black",
        linewidth=1.5,
        label="26.5°N annual mean"
    ),
    Line2D(
        [0], [0],
        color="#4477AA",
        linewidth=1.5,
        label="45°N annual mean"
    ),
    Patch(
        facecolor="#999999",
        alpha=0.25,
        label="Ensemble mean ±1 SD"
    ),
    Line2D(
        [0], [0],
        color="#C44E52",
        linestyle="--",
        linewidth=1.2,
        label="OLS trend (2015–2100)"
    ),
    Line2D(
        [0], [0],
        color="#777777",
        linestyle="--",
        linewidth=0.8,
        label="Historical / SSP5-8.5"
    ),
]

legend_ax.legend(
    handles=handles,
    loc="center left",
    frameon=False,
    fontsize=9.5,
    handlelength=2.6,
    labelspacing=1.2
)


# ============================================================
# 7. Title and layout
# ============================================================

fig.suptitle(
    "AMOC strength at 26.5°N and 45°N (1950–2100)",
    fontsize=14,
    x=0.53,
    y=0.965,
    ha="center"
)

fig.subplots_adjust(
    left=0.085,
    right=0.975,
    top=0.89,
    bottom=0.075,
    wspace=0.18,
    hspace=0.35
)


# ============================================================
# 8. Save figure
# ============================================================

for ext in ("png", "pdf"):

    path = OUT / f"Fig20_AMOC_absolute_trends.{ext}"

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    print("Saved:", path)

plt.show()
plt.close(fig)


# ============================================================
# 9. Save trend table
# ============================================================

trend_df = pd.DataFrame(trend_records)

trend_file = OUT / "Fig20_AMOC_trends_2015_2100.csv"

trend_df.to_csv(
    trend_file,
    index=False,
    float_format="%.4f"
)

print("\n2015–2100 AMOC trends (Sv/decade):")
print(trend_df.round(3).to_string(index=False))

print("\nSaved:", trend_file)

# ============================================================
# Figure 21: AMOC–warming hole relationship under SSP5-8.5
# Period: 2015–2100
#
# Single-panel figure
# No leave-one-out test
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, linregress
from matplotlib.lines import Line2D


# ============================================================
# 1. Paths
# ============================================================

ROOT = Path(
    str(PROJECT_ROOT / 'data/ssp585_analysis')
)

DATA_FILE = ROOT / "tables/all_metrics.csv"

OUT = Path(
    str(PROJECT_ROOT / 'results')
)

OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)
df = df.loc[df["region"] == "main"].copy()


# ============================================================
# 2. Settings
# ============================================================

MODELS = [
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "CanESM5",
    "CanESM5-1",
    "MPI-ESM1-2-LR",
]

MODEL_COLORS = {
    "ACCESS-CM2": "#4477AA",
    "ACCESS-ESM1-5": "#EE6677",
    "CanESM5": "#228833",
    "CanESM5-1": "#CCBB44",
    "MPI-ESM1-2-LR": "#AA3377",
}

XVAR = "amoc26_trend"
YVAR = "whi_trend"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "savefig.dpi": 300,
})


# ============================================================
# 3. Extract model means and standard deviations
# ============================================================

records = []

for model in MODELS:

    sub = df.loc[df["model"] == model]

    members = sub.loc[
        sub["member"] != "ensemble"
    ]

    ensemble = sub.loc[
        sub["member"] == "ensemble"
    ]

    if len(members) != 5:
        raise ValueError(
            f"{model}: expected 5 members, "
            f"found {len(members)}"
        )

    if len(ensemble) != 1:
        raise ValueError(
            f"{model}: expected 1 ensemble record"
        )

    records.append({
        "model": model,
        "x": float(ensemble.iloc[0][XVAR]),
        "y": float(ensemble.iloc[0][YVAR]),
        "x_sd": members[XVAR].std(ddof=1),
        "y_sd": members[YVAR].std(ddof=1),
    })

summary = pd.DataFrame(records)

x = summary["x"].to_numpy()
y = summary["y"].to_numpy()


# ============================================================
# 4. Pearson correlation and OLS regression
# ============================================================

r, p = pearsonr(x, y)

reg = linregress(x, y)

print(f"Pearson r = {r:.4f}")
print(f"p-value   = {p:.4f}")
print(f"n         = {len(MODELS)}")

# ============================================================
# 5. Plot
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5.5))

fig.subplots_adjust(
    left=0.15,
    right=0.96,
    top=0.88,
    bottom=0.14
)

# ------------------------------------------------------------
# 5.1 Model points with ±1 SD
# ------------------------------------------------------------

for _, row in summary.iterrows():

    model = row["model"]
    color = MODEL_COLORS[model]

    ax.errorbar(
        row["x"],
        row["y"],
        xerr=row["x_sd"],
        yerr=row["y_sd"],
        fmt="o",
        markersize=5.5,
        color=color,
        ecolor=color,
        elinewidth=0.9,
        capsize=2,
        capthick=0.9,
        alpha=0.90,
        zorder=4
    )

# ------------------------------------------------------------
# 5.2 OLS regression: gray dashed line
# ------------------------------------------------------------

xmin = np.min(x - summary["x_sd"].to_numpy())
xmax = np.max(x + summary["x_sd"].to_numpy())

padding = 0.08 * (xmax - xmin)

xx = np.linspace(
    xmin - padding,
    xmax + padding,
    200
)

ax.plot(
    xx,
    reg.intercept + reg.slope * xx,
    color="#888888",
    linestyle="--",
    linewidth=1.2,
    zorder=3
)

# ------------------------------------------------------------
# 5.3 Correlation annotation
# ------------------------------------------------------------

ax.text(
    0.04,
    0.96,
    f"r = {r:.3f}\n"
    f"p = {p:.4f}\n"
    f"n = {len(MODELS)}",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=10,
    bbox=dict(
        facecolor="white",
        edgecolor="none",
        alpha=0.85,
        pad=2
    ),
    zorder=6
)

# ------------------------------------------------------------
# 5.4 Axes and title
# ------------------------------------------------------------

ax.set_xlabel(
    "AMOC trend (Sv/decade)",
    fontsize=11
)

ax.set_ylabel(
    "Warming-hole trend (°C/decade)",
    fontsize=11
)

ax.set_title(
    "AMOC–warming hole (2015–2100)",
    fontsize=13,
    pad=13
)

# Remove all reference lines and background grid
ax.grid(False)

# Complete borders
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("#555555")
    spine.set_linewidth(0.75)

ax.tick_params(
    axis="both",
    labelsize=10,
    direction="out"
)

# ============================================================
# 6. Model legend: inside lower-right corner
# ============================================================

handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="None",
        markerfacecolor=MODEL_COLORS[m],
        markeredgecolor=MODEL_COLORS[m],
        markersize=6,
        label=m
    )
    for m in MODELS
]

ax.legend(
    handles=handles,
    title="CMIP6 models",
    loc="lower right",
    bbox_to_anchor=(0.98, 0.02),
    frameon=False,
    fontsize=9,
    title_fontsize=9,
    labelspacing=0.5,
    handletextpad=0.5
)

# ============================================================
# 7. Save
# ============================================================

for ext in ("png", "pdf"):

    path = OUT / f"Fig21_AMOC_WHI_relationship.{ext}"

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    print("Saved:", path)

plt.show()
plt.close(fig)


