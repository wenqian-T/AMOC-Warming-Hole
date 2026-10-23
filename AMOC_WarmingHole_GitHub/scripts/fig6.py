from pathlib import Path
import os
_PROJECT_START = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
PROJECT_ROOT = next((p for p in [_PROJECT_START, *_PROJECT_START.parents] if (p / "PROJECT_ROOT.marker").is_file()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Run from the project folder or its notebooks/scripts subfolder.")


# Rebuild the statistics required by the original Figure 8 plotting cell.
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, linregress
OUT = PROJECT_ROOT / "results/figures"
OUT.mkdir(parents=True, exist_ok=True)
sst = pd.read_csv(OUT / "Fig02_SST_member_trends.csv")
amoc = pd.read_csv(OUT / "AMOC_member_trends.csv")
whi = sst.loc[sst["index"].eq("Global-SPNA"), ["model", "member", "trend"]].rename(columns={"trend":"whi"})
amoc = amoc.loc[amoc["period"].eq("1950-2014"), ["model", "member", "trend_Sv_decade"]].rename(columns={"trend_Sv_decade":"amoc"})
paired = amoc.merge(whi, on=["model", "member"], validate="one_to_one")
if len(paired) != 60 or not paired.groupby("model").size().eq(5).all():
    raise ValueError("Expected 12 models with 5 matched members each")
MODELS_ORDER = ["ACCESS-CM2", "ACCESS-ESM1-5", "CESM2", "CanESM5", "CanESM5-1", "ICON-ESM-LR", "MIROC6", "MPI-ESM1-2-HR", "MPI-ESM1-2-LR", "MRI-ESM2-0", "FGOALS-g3", "NorCPM1"]
model_stats = paired.groupby("model").agg(amoc_mean=("amoc","mean"), amoc_sd=("amoc","std"), whi_mean=("whi","mean"), whi_sd=("whi","std"), n_members=("member","size")).loc[MODELS_ORDER].reset_index()
x = model_stats.amoc_mean.to_numpy()
y = model_stats.whi_mean.to_numpy()
fit = linregress(x, y)
r_full, p_full = pearsonr(x, y)
loo_rows = []
for model in MODELS_ORDER:
    sub = model_stats.loc[model_stats.model.ne(model)]
    rr, pp = pearsonr(sub.amoc_mean, sub.whi_mean)
    loo_rows.append({"excluded_model":model,"n_models":len(sub),"pearson_r":rr,"pvalue":pp})
loo_df = pd.DataFrame(loo_rows)
model_stats.to_csv(OUT / "Fig08_model_statistics.csv", index=False)
loo_df.to_csv(OUT / "Fig08_leave_one_out.csv", index=False)



# ============================================================
# 7. Plot Figure 8
#    Different colors for 12 CMIP6 models
#    Shared model legend on the right
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ------------------------------------------------------------
# 1. Model colors
# ------------------------------------------------------------

# 使用 tab20 中区分度较高的12种颜色
cmap = plt.get_cmap("tab20")

color_indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3]

MODEL_COLORS = {
    model: cmap(i)
    for model, i in zip(MODELS_ORDER, color_indices)
}

COLOR_REGRESSION = "black"
COLOR_REFERENCE = "red"
COLOR_ZERO = "#888888"


# ------------------------------------------------------------
# 2. Create figure
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1, 2,
    figsize=(15, 6)
)

ax1, ax2 = axes


# ============================================================
# (a) Between-model relationship
# ============================================================

for _, row in model_stats.iterrows():

    model = row["model"]
    color = MODEL_COLORS[model]

    ax1.errorbar(
        row["amoc_mean"],
        row["whi_mean"],
        xerr=row["amoc_sd"],
        yerr=row["whi_sd"],
        fmt="o",
        markersize=7,
        color=color,
        markeredgecolor="white",
        markeredgewidth=0.5,
        ecolor=color,
        elinewidth=1.0,
        capsize=2,
        alpha=0.9,
        zorder=3
    )


# OLS regression line
xx = np.linspace(
    x.min() - 0.03,
    x.max() + 0.03,
    200
)

ax1.plot(
    xx,
    fit.intercept + fit.slope * xx,
    color=COLOR_REGRESSION,
    linewidth=1.6,
    zorder=2
)

# Zero reference lines
ax1.axhline(
    0,
    color=COLOR_ZERO,
    linestyle="--",
    linewidth=0.8
)

ax1.axvline(
    0,
    color=COLOR_ZERO,
    linestyle="--",
    linewidth=0.8
)

# Correlation statistics
ax1.text(
    0.04, 0.96,
    f"r = {r_full:.3f}\n"
    f"p = {p_full:.4f}\n"
    f"n = 12",
    transform=ax1.transAxes,
    va="top",
    ha="left",
    fontsize=10
)

ax1.set_title(
    "(a) Cross-model relationship",
    fontsize=12,
    fontweight="bold"
)

ax1.set_xlabel(
    "AMOC trend (Sv/decade)"
)

ax1.set_ylabel(
    "Warming-hole intensity (°C/decade)"
)


# ============================================================
# (b) Leave-one-out sensitivity
# ============================================================

positions = np.arange(len(loo_df))

for i, row in loo_df.iterrows():

    model = row["excluded_model"]
    color = MODEL_COLORS[model]

    ax2.scatter(
        row["pearson_r"],
        i,
        s=55,
        color=color,
        edgecolor="white",
        linewidth=0.5,
        zorder=3
    )

# Full 12-model reference correlation
ax2.axvline(
    r_full,
    color=COLOR_REFERENCE,
    linewidth=1.6,
    linestyle="--",
    label=f"All 12 models (r = {r_full:.3f})",
    zorder=2
)

# 保留12个模式各自的纵向位置，但隐藏纵坐标
ax2.set_yticks([])
ax2.set_ylim(len(loo_df) - 0.5, -0.5)
ax2.tick_params(axis="y", left=False, labelleft=False)

ax2.invert_yaxis()

ax2.set_xlabel(
    "Pearson correlation (r)"
)

ax2.set_title(
    "(b) Leave-one-out tests",
    fontsize=12,
    fontweight="bold"
)

ax2.legend(
    loc="upper left",
    fontsize=9,
    frameon=False
)

margin = 0.01

ax2.set_xlim(
    min(loo_df["pearson_r"].min(), r_full) - margin,
    max(loo_df["pearson_r"].max(), r_full) + margin
)


# ============================================================
# 3. Common styling
# ============================================================

for ax in axes:

    ax.grid(
        axis="both",
        color="#DDDDDD",
        linestyle="--",
        linewidth=0.5,
        alpha=0.7
    )

    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#555555")
        spine.set_linewidth(0.7)


# ============================================================
# 4. Shared model legend on the far right
# ============================================================

model_handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="None",
        markerfacecolor=MODEL_COLORS[model],
        markeredgecolor="white",
        markeredgewidth=0.5,
        markersize=8,
        label=model
    )
    for model in MODELS_ORDER
]

fig.legend(
    handles=model_handles,
    loc="center left",
    bbox_to_anchor=(0.83, 0.51),
    fontsize=9,
    frameon=False,
    ncol=1,
    handletextpad=0.5,
    labelspacing=0.8,
    title="CMIP6 models",
    title_fontsize=10
)


# ============================================================
# 5. Title and layout
# ============================================================

fig.suptitle(
    "AMOC–warming hole relationship and sensitivity",
    fontsize=18,
    x=0.44,
    y=0.98
)

fig.subplots_adjust(
    left=0.08,
    right=0.79,
    top=0.88,
    bottom=0.13,
    wspace=0.15   # 原来0.48，缩小两图间距
)

# ============================================================
# 6. Save
# ============================================================

for ext in ("png", "pdf"):

    fig.savefig(
        OUT / f"Fig08_AMOC_WHI_relationship.{ext}",
        dpi=300,
        bbox_inches="tight"
    )

plt.show()
plt.close(fig)
