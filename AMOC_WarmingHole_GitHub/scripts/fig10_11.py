from pathlib import Path
import os
_PROJECT_START = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
PROJECT_ROOT = next((p for p in [_PROJECT_START, *_PROJECT_START.parents] if (p / "PROJECT_ROOT.marker").is_file()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Run from the project folder or its notebooks/scripts subfolder.")


"""Figures 14–16: SSP5-8.5 SST trends and late-century changes.

Reads the three precomputed inputs only; no monthly NetCDF processing.
Run: python plot_fig14_16_ssp585.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle

ROOT = Path(str(PROJECT_ROOT / 'data/ssp585_analysis'))
MAP_FILE = ROOT / 'annual/maps_2deg.nc'
METRICS_FILE = ROOT / 'tables/all_metrics.csv'
ANNUAL_FILE = ROOT / 'tables/annual_indices_1950_2100.csv'
OUT = Path(str(PROJECT_ROOT / 'results/figures'))
OUT.mkdir(parents=True, exist_ok=True)

# Prefer same sequence as historical figures, limited to five available models.
PREFERRED_ORDER = [
    'ACCESS-CM2', 'ACCESS-ESM1-5', 'CESM2', 'CanESM5',
    'CanESM5-1', 'ICON-ESM-LR', 'MIROC6', 'MPI-ESM1-2-HR',
    'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'FGOALS-g3', 'NorCPM1',
]
RED, BLUE, DARK, GRAY = '#C0392B', '#2166AC', '#273746', '#969696'
DPI = 300
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 12, 'axes.labelsize': 10,
    'axes.spines.top': True, 'axes.spines.right': True,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

for p in (MAP_FILE, METRICS_FILE, ANNUAL_FILE):
    if not p.exists():
        raise FileNotFoundError(f'Input does not exist: {p}')

metrics = pd.read_csv(METRICS_FILE)
annual = pd.read_csv(ANNUAL_FILE)
metrics = metrics.loc[metrics['region'].eq('main')].copy()
annual = annual.loc[annual['region'].eq('main')].copy()
with xr.open_dataset(MAP_FILE) as ds:
    maps = ds.load()

available_models = set(metrics['model'].unique())
models = [m for m in PREFERRED_ORDER if m in available_models]
models += sorted(available_models - set(models))
if len(models) != 5:
    raise ValueError(f'Expected five future models, found {models}')
map_models = set(maps['model'].values.astype(str))
if set(models) != map_models:
    raise ValueError(f'Map models {map_models} differ from metric models {set(models)}')

# Confirm full 25-member set and precomputed indices; do not silently omit members.
for model in models:
    block = metrics.loc[metrics.model.eq(model)]
    member_rows = block.loc[block.member.ne('ensemble')]
    ensemble_rows = block.loc[block.member.eq('ensemble')]
    if len(member_rows) != 5 or len(ensemble_rows) != 1:
        raise ValueError(f'{model}: expected five members + ensemble, got {len(member_rows)} + {len(ensemble_rows)}')
    a = annual.loc[annual.model.eq(model)]
    if not set(member_rows.member).issubset(set(a.member)):
        raise ValueError(f'{model}: metrics and annual member IDs do not match')


def save(fig, stem):
    for ext in ('png', 'pdf'):
        path = OUT / f'{stem}.{ext}'
        fig.savefig(path, dpi=DPI, bbox_inches='tight')
        print('Saved:', path)
    plt.show()
    plt.close(fig)


def finish_axes(ax):
    ax.grid(axis='x', color='#D7D7D7', lw=.65, ls='--', alpha=.85)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color('#555555')
        s.set_linewidth(.8)
    ax.tick_params(labelsize=9)


# ============================================================
# Figure 14 — Five-model North Atlantic SST trend maps (2015–2100)
# ============================================================
if 'tos_trend' not in maps:
    raise KeyError('maps_2deg.nc is missing tos_trend')
trend_stack = maps['tos_trend'].sel(model=models).values
# Common absolute range; future SST is not forced to historical ±0.4°C/decade.
lim = float(np.nanpercentile(np.abs(trend_stack), 98))
lim = max(.1, np.ceil(lim * 10) / 10)
lon = maps['lon'].values
lat = maps['lat'].values
fig, axs = plt.subplots(3, 2, figsize=(12, 11), subplot_kw={'projection': ccrs.PlateCarree()})
fig.subplots_adjust(left=.075, right=.965, top=.955, bottom=.105, wspace=.11, hspace=.18)
last_im = None
for i, ax in enumerate(axs.flat):
    if i < 5:
        model = models[i]
        field = maps['tos_trend'].sel(model=model).values
        title = model
    else:
        # Recompute directly to enforce equal weight per model.
        field = np.mean(trend_stack, axis=0)
        title = 'Multi-model mean'
    last_im = ax.pcolormesh(lon, lat, field, cmap='RdBu_r', vmin=-lim, vmax=lim,
                            shading='auto', rasterized=True, transform=ccrs.PlateCarree())
    ax.set_extent([-80, 20, 20, 75], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale('110m'), facecolor='#D9D9D9', zorder=3)
    ax.coastlines('110m', lw=.55, zorder=4)
    ax.add_patch(Rectangle((-60, 45), 50, 15, lw=.95, fill=False,
                           edgecolor='black', transform=ccrs.PlateCarree(), zorder=8))
    gl = ax.gridlines(draw_labels=True, xlocs=[-60, -40, -20, 0],
                      ylocs=[30, 40, 50, 60, 70], lw=.4, color='#BDBDBD',
                      alpha=.6, ls='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = (i % 2 == 0)
    gl.bottom_labels = (i >= 4)
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    gl.xpadding = 3
    gl.ypadding = 3
    ax.set_title(f'({chr(97+i)}) {title}', fontsize=11, loc='left', pad=6)
    ax.spines['geo'].set_visible(True)
    ax.spines['geo'].set_edgecolor('#444444')
    ax.spines['geo'].set_linewidth(.75)
    ax.spines['geo'].set_zorder(20)

cax = fig.add_axes([.20, .055, .60, .015])
cb = fig.colorbar(last_im, cax=cax, orientation='horizontal', extend='both')
cb.set_label(r'SST trend ($^\circ\mathrm{C}\,\mathrm{decade}^{-1}$)', fontsize=10)
# No overall title, matching historical panel-first presentation.
save(fig, 'Fig14_SSP585_SST_trend_maps')


# ============================================================
# Figures 15/16 — Model and member dot-whisker comparisons
# ============================================================
def plot_three_panel(fig_number, indicators, xlabel, title_labels, stem):
    for name in indicators:
        if name not in metrics:
            raise KeyError(f'Column {name!r} not present in all_metrics.csv')

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 6.2), sharey=True)
    fig.subplots_adjust(left=.15, right=.985, top=.91, bottom=.12, wspace=.15)
    yy = np.arange(len(models))
    rows_out = []
    for ax, key, panel_title in zip(axes, indicators, title_labels):
        for i, model in enumerate(models):
            subset = metrics.loc[metrics.model.eq(model)]
            mem = subset.loc[subset.member.ne('ensemble')].sort_values('member')
            ens = subset.loc[subset.member.eq('ensemble')]
            v = mem[key].to_numpy(dtype=float)
            mean_from_ensemble = float(ens[key].iloc[0])
            member_sd = float(np.std(v, ddof=1))
            # Plot signed individual members in grey; mode-mean points in dark blue.
            jitter = np.linspace(-.13, .13, len(v))
            ax.scatter(v, i+jitter, s=23, facecolors=GRAY, alpha=.66,
                       edgecolors='none', zorder=4)
            ax.errorbar(mean_from_ensemble, i, xerr=member_sd, fmt='o',
                        color=DARK, ecolor=DARK, markersize=6,
                        elinewidth=1.2, capsize=3, zorder=5)
            rows_out.append({'figure':fig_number, 'model':model,
                             'indicator':key, 'ensemble_value':mean_from_ensemble,
                             'member_mean':float(np.mean(v)),
                             'member_sd':member_sd, 'n_members':len(v)})
        if key.startswith('whi_'):
            ax.axvline(0, color='#A33A3A', ls='--', lw=1.0, zorder=3)
        elif min(metrics[key]) <= 0 <= max(metrics[key]):
            ax.axvline(0, color='#444444', ls='--', lw=.85, zorder=3)
        ax.set_title(panel_title, fontsize=12, weight='semibold', loc='center', pad=8)
        ax.set_xlabel(xlabel, fontsize=10)
        finish_axes(ax)
        ax.set_ylim(len(models)-.55, -.55)
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels(models, fontsize=9)
    for ax in axes[1:]:
        ax.tick_params(axis='y', left=False, labelleft=False)
    # Do not share an x-axis scale across SPNA/Global/WHI: different magnitudes.
    save(fig, stem)
    pd.DataFrame(rows_out).to_csv(OUT/f'{stem}_values.csv', index=False)


plot_three_panel(
    15,
    indicators=('sst_trend', 'global_sst_trend', 'whi_trend'),
    xlabel=r'SST trend ($^\circ\mathrm{C}\,\mathrm{decade}^{-1}$)',
    title_labels=('(a) SPNA', '(b) Global', '(c) Global−SPNA'),
    stem='Fig15_SSP585_SST_trends',
)

plot_three_panel(
    16,
    indicators=('sst_delta', 'global_sst_delta', 'whi_delta'),
    xlabel=r'SST change ($^\circ\mathrm{C}$)',
    title_labels=('(a) SPNA', '(b) Global', '(c) Global−SPNA'),
    stem='Fig16_SSP585_SST_late_century_changes',
)

print('Finished Figures 14–16. Output:', OUT)
print('Fig 15: OLS trends, 2015–2100; Fig 16: 2081–2100 minus 1995–2014.')
print('WHI is Global ocean SST minus SPNA SST.')


"""Figures 14–16: SSP5-8.5 SST trends and late-century changes.

Reads the three precomputed inputs only; no monthly NetCDF processing.
Run: python plot_fig14_16_ssp585.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle

ROOT = Path(str(PROJECT_ROOT / 'data/ssp585_analysis'))
MAP_FILE = ROOT / 'annual/maps_2deg.nc'
METRICS_FILE = ROOT / 'tables/all_metrics.csv'
ANNUAL_FILE = ROOT / 'tables/annual_indices_1950_2100.csv'
OUT = Path(str(PROJECT_ROOT / 'results/figures'))
OUT.mkdir(parents=True, exist_ok=True)

# Prefer same sequence as historical figures, limited to five available models.
PREFERRED_ORDER = [
    'ACCESS-CM2', 'ACCESS-ESM1-5', 'CESM2', 'CanESM5',
    'CanESM5-1', 'ICON-ESM-LR', 'MIROC6', 'MPI-ESM1-2-HR',
    'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'FGOALS-g3', 'NorCPM1',
]
RED, BLUE, DARK, GRAY = '#C0392B', '#2166AC', '#273746', '#969696'
DPI = 300
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 12, 'axes.labelsize': 10,
    'axes.spines.top': True, 'axes.spines.right': True,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

for p in (MAP_FILE, METRICS_FILE, ANNUAL_FILE):
    if not p.exists():
        raise FileNotFoundError(f'Input does not exist: {p}')

metrics = pd.read_csv(METRICS_FILE)
annual = pd.read_csv(ANNUAL_FILE)
metrics = metrics.loc[metrics['region'].eq('main')].copy()
annual = annual.loc[annual['region'].eq('main')].copy()
with xr.open_dataset(MAP_FILE) as ds:
    maps = ds.load()

available_models = set(metrics['model'].unique())
models = [m for m in PREFERRED_ORDER if m in available_models]
models += sorted(available_models - set(models))
if len(models) != 5:
    raise ValueError(f'Expected five future models, found {models}')
map_models = set(maps['model'].values.astype(str))
if set(models) != map_models:
    raise ValueError(f'Map models {map_models} differ from metric models {set(models)}')

# Confirm full 25-member set and precomputed indices; do not silently omit members.
for model in models:
    block = metrics.loc[metrics.model.eq(model)]
    member_rows = block.loc[block.member.ne('ensemble')]
    ensemble_rows = block.loc[block.member.eq('ensemble')]
    if len(member_rows) != 5 or len(ensemble_rows) != 1:
        raise ValueError(f'{model}: expected five members + ensemble, got {len(member_rows)} + {len(ensemble_rows)}')
    a = annual.loc[annual.model.eq(model)]
    if not set(member_rows.member).issubset(set(a.member)):
        raise ValueError(f'{model}: metrics and annual member IDs do not match')


def save(fig, stem):
    for ext in ('png', 'pdf'):
        path = OUT / f'{stem}.{ext}'
        fig.savefig(path, dpi=DPI, bbox_inches='tight')
        print('Saved:', path)
    plt.show()
    plt.close(fig)


def finish_axes(ax):
    ax.grid(axis='x', color='#D7D7D7', lw=.65, ls='--', alpha=.85)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color('#555555')
        s.set_linewidth(.8)
    ax.tick_params(labelsize=9)


# ============================================================
# Figure 14 — Five-model North Atlantic SST trend maps (2015–2100)
# ============================================================
if 'tos_trend' not in maps:
    raise KeyError('maps_2deg.nc is missing tos_trend')
trend_stack = maps['tos_trend'].sel(model=models).values
# Common absolute range; future SST is not forced to historical ±0.4°C/decade.
lim = float(np.nanpercentile(np.abs(trend_stack), 98))
lim = max(.1, np.ceil(lim * 10) / 10)
lon = maps['lon'].values
lat = maps['lat'].values
fig, axs = plt.subplots(3, 2, figsize=(12, 11), subplot_kw={'projection': ccrs.PlateCarree()})
fig.subplots_adjust(left=.075, right=.965, top=.955, bottom=.105, wspace=.11, hspace=.18)
last_im = None
for i, ax in enumerate(axs.flat):
    if i < 5:
        model = models[i]
        field = maps['tos_trend'].sel(model=model).values
        title = model
    else:
        # Recompute directly to enforce equal weight per model.
        field = np.mean(trend_stack, axis=0)
        title = 'Multi-model mean'
    last_im = ax.pcolormesh(lon, lat, field, cmap='RdBu_r', vmin=-lim, vmax=lim,
                            shading='auto', rasterized=True, transform=ccrs.PlateCarree())
    ax.set_extent([-80, 20, 20, 75], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale('110m'), facecolor='#D9D9D9', zorder=3)
    ax.coastlines('110m', lw=.55, zorder=4)
    ax.add_patch(Rectangle((-60, 45), 50, 15, lw=.95, fill=False,
                           edgecolor='black', transform=ccrs.PlateCarree(), zorder=8))
    gl = ax.gridlines(draw_labels=True, xlocs=[-60, -40, -20, 0],
                      ylocs=[30, 40, 50, 60, 70], lw=.4, color='#BDBDBD',
                      alpha=.6, ls='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = (i % 2 == 0)
    gl.bottom_labels = (i >= 4)
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    gl.xpadding = 3
    gl.ypadding = 3
    ax.set_title(f'({chr(97+i)}) {title}', fontsize=11, loc='left', pad=6)
    ax.spines['geo'].set_visible(True)
    ax.spines['geo'].set_edgecolor('#444444')
    ax.spines['geo'].set_linewidth(.75)
    ax.spines['geo'].set_zorder(20)
fig.suptitle(
    "North Atlantic SST trends (2015–2100)",
    fontsize=14,
    y=0.98
    
)
cax = fig.add_axes([.20, .055, .60, .015])
cb = fig.colorbar(last_im, cax=cax, orientation='horizontal', extend='both')
cb.set_label(r'SST trend ($^\circ\mathrm{C}\,\mathrm{decade}^{-1}$)', fontsize=10)
# No overall title, matching historical panel-first presentation.
save(fig, 'Fig14_SSP585_SST_trend_maps')


# ============================================================
# Figures 15/16 — Model and member dot-whisker comparisons
# ============================================================
def plot_three_panel(fig_number, indicators, xlabel, title_labels, stem):
    for name in indicators:
        if name not in metrics:
            raise KeyError(f'Column {name!r} not present in all_metrics.csv')

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 6.2), sharey=True)
    fig.subplots_adjust(left=.15, right=.985, top=.91, bottom=.12, wspace=.15)
    yy = np.arange(len(models))
    rows_out = []
    for ax, key, panel_title in zip(axes, indicators, title_labels):
        for i, model in enumerate(models):
            subset = metrics.loc[metrics.model.eq(model)]
            mem = subset.loc[subset.member.ne('ensemble')].sort_values('member')
            ens = subset.loc[subset.member.eq('ensemble')]
            v = mem[key].to_numpy(dtype=float)
            mean_from_ensemble = float(ens[key].iloc[0])
            member_sd = float(np.std(v, ddof=1))
            # Plot signed individual members in grey; mode-mean points in dark blue.
            jitter = np.linspace(-.13, .13, len(v))
            ax.scatter(v, i+jitter, s=23, facecolors=GRAY, alpha=.66,
                       edgecolors='none', zorder=4)
            ax.errorbar(mean_from_ensemble, i, xerr=member_sd, fmt='o',
                        color=DARK, ecolor=DARK, markersize=6,
                        elinewidth=1.2, capsize=3, zorder=5)
            rows_out.append({'figure':fig_number, 'model':model,
                             'indicator':key, 'ensemble_value':mean_from_ensemble,
                             'member_mean':float(np.mean(v)),
                             'member_sd':member_sd, 'n_members':len(v)})
        if key.startswith('whi_'):
            ax.axvline(0, color='#A33A3A', ls='--', lw=1.0, zorder=3)
        elif min(metrics[key]) <= 0 <= max(metrics[key]):
            ax.axvline(0, color='#444444', ls='--', lw=.85, zorder=3)
        ax.set_title(panel_title, fontsize=12, weight='semibold', loc='center', pad=8)
        ax.set_xlabel(xlabel, fontsize=10)
        finish_axes(ax)
        ax.set_ylim(len(models)-.55, -.55)
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels(models, fontsize=9)
    for ax in axes[1:]:
        ax.tick_params(axis='y', left=False, labelleft=False)
    # Do not share an x-axis scale across SPNA/Global/WHI: different magnitudes.
    save(fig, stem)
    pd.DataFrame(rows_out).to_csv(OUT/f'{stem}_values.csv', index=False)


plot_three_panel(
    15,
    indicators=('sst_trend', 'global_sst_trend', 'whi_trend'),
    xlabel=r'SST trend ($^\circ\mathrm{C}\,\mathrm{decade}^{-1}$)',
    title_labels=('(a) SPNA', '(b) Global', '(c) Global−SPNA'),
    stem='Fig15_SSP585_SST_trends',
)

plot_three_panel(
    16,
    indicators=('sst_delta', 'global_sst_delta', 'whi_delta'),
    xlabel=r'SST change ($^\circ\mathrm{C}$)',
    title_labels=('(a) SPNA', '(b) Global', '(c) Global−SPNA'),
    stem='Fig16_SSP585_SST_late_century_changes',
)

print('Finished Figures 14–16. Output:', OUT)
print('Fig 15: OLS trends, 2015–2100; Fig 16: 2081–2100 minus 1995–2014.')
print('WHI is Global ocean SST minus SPNA SST.')


# ============================================================
# Figure 15: Future SST trends across SSP5-8.5 models
# Period: 2015–2100
# Panels: SPNA, Global ocean, Global–SPNA
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# ------------------------------------------------------------
# 1. Data paths
# ------------------------------------------------------------

ROOT = Path(
    str(PROJECT_ROOT / 'results')
)

METRICS_FILE = str(PROJECT_ROOT / 'data/ssp585_analysis/tables/all_metrics.csv')

OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Read the previously calculated metrics
met = pd.read_csv(METRICS_FILE)

# Select the main SPNA region
met = met.loc[
    met["region"] == "main"
].copy()


# ------------------------------------------------------------
# 2. Models and colors
# ------------------------------------------------------------

MODELS_ORDER = [
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

panels = [
    ("sst_trend", "(a) SPNA"),
    ("global_sst_trend", "(b) Global"),
    ("whi_trend", "(c) Global−SPNA"),
]


# ------------------------------------------------------------
# 3. Validate input data
# ------------------------------------------------------------

required_columns = {
    "model",
    "member",
    "region",
    "sst_trend",
    "global_sst_trend",
    "whi_trend",
}

missing = required_columns - set(met.columns)

if missing:
    raise ValueError(
        f"Missing columns in all_metrics.csv: {missing}"
    )

for model in MODELS_ORDER:

    sub = met.loc[
        (met["model"] == model)
        & (met["member"] != "ensemble")
    ]

    ens = met.loc[
        (met["model"] == model)
        & (met["member"] == "ensemble")
    ]

    if len(sub) != 5 or len(ens) != 1:
        raise ValueError(
            f"{model}: expected 5 members and "
            f"1 ensemble record, got {len(sub)} and {len(ens)}"
        )


# ------------------------------------------------------------
# 4. Figure layout
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1, 3,
    figsize=(14.5, 5.3),
    sharey=True
)

positions = np.arange(len(MODELS_ORDER))


# ------------------------------------------------------------
# 5. Plot
# ------------------------------------------------------------

for ax, (column, title) in zip(axes, panels):

    for i, model in enumerate(MODELS_ORDER):

        color = MODEL_COLORS[model]

        # Five individual members
        sub = met.loc[
            (met["model"] == model)
            & (met["member"] != "ensemble")
        ].sort_values("member")

        values = sub[column].to_numpy(dtype=float)

        # Ensemble mean from the original analysis
        ens = met.loc[
            (met["model"] == model)
            & (met["member"] == "ensemble")
        ]

        mean = float(ens[column].iloc[0])

        # Between-member sample standard deviation
        sd = np.std(values, ddof=1)

        # Individual member points
        ax.scatter(
            values,
            i + np.linspace(
                -0.12, 0.12, len(values)
            ),
            s=26,
            color=color,
            alpha=0.45,
            linewidths=0,
            zorder=3
        )

        # Ensemble mean ± 1 SD
        ax.errorbar(
            mean,
            i,
            xerr=sd,
            fmt="o",
            color=color,
            ecolor=color,
            markersize=6.5,
            elinewidth=1.4,
            capsize=3,
            capthick=1,
            zorder=5
        )

    # Centered panel title
    ax.set_title(
        title,
        fontsize=12,
        fontweight="bold",
        loc="center",
        pad=10
    )

    ax.set_yticks(positions)
    ax.set_ylim(
        len(MODELS_ORDER) - 0.5,
        -0.5
    )

    ax.set_xlabel(
        r"SST trend ($^\circ$C decade$^{-1}$)",
        fontsize=10
    )

    # Grid
    ax.grid(
        axis="x",
        linestyle="--",
        color="#DDDDDD",
        linewidth=0.65
    )

    ax.set_axisbelow(True)

    # Full panel borders
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color("#555555")

    ax.tick_params(
        axis="both",
        labelsize=9,
        direction="out"
    )


# ------------------------------------------------------------
# 6. Model labels
# ------------------------------------------------------------

axes[0].set_yticklabels(
    MODELS_ORDER,
    fontsize=9
)

axes[1].tick_params(
    axis="y",
    labelleft=False
)

axes[2].tick_params(
    axis="y",
    labelleft=False
)


# ------------------------------------------------------------
# 7. Common x-axis for SPNA and Global
# ------------------------------------------------------------

# Same range and tick interval for panels (a) and (b)
for ax in axes[:2]:

    ax.set_xlim(0.15, 0.90)

    ax.xaxis.set_major_locator(
        MultipleLocator(0.15)
    )


# ------------------------------------------------------------
# 8. Independent x-axis for Global-SPNA
# ------------------------------------------------------------

axes[2].set_xlim(-0.35, 0.20)

axes[2].xaxis.set_major_locator(
    MultipleLocator(0.10)
)

# Zero reference line for WHI
axes[2].axvline(
    0,
    color="#C0392B",
    linestyle="--",
    linewidth=1.0,
    zorder=2
)


# ------------------------------------------------------------
# 9. Layout
# ------------------------------------------------------------
# Overall figure title
# 布局设置
fig.subplots_adjust(
    left=0.16,
    right=0.98,
    top=0.85,
    bottom=0.16,
    wspace=0.15
)

# 标题居中于三个子图区域，而非整张画布
fig.suptitle(
    "SST trends under SSP5-8.5 (2015–2100)",
    fontsize=16,
    x=(0.16 + 0.98) / 2,
    y=0.96,
    ha="center"
)

# ------------------------------------------------------------
# 10. Save
# ------------------------------------------------------------

for ext in ("png", "pdf"):

    output_file = (
        OUT / f"Fig15_SST_trends.{ext}"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    print("Saved:", output_file)

plt.show()
plt.close(fig)

# ============================================================
# Figure 14 — Five-model North Atlantic SST trend maps (2015–2100)
# Layout: 2 rows × 3 columns
# ============================================================

if 'tos_trend' not in maps:
    raise KeyError('maps_2deg.nc is missing tos_trend')

trend_stack = maps['tos_trend'].sel(model=models).values

# Common color scale
lim = float(np.nanpercentile(np.abs(trend_stack), 98))
lim = max(0.1, np.ceil(lim * 10) / 10)

lon = maps['lon'].values
lat = maps['lat'].values

# ------------------------------------------------------------
# 1. Figure layout: 2 × 3
# ------------------------------------------------------------

fig, axs = plt.subplots(
    2, 3,
    figsize=(16, 7.5),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

# 增加顶部空间，拉开总标题与子图之间的距离
fig.subplots_adjust(
    left=0.065,
    right=0.965,
    top=0.86,
    bottom=0.16,
    wspace=0.12,
    hspace=0.20
)

# ------------------------------------------------------------
# 2. Plot maps
# ------------------------------------------------------------

last_im = None

for i, ax in enumerate(axs.flat):

    if i < 5:
        model = models[i]
        field = maps['tos_trend'].sel(model=model).values
        title = model

    else:
        # Equal weight for each model
        field = np.mean(trend_stack, axis=0)
        title = 'Multi-model mean'

    last_im = ax.pcolormesh(
        lon,
        lat,
        field,
        cmap='RdBu_r',
        vmin=-lim,
        vmax=lim,
        shading='auto',
        rasterized=True,
        transform=ccrs.PlateCarree()
    )

    ax.set_extent(
        [-80, 20, 20, 75],
        crs=ccrs.PlateCarree()
    )

    # Land and coastline
    ax.add_feature(
        cfeature.LAND.with_scale('110m'),
        facecolor='#D9D9D9',
        zorder=3
    )

    ax.coastlines(
        '110m',
        lw=0.55,
        zorder=4
    )

    # SPNA rectangle
    ax.add_patch(
        Rectangle(
            (-60, 45),
            50,
            15,
            lw=0.95,
            fill=False,
            edgecolor='black',
            transform=ccrs.PlateCarree(),
            zorder=8
        )
    )

    # Gridlines
    gl = ax.gridlines(
        draw_labels=True,
        xlocs=[-60, -40, -20, 0],
        ylocs=[30, 40, 50, 60, 70],
        lw=0.4,
        color='#BDBDBD',
        alpha=0.6,
        ls='--'
    )

    gl.top_labels = False
    gl.right_labels = False

    # 2 × 3 layout:
    # left labels only in first column
    # bottom labels only in second row
    gl.left_labels = (i % 3 == 0)
    gl.bottom_labels = (i >= 3)

    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}

    gl.xpadding = 3
    gl.ypadding = 3

    # Panel title
    ax.set_title(
        f'({chr(97+i)}) {title}',
        fontsize=11,
        loc='left',
        pad=8
    )

    # Complete map border
    ax.spines['geo'].set_visible(True)
    ax.spines['geo'].set_edgecolor('#444444')
    ax.spines['geo'].set_linewidth(0.75)
    ax.spines['geo'].set_zorder(20)


# ------------------------------------------------------------
# 3. Overall title
# ------------------------------------------------------------

fig.suptitle(
    "North Atlantic SST trends (2015–2100)",
    fontsize=16,
    x=0.515,
    y=0.955,
    ha='center'
)


# ------------------------------------------------------------
# 4. Shared colorbar
# ------------------------------------------------------------

cax = fig.add_axes([
    0.20,   # left
    0.075,  # bottom
    0.60,   # width
    0.018   # height
])

cb = fig.colorbar(
    last_im,
    cax=cax,
    orientation='horizontal',
    extend='both'
)

cb.set_label(
    r'SST trend ($^\circ\mathrm{C}\,\mathrm{decade}^{-1}$)',
    fontsize=10
)


# ------------------------------------------------------------
# 5. Save
# ------------------------------------------------------------

save(fig, 'Fig14_SSP585_SST_trend_maps')