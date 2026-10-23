from pathlib import Path
import os
_PROJECT_START = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
PROJECT_ROOT = next((p for p in [_PROJECT_START, *_PROJECT_START.parents] if (p / "PROJECT_ROOT.marker").is_file()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Run from the project folder or its notebooks/scripts subfolder.")



from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from scipy.spatial import cKDTree
from scipy.stats import linregress, t as student_t


# ============================================================
# 1. Configuration
# ============================================================

ROOT = Path(str(PROJECT_ROOT / 'data'))
ANNUAL_DIR = ROOT / "historical_analysis/annual"
AREA_ROOT = ROOT / "CMIP6"

OUT = Path(
    str(PROJECT_ROOT / 'results/figures')
)
OUT.mkdir(parents=True, exist_ok=True)

AMOC_FILE = OUT / "AMOC_member_annual.csv"

MODELS = [
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

YEARS = np.arange(1950, 2015)

EARLY = (1950, 1969)
LATE = (1995, 2014)

# SPNA box: 45–60N, 60–10W
SPNA = (-60, -10, 45, 60)

# Common visualization grid (2 degrees)
MAP_LON = np.arange(-80, 21, 2.0)
MAP_LAT = np.arange(20, 76, 2.0)
LON2D, LAT2D = np.meshgrid(MAP_LON, MAP_LAT)

# Nearest-neighbor search radius for map display
REGRID_RADIUS_DEG = 2.5

# Matching tolerance between annual and areacello grids
AREA_MATCH_TOL_DEG = 0.02

# Wang et al. (2023, Earth and Space Science, DOI:10.1029/2023EA002887)
# Annual MHT regressed against AMOC strength, CMIP6 ensemble slope.
# Units: PW per Sv. This is an externally sourced statistical coefficient,
# not a model-specific diagnosed transport or a temperature difference.
MHT_AMOC_COEFF = 0.054  # PW / Sv

DPI = 300

RED = "#C0392B"
BLUE = "#2166AC"
GRAY = "#777777"
BLACK = "#222222"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "axes.spines.top": True,
    "axes.spines.right": True,
})


# ============================================================
# 2. Helpers: files and grids
# ============================================================

def member_sort_key(member):
    nums = re.fullmatch(
        r"r(\d+)i(\d+)p(\d+)f(\d+)", member
    )
    return tuple(map(int, nums.groups()))


def find_member_files(model):
    found = {}

    for path in ANNUAL_DIR.glob(
        f"{model}_*_annual.nc"
    ):
        if path.name.startswith("._"):
            continue

        member = path.name[
            len(model) + 1:-len("_annual.nc")
        ]

        if re.fullmatch(r"r\d+i\d+p\d+f\d+", member):
            found[member] = path

    if len(found) != 5:
        raise ValueError(
            f"{model}: expected 5 annual members, "
            f"found {list(found)}"
        )

    return {
        m: found[m]
        for m in sorted(found, key=member_sort_key)
    }


MEMBER_FILES = {
    model: find_member_files(model)
    for model in MODELS
}


def get_var(ds, candidates):
    for name in candidates:
        if name in ds:
            return ds[name]

    raise KeyError(
        f"Cannot find {candidates}. "
        f"Available: {list(ds.data_vars)}"
    )


def get_years(da):
    if "year" in da.dims:
        years = np.asarray(da["year"].values).astype(int)
        tdim = "year"

    elif "time" in da.dims:
        years = np.asarray(
            da["time"].dt.year.values
        ).astype(int)
        tdim = "time"

    else:
        raise ValueError(
            f"Missing year/time dimension: {da.dims}"
        )

    if not np.array_equal(years, YEARS):
        raise ValueError(
            "Expected complete annual coverage 1950–2014; "
            f"got {years.min()}–{years.max()} "
            f"({len(years)} records)"
        )

    return tdim


def get_geo(ds, spatial_shape):
    """
    Return flattened native-grid latitude and longitude.

    Supports:
      lat(cell), lon(cell)
      lat(y,x), lon(y,x)
      lat(y), lon(x) regular grid
    """

    lat_da = get_var(
        ds, ["lat", "latitude", "nav_lat", "TLAT"]
    )
    lon_da = get_var(
        ds, ["lon", "longitude", "nav_lon", "TLONG"]
    )

    lat = np.asarray(lat_da.values, dtype=float)
    lon = np.asarray(lon_da.values, dtype=float)

    if lat.shape == spatial_shape and lon.shape == spatial_shape:
        pass

    elif (
        len(spatial_shape) == 2
        and lat.ndim == 1
        and lon.ndim == 1
        and len(lat) == spatial_shape[0]
        and len(lon) == spatial_shape[1]
    ):
        lon, lat = np.meshgrid(lon, lat)

    else:
        raise ValueError(
            f"Coordinates {lat.shape}, {lon.shape} "
            f"do not match spatial shape {spatial_shape}"
        )

    lon = (lon + 180) % 360 - 180

    return lat.ravel(), lon.ravel()


def xyz(lat, lon):
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)

    return np.column_stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ])


def chord_radius(angle_deg):
    return 2 * np.sin(np.deg2rad(angle_deg) / 2)


# ============================================================
# 3. Match original-grid areacello
# ============================================================

def area_candidates(model, member):
    """
    Prefer same member. If coordinates do not match,
    try another member's areacello from the same model.
    """

    root = AREA_ROOT / model / "historical"

    paths = [
        p for p in root.glob(
            "**/areacello_Ofx_*.nc"
        )
        if not p.name.startswith("._")
    ]

    paths.sort(
        key=lambda p: (
            member not in p.name,
            str(p)
        )
    )

    return paths


def native_area(ds, model, member, lat, lon, spatial_shape):
    """
    Match annual native points against area coordinates.

    Reject incompatible grids instead of blindly flattening
    an areacello file.
    """

    for name in ("areacello", "cell_area", "area"):
        if name in ds:
            candidate = np.asarray(
                ds[name].values, dtype=float
            ).squeeze()

            if candidate.shape == spatial_shape:
                return candidate.ravel()

    valid_target = (
        np.isfinite(lat) & np.isfinite(lon)
    )

    for path in area_candidates(model, member):

        try:
            with xr.open_dataset(path) as ads:
                area_da = get_var(
                    ads, ["areacello", "cell_area"]
                ).squeeze(drop=True)

                shape = area_da.shape
                a_lat, a_lon = get_geo(ads, shape)

                area = np.asarray(
                    area_da.values, dtype=float
                ).ravel()

            valid_source = (
                np.isfinite(a_lat)
                & np.isfinite(a_lon)
                & np.isfinite(area)
                & (area > 0)
            )

            target_xyz = xyz(
                lat[valid_target],
                lon[valid_target]
            )

            source_xyz = xyz(
                a_lat[valid_source],
                a_lon[valid_source]
            )

            if len(source_xyz) == 0:
                continue

            tree = cKDTree(source_xyz)

            dist, index = tree.query(
                target_xyz,
                distance_upper_bound=chord_radius(
                    AREA_MATCH_TOL_DEG
                )
            )

            matched = np.isfinite(dist)

            # All finite native positions must be matched
            # without duplicated source cells.
            if (
                not matched.all()
                or len(np.unique(index)) != len(index)
            ):
                continue

            source_area = area[valid_source]
            result = np.full(lat.size, np.nan)

            result[valid_target] = source_area[index]

            print(
                f"  area matched: {model}/{member} "
                f"using {path.name}"
            )

            return result

        except Exception as exc:
            warnings.warn(
                f"Area candidate failed: {path.name}: {exc}"
            )

    raise ValueError(
        f"No grid-compatible areacello found: "
        f"{model}/{member}"
    )


# ============================================================
# 4. OLS spatial trends + p-values
# ============================================================

def spatial_ols(fields, years):
    """
    fields: (time, lat, lon)

    Two-sided ordinary OLS test, without autocorrelation
    correction or multiple-testing correction.
    """

    nt, ny, nx = fields.shape
    values = fields.reshape(nt, -1).astype(float)

    result_slope = np.full(values.shape[1], np.nan)
    result_p = np.full(values.shape[1], np.nan)

    valid_count = np.isfinite(values).sum(axis=0)
    columns = np.where(valid_count >= 10)[0]

    if len(columns) == 0:
        return (
            result_slope.reshape(ny, nx),
            result_p.reshape(ny, nx)
        )

    # Vectorized OLS with NaN-aware sums
    a = values[:, columns]
    good = np.isfinite(a)
    n = good.sum(axis=0)

    xx = years.astype(float)[:, None]
    xx0 = np.where(good, xx, 0.0)
    yy0 = np.where(good, a, 0.0)

    sx = xx0.sum(axis=0)
    sy = yy0.sum(axis=0)
    sxx = (xx0 * xx0).sum(axis=0)
    sxy = (xx0 * yy0).sum(axis=0)

    denom = sxx - sx * sx / n

    slope = np.full(len(columns), np.nan)
    pvalue = np.full(len(columns), np.nan)

    ok = denom > 0

    slope[ok] = (
        sxy[ok] - sx[ok] * sy[ok] / n[ok]
    ) / denom[ok]

    intercept = sy / n - slope * sx / n

    fitted = intercept[None, :] + slope[None, :] * xx

    residual = np.where(
        good, a - fitted, 0.0
    )

    rss = (residual**2).sum(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        stderr = np.sqrt(
            (rss / (n - 2)) / denom
        )
        tstat = slope / stderr

    pvalue[ok] = 2 * student_t.sf(
        np.abs(tstat[ok]),
        df=n[ok] - 2
    )

    result_slope[columns] = slope * 10
    result_p[columns] = pvalue

    return (
        result_slope.reshape(ny, nx),
        result_p.reshape(ny, nx)
    )


# ============================================================
# 5. Read hfds, calculate native regional indices,
#    and remap to 2-degree visualization grid
# ============================================================

target_xyz = xyz(
    LAT2D.ravel(),
    LON2D.ravel()
)

surface_rows = []
surface_change_rows = []
map_results = {}

for model in MODELS:

    print(f"\nProcessing {model}", flush=True)

    ensemble_sum = np.zeros(
        (len(YEARS),) + LAT2D.shape,
        dtype=np.float64
    )

    ensemble_count = np.zeros(
        (len(YEARS),) + LAT2D.shape,
        dtype=np.int16
    )

    for member, path in MEMBER_FILES[model].items():

        with xr.open_dataset(path) as ds:

            da = get_var(
                ds,
                ["hfds", "HFDS", "hfds_annual"]
            )

            tdim = get_years(da)

            spatial_dims = [
                d for d in da.dims if d != tdim
            ]

            da = da.transpose(tdim, *spatial_dims)

            spatial_shape = da.shape[1:]
            lat, lon = get_geo(ds, spatial_shape)

            area = native_area(
                ds, model, member,
                lat, lon, spatial_shape
            )

            units = str(
                da.attrs.get("units", "")
            ).lower().replace(" ", "")

            if units and not (
                "w" in units and (
                    "m-2" in units
                    or "m^-2" in units
                    or "/m2" in units
                    or "m**-2" in units
                )
            ):
                raise ValueError(
                    f"{path.name}: unexpected hfds units "
                    f"{da.attrs.get('units')}"
                )

            # SPNA mask on native grid
            region = (
                (lon >= SPNA[0])
                & (lon <= SPNA[1])
                & (lat >= SPNA[2])
                & (lat <= SPNA[3])
                & np.isfinite(area)
                & (area > 0)
            )

            if not np.any(region):
                raise ValueError(
                    f"SPNA has no valid cells: {model}/{member}"
                )

            # Native cells for visualization near Atlantic
            near = (
                (lon >= -83)
                & (lon <= 23)
                & (lat >= 17)
                & (lat <= 78)
                & np.isfinite(lat)
                & np.isfinite(lon)
            )

            source_indices = np.flatnonzero(near)

            tree = cKDTree(
                xyz(lat[near], lon[near])
            )

            distances, indices = tree.query(
                target_xyz,
                distance_upper_bound=chord_radius(
                    REGRID_RADIUS_DEG
                )
            )

            valid_map = np.isfinite(distances)
            map_indices = np.zeros(len(target_xyz), dtype=int)
            map_indices[valid_map] = source_indices[
                indices[valid_map]
            ]

            # Fixed valid-ocean mask over 1950-2014: avoid spurious
            # changes in integrated power caused by varying missing cells.
            stable_region = region.copy()
            for k in range(len(YEARS)):
                stable_region &= np.isfinite(
                    np.asarray(da.isel({tdim: k}).values, dtype=float).ravel()
                )
            if not stable_region.any():
                raise ValueError(f"No stable hfds cells: {model}/{member}")
            valid_fraction = np.sum(area[stable_region]) / np.sum(area[region])
            if valid_fraction < 0.99:
                warnings.warn(
                    f"{model}/{member}: fixed-mask coverage "
                    f"{valid_fraction:.2%} of SPNA native area"
                )

            annual_region = []
            annual_integral = []

            for k, year in enumerate(YEARS):

                field = np.asarray(
                    da.isel({tdim: k}).values,
                    dtype=float
                ).ravel()

                selected = stable_region

                weights = area[selected]
                values = field[selected]

                qmean = np.sum(
                    values * weights
                ) / np.sum(weights)

                qtotal = np.sum(
                    values * weights
                ) / 1e15

                annual_region.append(qmean)
                annual_integral.append(qtotal)

                surface_rows.append({
                    "model": model,
                    "member": member,
                    "year": int(year),
                    "hfds_spna_Wm2": qmean,
                    "surface_heat_PW": qtotal,
                })

                # Common display grid, NaN where no nearby point
                mapped = np.full(
                    target_xyz.shape[0], np.nan
                )

                mapped[valid_map] = field[
                    map_indices[valid_map]
                ]

                mapped = mapped.reshape(LAT2D.shape)
                finite = np.isfinite(mapped)

                ensemble_sum[k, finite] += mapped[finite]
                ensemble_count[k, finite] += 1

            annual_region = np.asarray(annual_region)
            annual_integral = np.asarray(annual_integral)

            region_fit = linregress(
                YEARS, annual_region
            )

            early_mask = (
                (YEARS >= EARLY[0])
                & (YEARS <= EARLY[1])
            )

            late_mask = (
                (YEARS >= LATE[0])
                & (YEARS <= LATE[1])
            )

            dq = (
                annual_integral[late_mask].mean()
                - annual_integral[early_mask].mean()
            )

            surface_change_rows.append({
                "model": model,
                "member": member,
                "hfds_trend_Wm2_decade": (
                    region_fit.slope * 10
                ),
                "hfds_pvalue": region_fit.pvalue,
                "delta_surface_PW": dq,
            })

    ensemble = np.full(
        ensemble_sum.shape, np.nan
    )

    np.divide(
        ensemble_sum,
        ensemble_count,
        out=ensemble,
        where=ensemble_count > 0
    )

    trend, pvalue = spatial_ols(
        ensemble, YEARS
    )

    map_results[model] = {
        "trend": trend,
        "pvalue": pvalue,
    }


surface_df = pd.DataFrame(surface_rows)
surface_change_df = pd.DataFrame(surface_change_rows)

surface_df.to_csv(
    OUT / "Fig11_13_hfds_member_annual.csv",
    index=False
)

surface_change_df.to_csv(
    OUT / "Fig12_13_hfds_member_statistics.csv",
    index=False
)


# ============================================================
# 6. Save Figure 11 reusable gridded data
# ============================================================

ds_plot = xr.Dataset(
    {
        "hfds_trend": (
            ("model", "lat", "lon"),
            np.stack([
                map_results[m]["trend"]
                for m in MODELS
            ])
        ),
        "pvalue": (
            ("model", "lat", "lon"),
            np.stack([
                map_results[m]["pvalue"]
                for m in MODELS
            ])
        )
    },
    coords={
        "model": MODELS,
        "lat": MAP_LAT,
        "lon": MAP_LON,
    },
    attrs={
        "period": "1950–2014",
        "trend_units": "W/m2/decade",
        "SPNA": "45–60N, 60–10W",
        "test": "Two-sided ordinary OLS",
        "remapping": "Nearest neighbor to common 2-degree grid",
    }
)

ds_plot.to_netcdf(
    OUT / "Fig11_hfds_plot_data.nc"
)


# ============================================================
# 7. Figure 11: North Atlantic hfds trend maps
# ============================================================

all_trends = np.stack([
    map_results[m]["trend"] for m in MODELS
])

# Same symmetric color limits for all maps
abs_limit = np.nanpercentile(
    np.abs(all_trends), 98
)

abs_limit = max(
    1.0,
    np.ceil(abs_limit)
)

norm = mcolors.TwoSlopeNorm(
    vmin=-abs_limit,
    vcenter=0,
    vmax=abs_limit
)

fig, axes = plt.subplots(
    3, 4,
    figsize=(17, 9),
    subplot_kw={
        "projection": ccrs.PlateCarree()
    }
)

for i, model in enumerate(MODELS):

    ax = axes.flat[i]

    trend = map_results[model]["trend"]
    pval = map_results[model]["pvalue"]

    image = ax.pcolormesh(
        LON2D,
        LAT2D,
        trend,
        cmap="RdBu_r",
        norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
        rasterized=True,
    )

    # Sparse stippling for p < 0.05
    rr, cc = np.where(
        np.isfinite(pval)
        & (pval < 0.05)
        & np.isfinite(trend)
    )

    keep = (
        (rr % 2 == 0)
        & (cc % 2 == 0)
    )

    ax.scatter(
        LON2D[rr[keep], cc[keep]],
        LAT2D[rr[keep], cc[keep]],
        s=1.5,
        color="black",
        marker=".",
        transform=ccrs.PlateCarree(),
        zorder=4
    )

    # SPNA rectangle
    ax.plot(
        [-60, -10, -10, -60, -60],
        [45, 45, 60, 60, 45],
        color="black",
        linewidth=0.9,
        transform=ccrs.PlateCarree(),
        zorder=5
    )

    ax.add_feature(
        cfeature.LAND,
        facecolor="#D9D9D9",
        zorder=6
    )

    ax.coastlines(
        resolution="110m",
        linewidth=0.5
    )

    ax.set_extent(
        [-80, 20, 20, 75],
        crs=ccrs.PlateCarree()
    )

    name = model + ("*" if model == "FGOALS-g3" else "")

    ax.set_title(
        f"({chr(97+i)}) {name}",
        loc="left",
        fontsize=10
    )

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.35,
        color="#BDBDBD",
        alpha=0.6,
        linestyle="--"
    )

    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = (i % 4 == 0)
    gl.bottom_labels = (i >= 8)
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

fig.subplots_adjust(
    left=0.05,
    right=0.96,
    top=0.92,
    bottom=0.13,
    wspace=0.08,
    hspace=0.15
)

cax = fig.add_axes(
    [0.20, 0.065, 0.60, 0.014]
)

cbar = fig.colorbar(
    image,
    cax=cax,
    orientation="horizontal",
    extend="both"
)

cbar.set_label(
    "Surface heat flux trend (W/m²/decade)"
)

fig.suptitle(
    "North Atlantic surface heat flux trends (1950–2014)",
    fontsize=15,
    y=0.97
)

for ext in ("png", "pdf"):
    fig.savefig(
        OUT / f"Fig11_hfds_spatial_trends.{ext}",
        dpi=DPI,
        bbox_inches="tight"
    )

plt.show()
plt.close(fig)


# ============================================================
# 8. Figure 12: SPNA mean hfds trends
# ============================================================

fig, ax = plt.subplots(figsize=(8.5, 7))

positions = np.arange(len(MODELS))

for i, model in enumerate(MODELS):

    subset = surface_change_df.loc[
        surface_change_df["model"] == model
    ]

    vals = subset[
        "hfds_trend_Wm2_decade"
    ].to_numpy(dtype=float)

    mean = vals.mean()
    sd = vals.std(ddof=1)

    color = RED if mean > 0 else BLUE

    ax.scatter(
        vals,
        i + np.linspace(-0.14, 0.14, len(vals)),
        s=20,
        c=[RED if v > 0 else BLUE for v in vals],
        alpha=0.5,
        zorder=3
    )

    ax.errorbar(
        mean,
        i,
        xerr=sd,
        fmt="o",
        markersize=6,
        color=color,
        ecolor=color,
        elinewidth=1.2,
        capsize=3,
        zorder=5
    )

ax.axvline(
    0,
    color=BLACK,
    linestyle="--",
    linewidth=0.9
)

ax.set_yticks(positions)
ax.set_yticklabels([
    m + ("*" if m == "FGOALS-g3" else "")
    for m in MODELS
])

ax.invert_yaxis()

ax.set_xlabel(
    "SPNA surface heat flux trend (W/m²/decade)"
)

ax.set_title(
    "SPNA surface heat flux trends (1950–2014)",
    fontsize=13
)

ax.grid(
    axis="x",
    linestyle="--",
    color="#DDDDDD",
    linewidth=0.6
)

ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("#555555")
    spine.set_linewidth(0.7)

fig.subplots_adjust(
    left=0.25,
    right=0.96,
    top=0.91,
    bottom=0.12
)

for ext in ("png", "pdf"):
    fig.savefig(
        OUT / f"Fig12_SPNA_hfds_trends.{ext}",
        dpi=DPI,
        bbox_inches="tight"
    )

plt.show()
plt.close(fig)


# ============================================================
# 9. Figure 13: Order-of-magnitude comparison
# ============================================================

amoc_df = pd.read_csv(AMOC_FILE)

required_columns = {
    "model", "member", "year", "amoc_Sv"
}

if not required_columns.issubset(amoc_df.columns):
    raise ValueError(
        f"AMOC CSV missing columns: "
        f"{required_columns - set(amoc_df.columns)}"
    )

amoc_change_rows = []

for (model, member), group in amoc_df.groupby(
    ["model", "member"]
):

    early = group.loc[
        group["year"].between(*EARLY),
        "amoc_Sv"
    ]

    late = group.loc[
        group["year"].between(*LATE),
        "amoc_Sv"
    ]

    if len(early) != 20 or len(late) != 20:
        raise ValueError(
            f"Incomplete AMOC periods: {model}/{member}"
        )

    delta_amoc = late.mean() - early.mean()

    amoc_change_rows.append({
        "model": model,
        "member": member,
        "delta_amoc_Sv": delta_amoc,
    })

amoc_change_df = pd.DataFrame(amoc_change_rows)

combined = surface_change_df.merge(
    amoc_change_df,
    on=["model", "member"],
    how="inner",
    validate="one_to_one"
)

if len(combined) != 60:
    raise ValueError(
        f"Expected 60 matched members, found {len(combined)}"
    )


# ============================================================
# Parameterized AMOC-related heat-transport scale
# ============================================================
# For each ensemble member:
#   delta_AMOC = mean(1995-2014) - mean(1950-1969)  [Sv]
#   delta_MHT_est = 0.054 [PW/Sv] * delta_AMOC [Sv] [PW]
#
# Source: Wang et al. (2023), Earth and Space Science,
#         https://doi.org/10.1029/2023EA002887
# 0.054 PW/Sv is the published CMIP6 ensemble-mean regression
# slope of annual MHT onto AMOC strength at 26.5 N.
#
# IMPORTANT: This annual/interannual statistical sensitivity is
# assumed here to approximate the heat-transport scale associated
# with the MULTIDECADAL AMOC difference. The coefficient is fixed
# for all 12 models: coefficient uncertainty is NOT estimated.
# Estimated transport at 26.5 N is NOT the SPNA box's actual net
# ocean heat convergence. Figure 13 is a magnitude/sign check,
# NOT a closed heat budget or an AMOC causal attribution.

combined["delta_mht_est_PW"] = (
    MHT_AMOC_COEFF * combined["delta_amoc_Sv"]
)
combined["mht_amoc_coeff_PW_per_Sv"] = MHT_AMOC_COEFF

if combined[["delta_surface_PW", "delta_amoc_Sv",
             "delta_mht_est_PW"]].isna().any().any():
    raise ValueError("NaNs in Figure 13 member changes")

combined.to_csv(
    OUT / "Fig13_member_heat_changes.csv", index=False
)

# The +/- 1 SD bars in BOTH panels are ensemble-member spreads.
# No arbitrary 5--15 K temperature interval is used.
summary = combined.groupby("model", as_index=False).agg(
    surface_mean=("delta_surface_PW", "mean"),
    surface_sd=("delta_surface_PW", "std"),
    amoc_mean=("delta_amoc_Sv", "mean"),
    amoc_sd=("delta_amoc_Sv", "std"),
    mht_est_mean=("delta_mht_est_PW", "mean"),
    mht_est_sd=("delta_mht_est_PW", "std"),
    n_members=("member", "count"),
)
summary = summary.set_index("model").loc[MODELS].reset_index()
if not (summary["n_members"] == 5).all():
    raise ValueError("Figure 13: 5 paired members per model required")
summary["mht_amoc_coeff_PW_per_Sv"] = MHT_AMOC_COEFF
summary.to_csv(OUT / "Fig13_model_heat_changes.csv", index=False)

print("Figure 13 parameterization: 0.054 PW/Sv (Wang et al., 2023)")
print(summary[["model", "surface_mean", "surface_sd",
               "amoc_mean", "mht_est_mean", "mht_est_sd"]].to_string(index=False))


# ------------------------------------------------------------
# Plot Figure 13
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 7),
    sharey=True
)

ax1, ax2 = axes

for i, row in summary.iterrows():

    # (a) Surface heat exchange
    ax1.errorbar(
        row["surface_mean"],
        i,
        xerr=row["surface_sd"],
        fmt="s",
        color=BLACK,
        markersize=5,
        elinewidth=1,
        capsize=3,
        zorder=3
    )

    # (b) Literature-calibrated AMOC-linked MHT scale.
    # Horizontal bar: between-member +/- 1 SD, not delta-T sensitivity.
    scale = row["mht_est_mean"]
    scale_sd = row["mht_est_sd"]
    color = RED if scale > 0 else BLUE

    ax2.errorbar(
        scale, i, xerr=scale_sd,
        fmt="o", markersize=6,
        color=color, ecolor=color,
        elinewidth=1.2, capsize=3, zorder=3
    )

for ax in axes:
    ax.axvline(
        0,
        color=BLACK,
        linestyle="--",
        linewidth=0.9
    )

    ax.grid(
        axis="x",
        linestyle="--",
        color="#DDDDDD",
        linewidth=0.6
    )

    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#555555")
        spine.set_linewidth(0.7)

axes[0].set_yticks(
    np.arange(len(MODELS))
)

axes[0].set_yticklabels([
    m + ("*" if m == "FGOALS-g3" else "")
    for m in MODELS
])

axes[0].invert_yaxis()

ax1.set_title(
    "(a) Surface heat exchange change",
    fontsize=12
)

ax2.set_title(
    "(b) Estimated AMOC-linked MHT change",
    fontsize=12
)

ax1.set_xlabel(
    "Surface heat exchange change (PW)"
)

ax2.set_xlabel(
    "Estimated MHT change (PW)"
)

fig.suptitle(
    "Surface heat exchange and estimated AMOC-linked MHT changes",
    fontsize=14,
    y=0.98
)

fig.subplots_adjust(
    left=0.20,
    right=0.97,
    top=0.89,
    bottom=0.12,
    wspace=0.17
)

for ext in ("png", "pdf"):
    fig.savefig(
        OUT / f"Fig13_heat_transport_scales.{ext}",
        dpi=DPI,
        bbox_inches="tight"
    )

plt.show()
plt.close(fig)


print("\nFigure 13: Panel (a) directly integrated SPNA hfds; ")
print("Panel (b) 0.054 PW/Sv times 26.5N AMOC change; both bars +/- 1 SD")
print("Caution: MHT scale is NOT an SPNA regional heat-budget closure.")
print("\nFinished all three figures.")
print("Output directory:", OUT)


# ============================================================
# 6. Save Figure 11 reusable gridded data
# ============================================================

ds_plot = xr.Dataset(
    {
        "hfds_trend": (
            ("model", "lat", "lon"),
            np.stack([
                map_results[m]["trend"]
                for m in MODELS
            ])
        ),
        "pvalue": (
            ("model", "lat", "lon"),
            np.stack([
                map_results[m]["pvalue"]
                for m in MODELS
            ])
        )
    },
    coords={
        "model": MODELS,
        "lat": MAP_LAT,
        "lon": MAP_LON,
    },
    attrs={
        "period": "1950–2014",
        "trend_units": "W/m2/decade",
        "SPNA": "45–60N, 60–10W",
        "test": "Two-sided ordinary OLS",
        "remapping": "Nearest neighbor to common 2-degree grid",
    }
)

ds_plot.to_netcdf(
    OUT / "Fig11_hfds_plot_data.nc"
)


# ============================================================
# 7. Figure 11: North Atlantic hfds trend maps
# ============================================================

all_trends = np.stack([
    map_results[m]["trend"] for m in MODELS
])

# Same symmetric color limits for all maps
abs_limit = np.nanpercentile(
    np.abs(all_trends), 98
)

abs_limit = max(
    1.0,
    np.ceil(abs_limit)
)

norm = mcolors.TwoSlopeNorm(
    vmin=-abs_limit,
    vcenter=0,
    vmax=abs_limit
)

fig, axes = plt.subplots(
    3, 4,
    figsize=(17, 9),
    subplot_kw={
        "projection": ccrs.PlateCarree()
    }
)

for i, model in enumerate(MODELS):

    ax = axes.flat[i]

    trend = map_results[model]["trend"]
    pval = map_results[model]["pvalue"]

    image = ax.pcolormesh(
        LON2D,
        LAT2D,
        trend,
        cmap="RdBu_r",
        norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
        rasterized=True,
    )

    # Sparse stippling for p < 0.05
    rr, cc = np.where(
        np.isfinite(pval)
        & (pval < 0.05)
        & np.isfinite(trend)
    )

    keep = (
        (rr % 2 == 0)
        & (cc % 2 == 0)
    )

    ax.scatter(
        LON2D[rr[keep], cc[keep]],
        LAT2D[rr[keep], cc[keep]],
        s=1.5,
        color="black",
        marker=".",
        transform=ccrs.PlateCarree(),
        zorder=4
    )

    # SPNA rectangle
    ax.plot(
        [-60, -10, -10, -60, -60],
        [45, 45, 60, 60, 45],
        color="black",
        linewidth=0.9,
        transform=ccrs.PlateCarree(),
        zorder=5
    )

    ax.add_feature(
        cfeature.LAND,
        facecolor="#D9D9D9",
        zorder=6
    )

    ax.coastlines(
        resolution="110m",
        linewidth=0.5
    )

    ax.set_extent(
        [-80, 20, 20, 75],
        crs=ccrs.PlateCarree()
    )
        # 显示完整的四周地图边框
    ax.spines["geo"].set_visible(True)
    ax.spines["geo"].set_edgecolor("black")
    ax.spines["geo"].set_linewidth(0.8)
    ax.spines["geo"].set_zorder(20)

    # 避免其他边框设置干扰
    for name in ["left", "right", "top", "bottom"]:
        ax.spines[name].set_visible(False)

    name = model 

    ax.set_title(
        f"({chr(97+i)}) {name}",
        loc="left",
        fontsize=10
    )

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.35,
        color="#BDBDBD",
        alpha=0.6,
        linestyle="--"
    )

    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = (i % 4 == 0)
    gl.bottom_labels = (i >= 8)
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

fig.subplots_adjust(
    left=0.05,
    right=0.96,
    top=0.92,
    bottom=0.13,
    wspace=0.08,
    hspace=0.15
)

cax = fig.add_axes(
    [0.20, 0.065, 0.60, 0.014]
)

cbar = fig.colorbar(
    image,
    cax=cax,
    orientation="horizontal",
    extend="both"
)

cbar.set_label(
    r"Surface heat flux trend ($\mathrm{W\,m^{-2}\,decade^{-1}}$)"
)

fig.suptitle(
    "North Atlantic surface heat flux trends (1950–2014)",
    fontsize=20,
    y=0.97
)

for ext in ("png", "pdf"):
    fig.savefig(
        OUT / f"Fig11_hfds_spatial_trends.{ext}",
        dpi=DPI,
        bbox_inches="tight"
    )

plt.show()
plt.close(fig)



# ============================================================
# 8. Figure 12: SPNA mean hfds trends
# ============================================================

fig, ax = plt.subplots(figsize=(8.5, 7))

positions = np.arange(len(MODELS))

for i, model in enumerate(MODELS):

    subset = surface_change_df.loc[
        surface_change_df["model"] == model
    ]

    vals = subset[
        "hfds_trend_Wm2_decade"
    ].to_numpy(dtype=float)

    mean = vals.mean()
    sd = vals.std(ddof=1)

    color = RED if mean > 0 else BLUE

    ax.scatter(
        vals,
        i + np.linspace(-0.14, 0.14, len(vals)),
        s=20,
        c=[RED if v > 0 else BLUE for v in vals],
        alpha=0.5,
        zorder=3
    )

    ax.errorbar(
        mean,
        i,
        xerr=sd,
        fmt="o",
        markersize=6,
        color=color,
        ecolor=color,
        elinewidth=1.2,
        capsize=3,
        zorder=5
    )

ax.axvline(
    0,
    color=BLACK,
    linestyle="--",
    linewidth=0.9
)

ax.set_yticks(positions)
ax.set_yticklabels([
    m 
    for m in MODELS
])

ax.invert_yaxis()

ax.set_xlabel(
    r"($\mathrm{W\,m^{-2}\,decade^{-1}}$)"
)

ax.set_title(
    "SPNA surface heat flux trends (1950–2014)",
    fontsize=13
)

ax.grid(
    axis="x",
    linestyle="--",
    color="#DDDDDD",
    linewidth=0.6
)

ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("#555555")
    spine.set_linewidth(0.7)

fig.subplots_adjust(
    left=0.25,
    right=0.96,
    top=0.91,
    bottom=0.12
)

for ext in ("png", "pdf"):
    fig.savefig(
        OUT / f"Fig12_SPNA_hfds_trends.{ext}",
        dpi=DPI,
        bbox_inches="tight"
    )

plt.show()
plt.close(fig)

# ------------------------------------------------------------
# Plot Figure 13
# ------------------------------------------------------------
from matplotlib.ticker import FuncFormatter
fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 7),
    sharey=True
)

ax1, ax2 = axes

for i, row in summary.iterrows():

    # (a) Surface heat exchange
    ax1.errorbar(
        row["surface_mean"],
        i,
        xerr=row["surface_sd"],
        fmt="s",
        color=BLACK,
        ecolor=BLACK,
        markersize=5,
        elinewidth=1,
        capsize=3,
        zorder=3
    )

    # (b) Literature-calibrated AMOC-linked MHT scale
    # Error bar: between-member +/- 1 SD
    scale = row["mht_est_mean"]
    scale_sd = row["mht_est_sd"]
    color = RED if scale > 0 else BLUE

    ax2.errorbar(
        scale,
        i,
        xerr=scale_sd,
        fmt="o",
        markersize=6,
        color=color,
        ecolor=color,
        elinewidth=1.2,
        capsize=3,
        zorder=3
    )


# ------------------------------------------------------------
# Common x-axis settings
# ------------------------------------------------------------

# 左右两图使用相同的横坐标范围
XMIN = -0.14
XMAX = 0.08

# 相同的刻度间隔
XTICKS = np.arange(-0.125, 0.076, 0.025)

for ax in axes:

    ax.set_xlim(XMIN, XMAX)
    ax.set_xticks(XTICKS)
    ax.xaxis.set_major_formatter(
    FuncFormatter(lambda x, pos: "0" if abs(x) < 1e-10 else f"{x:.3f}")
)
    # Zero reference line
    ax.axvline(
        0,
        color=BLACK,
        linestyle="--",
        linewidth=0.9,
        zorder=2
    )

    # Grid
    ax.grid(
        axis="x",
        linestyle="--",
        color="#DDDDDD",
        linewidth=0.6
    )

    ax.set_axisbelow(True)

    # Full panel borders
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#555555")
        spine.set_linewidth(0.7)

    # 只保留横坐标单位
    ax.set_xlabel(
        "(PW)",
        fontsize=11
    )

    ax.tick_params(
        axis="both",
        labelsize=9,
        direction="out"
    )


# ------------------------------------------------------------
# Model labels
# ------------------------------------------------------------

ax1.set_yticks(
    np.arange(len(MODELS))
)

ax1.set_yticklabels([
    m 
    for m in MODELS
])

ax1.invert_yaxis()

# 右图不显示重复的模式标签
ax2.tick_params(
    axis="y",
    labelleft=False
)


# ------------------------------------------------------------
# Centered panel titles
# ------------------------------------------------------------

ax1.set_title(
    "(a) Surface heat exchange change",
    fontsize=12,
    loc="center",
    pad=10
)

ax2.set_title(
    "(b) Estimated AMOC-linked MHT change",
    fontsize=12,
    loc="center",
    pad=10
)

# 删除原来的 fig.suptitle()


# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------

fig.subplots_adjust(
    left=0.20,
    right=0.97,
    top=0.92,
    bottom=0.12,
    wspace=0.17
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

for ext in ("png", "pdf"):
    fig.savefig(
        OUT / f"Fig13_heat_transport_scales.{ext}",
        dpi=DPI,
        bbox_inches="tight"
    )

plt.show()
plt.close(fig)


print("\nFigure 13: Panel (a) directly integrated SPNA hfds.")
print("Panel (b): 0.054 PW/Sv times 26.5N AMOC change.")
print("Both error bars represent between-member ±1 SD.")
print("Caution: This is NOT an SPNA regional heat-budget closure.")
print("Output directory:", OUT)