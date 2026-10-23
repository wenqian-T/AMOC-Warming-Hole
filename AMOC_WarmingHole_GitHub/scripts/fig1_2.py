from pathlib import Path
import os
_PROJECT_START = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
PROJECT_ROOT = next((p for p in [_PROJECT_START, *_PROJECT_START.parents] if (p / "PROJECT_ROOT.marker").is_file()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Run from the project folder or its notebooks/scripts subfolder.")



# ============================================================
# CMIP6 Historical | Final revised Figure 1 and Figure 2
# Supports (year, cell) and (year, y, x)
# ============================================================

from pathlib import Path
import re
import gc
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from scipy.spatial import cKDTree
from scipy.stats import t as t_dist


# ============================================================
# 1. CONFIG
# ============================================================

ROOT = Path(str(PROJECT_ROOT / 'data'))
ANNUAL_DIR = ROOT / "historical_analysis/annual"
CMIP6_DIR = ROOT / "CMIP6"
OBS_DIR = ROOT / "OBS"

OUT = Path(str(PROJECT_ROOT / 'results/figures'))
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

START, END = 1950, 2014
YEARS = np.arange(START, END + 1)
NYEAR = len(YEARS)

SPNA = (-60, -10, 45, 60)  # W, E, S, N
MAP_EXTENT = (-80, 20, 20, 75)

CMAP = "RdBu_r"
VMIN, VMAX = -0.4, 0.4
P_THRESHOLD = 0.05

REGRID_DEG = 2.0
MAX_DISTANCE_DEG = 2.5

# Only one ERSSTv5 file should be selected.
# If this directory contains multiple NC files, specify the exact path.
OBS_FILES = {
    "ERSSTv5": OBS_DIR / "ERSSTv5",
    "HadISST": OBS_DIR / "HadISST/HadISST_sst.nc",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 10,
    "savefig.dpi": 300,
})


# ============================================================
# 2. DISCOVER FILES
# ============================================================

def sort_member(s):
    m = re.fullmatch(r"r(\d+)i(\d+)p(\d+)f(\d+)", s)
    return tuple(map(int, m.groups()))


MEMBER_FILES = {}
MEMBERS = {}

for model in MODELS:
    files = {}

    for p in ANNUAL_DIR.glob(f"{model}_*_annual.nc"):
        if p.name.startswith("._"):
            continue

        member = p.name[len(model) + 1:-len("_annual.nc")]

        if re.fullmatch(r"r\d+i\d+p\d+f\d+", member):
            files[member] = p

    members = sorted(files, key=sort_member)

    if len(members) != 5:
        raise ValueError(
            f"{model}: expected 5 members, found {members}"
        )

    MEMBER_FILES[model] = files
    MEMBERS[model] = members
    print(model, members, flush=True)


def resolve_obs(path):
    if path.is_file():
        return path

    if not path.is_dir():
        raise FileNotFoundError(path)

    files = [
        p for p in sorted(path.rglob("*.nc"))
        if not p.name.startswith("._")
    ]

    if len(files) != 1:
        raise ValueError(
            f"Observation directory {path} contains "
            f"{len(files)} NetCDF files. Set the exact file path.\n"
            + "\n".join(map(str, files[:20]))
        )

    return files[0]


OBS_FILES = {
    k: resolve_obs(v) for k, v in OBS_FILES.items()
}


# ============================================================
# 3. COORDINATES, TIME AND SST
# ============================================================

def find_var(ds):
    for name in (
        "tos", "tos_annual", "sst", "sst_annual",
        "SST", "SST_annual"
    ):
        if name in ds:
            return ds[name]

    raise KeyError(
        f"No SST variable in file. Variables: {list(ds.data_vars)}"
    )


def get_time_info(da):
    if "year" in da.dims:
        tdim = "year"
        yy = np.asarray(da["year"].values, dtype=int)
    elif "time" in da.dims:
        tdim = "time"
        yy = np.asarray(da["time"].dt.year.values, dtype=int)
    else:
        raise ValueError(f"No year/time dimension: {da.dims}")

    return tdim, yy


def get_latlon(ds, shape):
    """
    Return spatial coordinates matching SST shape.

    Supported:
      SST(year, cell)   -> lat(cell), lon(cell)
      SST(year, y, x)   -> lat(y,x), lon(y,x)
      SST(year, lat,lon)-> lat(lat), lon(lon)
    """

    lat_name = next(
        (
            n for n in
            ("lat", "latitude", "nav_lat", "TLAT")
            if n in ds
        ),
        None
    )

    lon_name = next(
        (
            n for n in
            ("lon", "longitude", "nav_lon", "TLONG")
            if n in ds
        ),
        None
    )

    if lat_name is None or lon_name is None:
        raise KeyError(
            "Missing latitude/longitude. "
            f"Available variables: {list(ds.variables)}"
        )

    lat = np.asarray(ds[lat_name].values, dtype=float)
    lon = np.asarray(ds[lon_name].values, dtype=float)
    lon = (lon + 180) % 360 - 180

    if lat.shape == shape and lon.shape == shape:
        return lat, lon

    if (
        len(shape) == 2
        and lat.ndim == 1
        and lon.ndim == 1
    ):
        xx, yy = np.meshgrid(lon, lat)
        if yy.shape == shape:
            return yy, xx

    raise ValueError(
        f"Coordinate mismatch: SST={shape}, "
        f"lat={lat.shape}, lon={lon.shape}"
    )


def select_annual_sst(path):
    """
    Open without loading full SST into RAM.
    Returns dataset, transposed variable, annual indices,
    flattened lat/lon, and Kelvin flag.
    """

    ds = xr.open_dataset(path, decode_times=True)

    try:
        da = find_var(ds)
        tdim, yy = get_time_info(da)

        spatial_dims = [d for d in da.dims if d != tdim]

        if len(spatial_dims) not in (1, 2):
            raise ValueError(
                f"Unsupported spatial dimensions: {da.dims}"
            )

        da = da.transpose(tdim, *spatial_dims)

        indices = []
        for year in YEARS:
            loc = np.flatnonzero(yy == year)
            if len(loc) != 1:
                raise ValueError(
                    f"{path.name}: year {year} has "
                    f"{len(loc)} records; expected one annual field"
                )
            indices.append(int(loc[0]))

        lat, lon = get_latlon(ds, da.shape[1:])

        units = str(da.attrs.get("units", "")).lower()
        is_kelvin = units in ("k", "kelvin")

        return (
            ds, da, tdim, indices,
            lat.ravel(), lon.ravel(), is_kelvin
        )

    except Exception:
        ds.close()
        raise


def read_one_year(da, tdim, index, is_kelvin):
    """Read one annual field as flat 1D data."""

    field = np.asarray(
        da.isel({tdim: index}).values,
        dtype=np.float64
    ).ravel()

    field[~np.isfinite(field)] = np.nan
    field[np.abs(field) > 1e19] = np.nan

    if is_kelvin:
        field -= 273.15

    return field


# ============================================================
# 4. AREA MATCHING
# ============================================================

def xyz(lat, lon):
    lat = np.deg2rad(np.asarray(lat))
    lon = np.deg2rad(np.asarray(lon))

    return np.column_stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat)
    ])


def area_candidates(model, member):
    base = CMIP6_DIR / model / "historical"

    own = list(
        (base / member / "areacello").glob(
            "**/areacello_Ofx_*.nc"
        )
    )

    others = list(
        base.glob("*/areacello/**/areacello_Ofx_*.nc")
    )

    paths = []
    for p in own + sorted(others):
        if p.name.startswith("._"):
            continue
        if p not in paths:
            paths.append(p)

    return paths


def area_from_annual(ds, target_size):
    """Prefer cell-aligned area already stored in annual file."""

    for name in ("cell_area", "areacello", "area"):
        if name in ds:
            arr = np.asarray(ds[name].values, dtype=float)
            if arr.size == target_size:
                return arr.ravel()

    return None


def match_area(model, member, ds, lat, lon):
    """
    Match areacello to SST cell coordinates.

    For flattened cell data, use coordinate matching,
    not blind area.ravel().

    Accept only near-exact geographic matches.
    """

    existing = area_from_annual(ds, len(lat))
    if existing is not None:
        return existing

    target_good = np.isfinite(lat) & np.isfinite(lon)

    if not target_good.all():
        raise ValueError(
            f"{model}: SST cell coordinates contain NaNs"
        )

    target_xyz = xyz(lat, lon)

    for path in area_candidates(model, member):
        try:
            with xr.open_dataset(path) as ads:
                area_da = ads["areacello"]
                area = np.asarray(area_da.values, dtype=float)

                alat, alon = get_latlon(ads, area.shape)

                area = area.ravel()
                alat = alat.ravel()
                alon = alon.ravel()

                # Prefer direct alignment when fully consistent.
                if area.size == lat.size:
                    dlat = np.abs(alat - lat)
                    dlon = np.abs(
                        (alon - lon + 180) % 360 - 180
                    )

                    if (
                        np.nanmax(dlat) < 1e-4
                        and np.nanmax(dlon) < 1e-4
                    ):
                        print(
                            f"  AREA direct match: {path.name}",
                            flush=True
                        )
                        return area

                # Geographical matching handles flattened cells
                # whose ordering differs from native 2D grid.
                good = (
                    np.isfinite(area)
                    & (area > 0)
                    & np.isfinite(alat)
                    & np.isfinite(alon)
                )

                tree = cKDTree(xyz(alat[good], alon[good]))
                dist, ix = tree.query(target_xyz)

                # Angular tolerance 0.001 degree.
                tolerance = 2 * np.sin(
                    np.deg2rad(0.001) / 2
                )

                if np.all(dist < tolerance):
                    selected = np.flatnonzero(good)[ix]

                    if len(np.unique(selected)) != len(selected):
                        continue

                    print(
                        f"  AREA coordinate match: {path.name}",
                        flush=True
                    )

                    return area[selected]

        except (OSError, KeyError, ValueError) as e:
            warnings.warn(f"Skipping {path.name}: {e}")

    raise ValueError(
        f"{model} {member}: no compatible areacello. "
        "Neither exact native-grid nor cell-coordinate "
        "matching succeeded. Check annual file metadata."
    )


# ============================================================
# 5. INDICES AND OLS
# ============================================================

def spna_mask(lat, lon):
    return (
        (lat >= SPNA[2])
        & (lat <= SPNA[3])
        & (lon >= SPNA[0])
        & (lon <= SPNA[1])
    )


def area_mean(field, area, region=None):
    valid = (
        np.isfinite(field)
        & np.isfinite(area)
        & (area > 0)
    )

    if region is not None:
        valid &= region

    if not valid.any():
        return np.nan

    return np.sum(
        field[valid] * area[valid]
    ) / np.sum(area[valid])


def slope_1d(y):
    y = np.asarray(y, dtype=float)

    if len(y) != NYEAR or not np.isfinite(y).all():
        raise ValueError("Incomplete annual SST index")

    x = YEARS.astype(float)
    x -= x.mean()

    return np.sum(x * y) / np.sum(x*x) * 10


def ols_spatial(data):
    """
    data: (year, selected_cells)
    Returns 1D slope (degC/decade) and two-sided p-value.
    Ordinary OLS without autocorrelation/FDR correction.
    """

    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2 or data.shape[0] != NYEAR:
        raise ValueError(
            f"Unexpected trend-array shape: {data.shape}"
        )

    good = np.isfinite(data).all(axis=0)

    x = YEARS.astype(float)
    x -= x.mean()

    y = np.where(np.isfinite(data), data, 0.0)
    mean_y = y.mean(axis=0)

    sxx = np.sum(x*x)
    slope = np.sum(
        x[:, None] * (y - mean_y),
        axis=0
    ) / sxx

    residual = y - (
        mean_y[None, :] + x[:, None] * slope[None, :]
    )

    sse = np.sum(residual**2, axis=0)

    dof = NYEAR - 2

    with np.errstate(divide="ignore", invalid="ignore"):
        stderr = np.sqrt((sse/dof)/sxx)
        tvalue = slope/stderr
        p = 2*t_dist.sf(np.abs(tvalue), dof)

    # Perfectly constant fields: zero slope, p=1.
    constant = good & (sse == 0) & (slope == 0)
    p[constant] = 1.0

    slope[~good] = np.nan
    p[~good] = np.nan

    return slope*10, p


# ============================================================
# 6. MAP REMAPPING
# ============================================================

TARGET_LON = np.arange(-80, 20.1, REGRID_DEG)
TARGET_LAT = np.arange(20, 75.1, REGRID_DEG)

LON, LAT = np.meshgrid(TARGET_LON, TARGET_LAT)


def remap(values, lat, lon):
    values = np.asarray(values).ravel()
    lat = np.asarray(lat).ravel()
    lon = np.asarray(lon).ravel()

    good = (
        np.isfinite(values)
        & np.isfinite(lat)
        & np.isfinite(lon)
    )

    output = np.full(LON.size, np.nan)

    if not good.any():
        return output.reshape(LON.shape)

    tree = cKDTree(xyz(lat[good], lon[good]))
    dist, index = tree.query(
        xyz(LAT.ravel(), LON.ravel())
    )

    max_dist = 2*np.sin(
        np.deg2rad(MAX_DISTANCE_DEG)/2
    )

    keep = dist <= max_dist
    output[keep] = values[good][index[keep]]

    return output.reshape(LON.shape)


# ============================================================
# 7. PROCESS MODELS — LOW MEMORY
# ============================================================

member_rows = []
model_maps = []

for model in MODELS:
    print(f"\nProcessing {model}", flush=True)

    ens_sum = None
    ens_count = None
    ref_lat = None
    ref_lon = None
    select = None

    for member in MEMBERS[model]:
        path = MEMBER_FILES[model][member]

        print(" ", member, flush=True)

        (
            ds, da, tdim, indices,
            lat, lon, kelvin
        ) = select_annual_sst(path)

        try:
            if ref_lat is None:
                ref_lat = lat.copy()
                ref_lon = lon.copy()

                select = np.flatnonzero(
                    (lat >= MAP_EXTENT[2]-2)
                    & (lat <= MAP_EXTENT[3]+2)
                    & (lon >= MAP_EXTENT[0]-2)
                    & (lon <= MAP_EXTENT[1]+2)
                )

                if len(select) == 0:
                    raise ValueError(
                        f"{model}: no cells in North Atlantic"
                    )

                ens_sum = np.zeros(
                    (NYEAR, len(select)),
                    dtype=np.float32
                )
                ens_count = np.zeros(
                    (NYEAR, len(select)),
                    dtype=np.uint8
                )

            else:
                if not (
                    np.allclose(lat, ref_lat, equal_nan=True)
                    and np.allclose(lon, ref_lon, equal_nan=True)
                ):
                    raise ValueError(
                        f"{model}: SST cell grids differ "
                        "between members. Regridding is required."
                    )

            area = match_area(
                model, member, ds, lat, lon
            )
            mask = spna_mask(lat, lon)

            regional = np.empty(NYEAR)
            global_ocean = np.empty(NYEAR)

            for k, index in enumerate(indices):
                field = read_one_year(
                    da, tdim, index, kelvin
                )

                regional[k] = area_mean(
                    field, area, mask
                )
                global_ocean[k] = area_mean(
                    field, area
                )

                part = field[select]
                valid = np.isfinite(part)

                ens_sum[k] += np.where(
                    valid, part, 0
                ).astype(np.float32)

                ens_count[k] += valid.astype(np.uint8)

                del field

            indices_dict = {
                "SPNA": regional,
                "Global": global_ocean,
                "Global-SPNA": global_ocean-regional
            }

            for index_name, values in indices_dict.items():
                member_rows.append({
                    "model": model,
                    "member": member,
                    "index": index_name,
                    "trend": slope_1d(values)
                })

        finally:
            ds.close()
            del ds, da
            gc.collect()

    n_members = len(MEMBERS[model])

    with np.errstate(divide="ignore", invalid="ignore"):
        ensemble = ens_sum / ens_count

    ensemble[ens_count != n_members] = np.nan

    slope, pval = ols_spatial(ensemble)

    model_maps.append({
        "name": model,
        "trend": remap(
            slope, ref_lat[select], ref_lon[select]
        ),
        "pval": remap(
            pval, ref_lat[select], ref_lon[select]
        )
    })

    del ens_sum, ens_count, ensemble
    del slope, pval
    gc.collect()

    print(" Finished", model, flush=True)


# ============================================================
# 8. PROCESS OBSERVATIONS
# ============================================================

def process_obs(name, path):
    print("Observation:", name, flush=True)

    with xr.open_dataset(path, decode_times=True) as ds:
        da = find_var(ds)
        tdim, input_years = get_time_info(da)

        spatial_dims = [d for d in da.dims if d != tdim]
        da = da.transpose(tdim, *spatial_dims)

        lat, lon = get_latlon(ds, da.shape[1:])
        lat = lat.ravel()
        lon = lon.ravel()

        units = str(da.attrs.get("units", "")).lower()
        kelvin = units in ("k", "kelvin")

        if tdim == "time":
            months = np.asarray(
                ds["time"].dt.month.values
            )
            days = np.asarray(
                ds["time"].dt.days_in_month.values,
                dtype=float
            )

        # Cosine-latitude weights for regular obs grids.
        area = np.cos(np.deg2rad(lat))
        area = np.maximum(area, 0)

        region = spna_mask(lat, lon)

        selected = np.flatnonzero(
            (lat >= MAP_EXTENT[2]-2)
            & (lat <= MAP_EXTENT[3]+2)
            & (lon >= MAP_EXTENT[0]-2)
            & (lon <= MAP_EXTENT[1]+2)
        )

        annual_map = np.full(
            (NYEAR, len(selected)),
            np.nan,
            dtype=np.float32
        )

        regional = np.empty(NYEAR)
        global_ocean = np.empty(NYEAR)

        for k, year in enumerate(YEARS):
            ids = np.flatnonzero(input_years == year)

            if len(ids) == 1:
                field = read_one_year(
                    da, tdim, int(ids[0]), kelvin
                )

            elif len(ids) == 12 and tdim == "time":
                if set(months[ids]) != set(range(1, 13)):
                    raise ValueError(
                        f"{name}: incomplete months in {year}"
                    )

                # Monthly fields: weighted annual mean
                total = np.zeros(len(lat), dtype=np.float64)
                count = np.zeros(len(lat), dtype=np.float64)

                for idx in ids:
                    m = read_one_year(
                        da, tdim, int(idx), kelvin
                    )
                    good = np.isfinite(m)
                    total[good] += m[good]*days[idx]
                    count[good] += days[idx]

                with np.errstate(
                    divide="ignore", invalid="ignore"
                ):
                    field = total/count

                field[count == 0] = np.nan

                del total, count

            else:
                raise ValueError(
                    f"{name}: year {year}, "
                    f"{len(ids)} records"
                )

            # Mask typical missing-value sentinels.
            field[(field < -5) | (field > 50)] = np.nan

            regional[k] = area_mean(
                field, area, region
            )
            global_ocean[k] = area_mean(
                field, area
            )

            annual_map[k] = field[selected].astype(
                np.float32
            )

            del field

        slope, pval = ols_spatial(annual_map)

        result = {
            "name": name,
            "trend": remap(
                slope, lat[selected], lon[selected]
            ),
            "pval": remap(
                pval, lat[selected], lon[selected]
            )
        }

        metrics = {
            "SPNA": slope_1d(regional),
            "Global": slope_1d(global_ocean),
            "Global-SPNA": slope_1d(
                global_ocean-regional
            )
        }

        return result, metrics


obs_maps = []
obs_rows = []

for name in ("ERSSTv5", "HadISST"):
    result, metrics = process_obs(
        name, OBS_FILES[name]
    )

    obs_maps.append(result)

    for index_name, value in metrics.items():
        obs_rows.append({
            "dataset": name,
            "index": index_name,
            "trend": value
        })


member_df = pd.DataFrame(member_rows)
obs_df = pd.DataFrame(obs_rows)

member_df.to_csv(
    OUT / "Fig02_SST_member_trends.csv",
    index=False
)

obs_df.to_csv(
    OUT / "Fig02_SST_observation_trends.csv",
    index=False
)


# ============================================================
# 9. FIGURE 1 — 14 MAPS + 2 BLANK PANELS
# ============================================================

all_maps = obs_maps + model_maps

fig, axes = plt.subplots(
    4, 4,
    figsize=(18, 13.5),
    subplot_kw={"projection": ccrs.PlateCarree()},
    constrained_layout=True
)

last_image = None

for n, item in enumerate(all_maps):
    ax = axes.flat[n]

    last_image = ax.pcolormesh(
        LON, LAT, item["trend"],
        cmap=CMAP,
        vmin=VMIN, vmax=VMAX,
        shading="auto",
        transform=ccrs.PlateCarree(),
        rasterized=True
    )

    significant = (
        np.isfinite(item["trend"])
        & np.isfinite(item["pval"])
        & (item["pval"] < P_THRESHOLD)
    )

    rr, cc = np.where(significant)
    sparse = (rr % 2 == 0) & (cc % 2 == 0)

    ax.scatter(
        LON[rr[sparse], cc[sparse]],
        LAT[rr[sparse], cc[sparse]],
        s=2,
        c="black",
        marker=".",
        linewidths=0,
        transform=ccrs.PlateCarree(),
        zorder=4
    )

    # SPNA rectangle
    ax.plot(
        [-60, -10, -10, -60, -60],
        [45, 45, 60, 60, 45],
        color="black",
        lw=1,
        transform=ccrs.PlateCarree(),
        zorder=5
    )

    ax.add_feature(
        cfeature.LAND,
        facecolor="lightgray",
        zorder=3
    )

    ax.coastlines(linewidth=0.5)
    ax.set_extent(
        MAP_EXTENT,
        crs=ccrs.PlateCarree()
    )

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        alpha=0.35,
        linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = n % 4 == 0
    gl.bottom_labels = n >= 10
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    ax.set_title(
        f"({chr(97+n)}) {item['name']}",
        loc="left",
        fontsize=10
    )

# Exactly two empty panels
for n in (14, 15):
    axes.flat[n].set_visible(False)

cbar = fig.colorbar(
    last_image,
    ax=axes.ravel().tolist(),
    orientation="horizontal",
    shrink=0.70,
    fraction=0.035,
    pad=0.035,
    extend="both"
)
cbar.set_label("SST trend (°C/decade)")

fig.suptitle(
    "North Atlantic SST trends (1950–2014)",
    fontsize=15
)

for suffix in ("png", "pdf"):
    fig.savefig(
        OUT / f"Fig01_SST_trends.{suffix}",
        dpi=300,
        bbox_inches="tight"
    )

plt.close(fig)
print("Figure 1 saved", flush=True)


# ============================================================
# 10. FIGURE 2 — SPNA / Global / Global-SPNA
# ============================================================

INDICES = ["SPNA", "Global", "Global-SPNA"]

OBS_COLORS = {
    "ERSSTv5": "#C56B31",
    "HadISST": "#327A98"
}

fig, axes = plt.subplots(
    1, 3,
    figsize=(16, 8),
    sharey=True,
    constrained_layout=True
)

for k, (ax, key) in enumerate(zip(axes, INDICES)):

    for j, model in enumerate(MODELS):

        values = member_df.loc[
            (member_df["model"] == model)
            & (member_df["index"] == key),
            "trend"
        ].to_numpy(dtype=float)

        if len(values) != 5:
            raise ValueError(
                f"{model} {key}: expected five trends"
            )

        mean = values.mean()
        sd = values.std(ddof=1)

        ax.scatter(
            values,
            j + np.linspace(-0.16, 0.16, 5),
            s=16,
            color="#999999",
            alpha=0.8,
            zorder=3
        )

        ax.errorbar(
            mean, j,
            xerr=sd,
            fmt="o",
            markersize=6,
            color="#25384C",
            elinewidth=1.1,
            capsize=3,
            zorder=5
        )

    ax.axvline(
        0,
        color="gray",
        linewidth=0.8
    )

    for name, color in OBS_COLORS.items():
        value = float(
            obs_df.loc[
                (obs_df.dataset == name)
                & (obs_df["index"] == key),
                "trend"
            ].iloc[0]
        )

        ax.axvline(
            value,
            color=color,
            linestyle="--",
            linewidth=1.5,
            label=f"{name}: {value:.3f}"
        )

    ax.set_title(
        f"({chr(97+k)}) {key}",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xlabel("SST trend (°C/decade)")
    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_yticklabels(MODELS)

    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)

    ax.legend(
        loc="lower right",
        fontsize=8,
        frameon=False
    )

fig.suptitle(
    "Regional and global SST trends across CMIP6 models",
    fontsize=14
)

for suffix in ("png", "pdf"):
    fig.savefig(
        OUT / f"Fig02_SST_indices.{suffix}",
        dpi=300,
        bbox_inches="tight"
    )

plt.close(fig)

print("\nCompleted!", flush=True)
print("Output directory:", OUT)




# 绘图数据保存

OUT = Path(str(PROJECT_ROOT / 'results/figures'))
OUT.mkdir(parents=True, exist_ok=True)

# 14组数据：2组观测 + 12个模式
all_maps = obs_maps + model_maps

ds_plot = xr.Dataset(
    data_vars={
        "sst_trend": (
            ("dataset", "lat", "lon"),
            np.stack([item["trend"] for item in all_maps])
        ),
        "pvalue": (
            ("dataset", "lat", "lon"),
            np.stack([item["pval"] for item in all_maps])
        ),
    },
    coords={
        "dataset": [item["name"] for item in all_maps],
        "lat": LAT[:, 0],
        "lon": LON[0, :],
    },
    attrs={
        "period": "1950–2014",
        "trend_units": "degC/decade",
        "test": "Two-sided ordinary OLS",
        "significance_threshold": 0.05,
        "regrid": "Nearest-neighbor to common 2-degree grid",
        "spna_region": "45–60N, 60–10W",
    }
)

output_path = OUT / "Fig01_SST_plot_data.nc"

ds_plot.to_netcdf(output_path)

print("Saved:", output_path)
print(ds_plot)

# ============================================================
# 9. FIGURE 1 — 14 MAPS + 2 BLANK PANELS
# ============================================================

all_maps = obs_maps + model_maps

fig, axes = plt.subplots(
    4, 4,
    figsize=(18, 13.5),
    subplot_kw={"projection": ccrs.PlateCarree()},
    constrained_layout=True
)

last_image = None

for n, item in enumerate(all_maps):
    ax = axes.flat[n]

    last_image = ax.pcolormesh(
        LON, LAT, item["trend"],
        cmap=CMAP,
        vmin=VMIN, vmax=VMAX,
        shading="auto",
        transform=ccrs.PlateCarree(),
        rasterized=True
    )

    significant = (
        np.isfinite(item["trend"])
        & np.isfinite(item["pval"])
        & (item["pval"] < P_THRESHOLD)
    )

    rr, cc = np.where(significant)
    sparse = (rr % 2 == 0) & (cc % 2 == 0)

    ax.scatter(
        LON[rr[sparse], cc[sparse]],
        LAT[rr[sparse], cc[sparse]],
        s=10,
        c="black",
        marker=".",
        linewidths=0,
        transform=ccrs.PlateCarree(),
        zorder=4
    )

    # SPNA rectangle
    ax.plot(
        [-60, -10, -10, -60, -60],
        [45, 45, 60, 60, 45],
        color="black",
        lw=1,
        transform=ccrs.PlateCarree(),
        zorder=5
    )

    ax.add_feature(
        cfeature.LAND,
        facecolor="lightgray",
        zorder=3
    )

    ax.coastlines(linewidth=0.5)
    ax.set_extent(
        MAP_EXTENT,
        crs=ccrs.PlateCarree()
    )

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        alpha=0.35,
        linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = n % 4 == 0
    gl.bottom_labels = n >= 10
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    ax.set_title(
        f"({chr(97+n)}) {item['name']}",
        loc="left",
        fontsize=10
    )

# Exactly two empty panels
for n in (14, 15):
    axes.flat[n].set_visible(False)

cbar = fig.colorbar(
    last_image,
    ax=axes.ravel().tolist(),
    orientation="horizontal",
    shrink=1.0,       # 保持最大长度
    fraction=0.015,   # 从 0.035 降至 0.015，更细
    pad=0.035,
    aspect=60,        # 增大长宽比
    extend="both"
)
cbar.set_label("SST trend (°C/decade)")

fig.suptitle(
    "North Atlantic SST trends (1950–2014)",
    fontsize=25
)

for suffix in ("png", "pdf"):
    fig.savefig(
        OUT / f"Fig01_SST_trends.{suffix}",
        dpi=300,
        bbox_inches="tight"
    )

plt.close(fig)
print("Figure 1 saved", flush=True)


INDICES = ["SPNA", "Global", "Global-SPNA"]

OBS_COLORS = {
    "ERSSTv5": "red",
    "HadISST": "blue"
}

fig, axes = plt.subplots(
    1, 3,
    figsize=(16, 8),
    sharey=True,
    constrained_layout=True
)

for k, (ax, key) in enumerate(zip(axes, INDICES)):

    for j, model in enumerate(MODELS):

        values = member_df.loc[
            (member_df["model"] == model)
            & (member_df["index"] == key),
            "trend"
        ].to_numpy(dtype=float)

        if len(values) != 5:
            raise ValueError(
                f"{model} {key}: expected five trends"
            )

        mean = values.mean()
        sd = values.std(ddof=1)

        ax.scatter(
            values,
            j + np.linspace(-0.16, 0.16, 5),
            s=16,
            color="#999999",
            alpha=0.8,
            zorder=3
        )

        ax.errorbar(
            mean, j,
            xerr=sd,
            fmt="o",
            markersize=6,
            color="#25384C",
            elinewidth=1.1,
            capsize=3,
            zorder=5
        )

    ax.axvline(
        0,
        color="gray",
        linewidth=0.8
    )

    for name, color in OBS_COLORS.items():
        value = float(
            obs_df.loc[
                (obs_df.dataset == name)
                & (obs_df["index"] == key),
                "trend"
            ].iloc[0]
        )

        ax.axvline(
            value,
            color=color,
            linestyle="--",
            linewidth=1.5,
            label=f"{name}: {value:.3f}"
        )

    ax.set_title(
        f"({chr(97+k)}) {key}",
        fontsize=18,
        fontweight="bold"
    )

    ax.set_xlabel("SST trend (°C/decade)")
    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_yticklabels(MODELS)

    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)

    ax.legend(
        loc="upper right",
        fontsize=8,
        frameon=False
    )
'''
fig.suptitle(
    "Regional and global SST trends across CMIP6 models",
    fontsize=14
)
'''

for suffix in ("png", "pdf"):
    fig.savefig(
        OUT / f"Fig02_SST_indices.{suffix}",
        dpi=300,
        bbox_inches="tight"
    )

plt.close(fig)

print("\nCompleted!", flush=True)
print("Output directory:", OUT)