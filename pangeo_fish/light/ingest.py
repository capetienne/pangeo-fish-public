"""Tag CSV ingestion helpers.

Converts raw manufacturer files into the standard
``(time, temperature, pressure, light)`` DataFrame and builds an
``xr.DataTree`` compatible with the rest of the pangeo-fish pipeline.

Supported tag types
-------------------
``"lotek"``
    Lotek LAT2810 — semicolon-separated, comma decimal,
    timestamp format ``%H:%M:%S %d/%m/%y``.

``"wc_psat_daily"``
    Wildlife Computers MiniPAT PSAT — daily *DailyData.csv* with an
    optional 10-minute *Series* CSV that is spliced in for the last
    days before pop-up.
"""

import numpy as np
import pandas as pd
import xarray as xr


def load_tag_csv(
    path,
    tag_type,
    time_correction=None,
    resample_freq=None,
    series_path=None,
):
    """Load a raw manufacturer CSV and return a standardised DataFrame.

    Parameters
    ----------
    path : str or path-like
        Path to the main CSV file.
    tag_type : {"lotek", "wc_psat_daily"}
        Manufacturer / format identifier.
    time_correction : pd.Timedelta or None
        Clock-drift correction added to every timestamp (Lotek only).
    resample_freq : str or None
        Pandas offset alias for resampling, e.g. ``"1min"`` (Lotek only).
        ``None`` keeps the original recording interval.
    series_path : str or path-like or None
        Path to the optional 10-minute Series CSV (WC PSAT only).
        When provided, Series rows replace daily rows from their start
        time onward.

    Returns
    -------
    pd.DataFrame
        Index ``time`` (UTC-naive), columns
        ``temperature``, ``pressure``, ``light``.

    Raises
    ------
    ValueError
        If *tag_type* is not recognised.
    """
    tag_type = tag_type.lower()

    if tag_type == "lotek":
        df_raw = pd.read_csv(
            path,
            sep=";",
            decimal=",",
            names=["TimeS", "LightIntensity", "ExtTemp", "Pressure", "C_TooDimFlag", "_"],
            skiprows=1,
            usecols=[0, 1, 2, 3],
        )
        df_raw = df_raw.dropna(subset=["TimeS"])
        timestamps = pd.to_datetime(
            df_raw["TimeS"].str.strip(), format="%H:%M:%S %d/%m/%y", errors="coerce"
        )
        if time_correction is not None:
            timestamps = timestamps + pd.Timedelta(time_correction)
        df_raw["time"] = timestamps
        df_raw = df_raw.dropna(subset=["time"]).set_index("time").sort_index()
        dst = df_raw.rename(columns={
            "ExtTemp":        "temperature",
            "Pressure":       "pressure",
            "LightIntensity": "light",
        })[["temperature", "pressure", "light"]]
        if resample_freq:
            dst = dst.resample(resample_freq).mean()

    elif tag_type == "wc_psat_daily":
        dd = pd.read_csv(path)
        dd["time"] = pd.to_datetime(dd["Date"], format="%m/%d/%Y") + pd.Timedelta("12h")
        dd = dd.dropna(subset=["MinTemp", "MaxTemp"]).sort_values("time").set_index("time")
        dst = pd.DataFrame({
            "temperature": (dd["MinTemp"] + dd["MaxTemp"]) / 2.0,
            "pressure":    (dd["MinDepth"].clip(lower=0) + dd["MaxDepth"]) / 2.0,
            "light":       dd["DeltaLight"],   # proxy only — HAS_LIGHT should be False
        })
        if series_path is not None:
            ser = pd.read_csv(series_path).dropna(subset=["Depth", "Temperature"])
            ser["time"] = pd.to_datetime(
                ser["Day"] + " " + ser["Time"], format="%d-%b-%Y %H:%M:%S"
            )
            ser = ser.sort_values("time").set_index("time")
            ser = ser.rename(columns={"Depth": "pressure", "Temperature": "temperature"})
            ser["light"] = np.nan
            t_start = ser.index.min()
            dst = pd.concat([
                dst[dst.index < t_start],
                ser[["temperature", "pressure", "light"]],
            ])
            print(f"  Spliced 10-min Series from {t_start} ({len(ser):,} rows)")

    else:
        raise ValueError(
            f"Unknown tag_type {tag_type!r}. Supported: 'lotek', 'wc_psat_daily'."
        )

    print(f"  {len(dst):,} rows | {dst.index.min()} → {dst.index.max()}")
    return dst


def build_tag_datatree(
    dst,
    tag_name,
    tag_type,
    release_date,
    release_lon,
    release_lat,
    recapture_date,
    recapture_lon,
    recapture_lat,
):
    """Build an ``xr.DataTree`` from a standardised DST DataFrame.

    The returned tree mirrors the structure produced by
    ``pangeo_fish.io.open_tag``, so all downstream pipeline cells work
    unchanged.

    Parameters
    ----------
    dst : pd.DataFrame
        Output of :func:`load_tag_csv`.
    tag_name : str
        Human-readable tag identifier (e.g. ``"281B-4949"``).
    tag_type : str
        Manufacturer string stored as a tree attribute.
    release_date : str or datetime-like
        Release timestamp (UTC).
    release_lon, release_lat : float
        Release position (decimal degrees).
    recapture_date : str or datetime-like
        Recapture / pop-up timestamp (UTC).
    recapture_lon, recapture_lat : float
        Recapture position (decimal degrees).

    Returns
    -------
    tag : xr.DataTree
        Full tree with ``/dst`` and ``/tagging_events`` sub-trees.
    tag_log : xr.Dataset
        Slice of ``/dst`` between release and recapture dates.
    time_slice : slice
        ``slice(release_date, recapture_date)`` as ``pd.Timestamp``.
    """
    events = pd.DataFrame([
        {
            "event_name": "release",
            "time":       pd.Timestamp(release_date),
            "latitude":   release_lat,
            "longitude":  release_lon,
        },
        {
            "event_name": "fish_death",
            "time":       pd.Timestamp(recapture_date),
            "latitude":   recapture_lat,
            "longitude":  recapture_lon,
        },
    ]).set_index("event_name")

    tag = xr.DataTree.from_dict({
        "/": xr.Dataset(attrs={"tag_name": tag_name, "tag_type": tag_type}),
        "dst": dst.rename_axis("time").to_xarray(),
        "tagging_events": events.to_xarray(),
    })
    tag.attrs["tag_name"] = tag_name

    time_slice = slice(pd.Timestamp(release_date), pd.Timestamp(recapture_date))
    tag_log = tag["dst"].ds.sel(time=time_slice).assign_attrs(tag.attrs)

    print(f"tag_log: {len(tag_log.time):,} timesteps")
    return tag, tag_log, time_slice
