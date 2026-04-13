"""Tag CSV ingestion helpers.

Converts raw manufacturer files into the standard
``(time, temperature, pressure, light)`` DataFrame and builds an
``xr.DataTree`` compatible with the rest of the pangeo-fish pipeline.

Supported tag types
-------------------
``"lotek"``
    Lotek LAT2810 — semicolon-separated, comma decimal,
    timestamp format ``%H:%M:%S %d/%m/%y``.

``"wc_psat"``
    Wildlife Computers MiniPAT PSAT — 10-minute Series CSV
    (``*-Series.csv``) with columns ``Day``, ``Time``, ``Depth``,
    ``Temperature``.  No raw light counts are available, so ``light``
    is set to NaN and ``HAS_LIGHT`` must be ``False``.
"""

import numpy as np
import pandas as pd
import xarray as xr


def load_tag_csv(path, tag_type):
    """Load a raw manufacturer CSV and return a standardised DataFrame.

    Parameters
    ----------
    path : str or path-like
        Path to the CSV file.
    tag_type : {"lotek", "wc_psat"}
        Manufacturer / format identifier.

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
        df_raw["time"] = pd.to_datetime(
            df_raw["TimeS"].str.strip(), format="%H:%M:%S %d/%m/%y", errors="coerce"
        )
        df_raw = df_raw.dropna(subset=["time"]).set_index("time").sort_index()
        dst = df_raw.rename(columns={
            "ExtTemp":        "temperature",
            "Pressure":       "pressure",
            "LightIntensity": "light",
        })[["temperature", "pressure", "light"]]

    elif tag_type == "wc_psat":
        df_raw = pd.read_csv(path)
        df_raw["time"] = pd.to_datetime(
            df_raw["Day"] + " " + df_raw["Time"], format="%d-%b-%Y %H:%M:%S"
        )
        df_raw = df_raw.dropna(subset=["Depth", "Temperature"]).sort_values("time").set_index("time")
        dst = pd.DataFrame({
            "temperature": df_raw["Temperature"],
            "pressure":    df_raw["Depth"],
            "light":       np.nan,
        })

    else:
        raise ValueError(
            f"Unknown tag_type {tag_type!r}. Supported: 'lotek', 'wc_psat'."
        )

    print(f"  {len(dst):,} rows | {dst.index.min()} → {dst.index.max()}")
    return dst


def build_tag_datatree(dst, tag_name, tag_type, tagging_events_path):
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
    tagging_events_path : str or path-like
        Path to a ``tagging_events.csv`` with columns
        ``event_name``, ``time``, ``longitude``, ``latitude``.
        Must contain at least ``release`` and ``fish_death`` rows.

    Returns
    -------
    tag : xr.DataTree
        Full tree with ``/dst`` and ``/tagging_events`` sub-trees.
    tag_log : xr.Dataset
        Slice of ``/dst`` between release and recapture dates.
    time_slice : slice
        ``slice(release_time, recapture_time)`` as ``pd.Timestamp``.
    """
    events = pd.read_csv(
        tagging_events_path, parse_dates=["time"], index_col="event_name"
    )
    events["time"] = pd.to_datetime(events["time"]).dt.tz_localize(None)

    tag = xr.DataTree.from_dict({
        "/": xr.Dataset(attrs={"tag_name": tag_name, "tag_type": tag_type}),
        "dst": dst.rename_axis("time").to_xarray(),
        "tagging_events": events.to_xarray(),
    })
    tag.attrs["tag_name"] = tag_name

    t_release   = events.loc["release",    "time"]
    t_recapture = events.loc["fish_death", "time"]
    time_slice = slice(pd.Timestamp(t_release), pd.Timestamp(t_recapture))
    tag_log = tag["dst"].ds.sel(time=time_slice).assign_attrs(tag.attrs)

    print(f"tag_log: {len(tag_log.time):,} timesteps")
    return tag, tag_log, time_slice
