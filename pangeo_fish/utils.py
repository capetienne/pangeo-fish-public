import re

import cf_xarray  # noqa: F401
import more_itertools
import pandas as pd
import xarray as xr


def clear_attrs(obj, variables=None):
    new_obj = obj.copy()
    new_obj.attrs.clear()

    if variables is None:
        variables = []
    elif variables == "all":
        variables = list(getattr(new_obj, "variables", new_obj.coords))

    for name in more_itertools.always_iterable(variables):
        new_obj[name].attrs.clear()
    return new_obj


def _drop_attr(obj, attr):
    new_obj = obj.copy()
    new_obj.attrs.pop(attr, None)

    variables = new_obj.variables if hasattr(new_obj, "variables") else new_obj.coords

    for var in variables.values():
        var.attrs.pop(attr, None)

    return new_obj


def drop_crs(obj, coord="spatial_ref", attr="crs"):
    """remove crs attributes

    Unfortunately, `rioxarray` and `odc-geo` store the crs information in a format that
    is not serializable to `netcdf4` / `zarr`. This could be solved using a registry of
    type → encoder that would be applied to the attrs, but until that is implemented,
    we'll just drop the crs before writing.
    """
    return obj.drop_vars(coord).pipe(_drop_attr, attr)


def postprocess_depth(ds):
    new_names = {
        detected: standard
        for standard, (detected,) in ds.cf.standard_names.items()
        if detected in ds.coords
    }
    return ds.rename(new_names)


def normalize(obj, dim):
    return obj / obj.sum(dim=dim)


def _detect_dims(ds, guesses):
    for guess in guesses:
        try:
            coords = ds.cf[guess]
        except KeyError:
            continue

        return list(coords.dims)

    return None


def _detect_spatial_dims(
    ds, guesses=[["Y", "X"], ["latitude", "longitude"], ["x", "y"]]
):
    spatial_dims = _detect_dims(ds, guesses)
    if spatial_dims is None:
        raise ValueError(
            "could not determine spatial dimensions. Try"
            " calling `.cf.guess_coord_axis()` on the dataset."
        )

    return spatial_dims


def _detect_temporal_dims(ds, guesses=["T", "time"]):
    temporal_dims = _detect_dims(ds, guesses)
    if temporal_dims is None:
        raise ValueError(
            "could not determine temporal dimensions. Try"
            " calling `.cf.guess_coord_axis()` on the dataset."
        )

    return temporal_dims


units_re = re.compile(r"timedelta64\[(?P<units>.+?)\]")


def timedelta_units(arr):
    dtype = arr.dtype

    if dtype.kind != "m":
        raise ValueError("not a timedelta64")

    match = units_re.fullmatch(dtype.name)
    if match is None:
        raise ValueError("timedelta64 without units")

    return match.group("units")


def temporal_resolution(time):
    from pandas.tseries.frequencies import to_offset

    freq = xr.infer_freq(time)
    timedelta = to_offset(freq).delta.to_numpy()
    units = timedelta_units(timedelta)

    return xr.DataArray(timedelta.astype("float"), dims=None, attrs={"units": units})


def encode_positions(df):
    position_labels = df["name"].to_list()

    indexed = df.reset_index().set_index("name")
    positions = [
        indexed.loc[name][["latitude", "longitude"]].to_list()
        for name in position_labels
    ]
    dates = [str(indexed.loc[name]["time"]) for name in position_labels]

    return {
        "position_labels": position_labels,
        "position_data": positions,
        "position_time": dates,
    }


def decode_positions(attrs):
    column_names = ["time", "name", "latitude", "longitude"]

    names = attrs["position_labels"]
    dates = attrs["position_time"]
    data = attrs["position_data"]

    return pd.DataFrame(
        data,
        columns=column_names[2:],
        index=pd.Index(dates, name=column_names[0]),
    ).assign(**{column_names[1]: names})[column_names[1:]]
