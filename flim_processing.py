import os
from typing import List, Tuple, Union

import numpy as np
from shapely.affinity import affine_transform


def open_flim(path: str, property_modifications: dict = {}):
    raw = np.load(path)
    if len(property_modifications.keys()) == 0:
        return raw
    else:
        processed = {}
        for k in raw.keys():
            if k in property_modifications.keys():
                processed[k] = raw[k] / property_modifications[k]
            else:
                processed[k] = raw[k]
        return processed


def process_raw_flim(
    path: str, spinning_disc_cutoff: int = 42, dtype=np.uint16
) -> None:
    raw = open_flim(path)
    processed = {}
    for key in raw.keys():
        processed[key] = cutflip_series(
            raw[key],
            axes=[1],
            cutoffs=[(0, -spinning_disc_cutoff)],
        )
        print(key, raw[key].shape, processed[key].shape)
    processed_path = os.path.join(
        os.path.dirname(path),
        ".".join(os.path.basename(path).split(".")[:-1]) + "_p.npz",
    )

    np.savez_compressed(
        processed_path,
        **{
            k: v
            for k, v in zip(
                raw.keys(), [processed[k].astype(dtype) for k in raw.keys()]
            )
        },
    )


def cutflip(img: np.ndarray, cutoff: int):
    return np.flipud(img[:-cutoff])


def array_slice(a, axis, start, end, step=1) -> np.ndarray:
    return a[(slice(None),) * (axis % a.ndim) + (slice(start, end, step),)]


def cutflip_series(series: np.ndarray, axes: List[int], cutoffs: List[Tuple[int, int]]):
    # print(series.shape)
    for ax, cutoff in zip(axes, cutoffs):
        series = array_slice(series, ax, cutoff[0], cutoff[1])
    # print(series.shape)
    return np.flip(series, axis=1)


def timestamp2frame_flim(time_stamps, flim_info: dict, offset_flim: float = 0):
    flim_timestamps = (
        np.asarray([*flim_info["FLIMIMAGE"]["TIMESTAMPS in ms"].values()]) * 1e-3
        + offset_flim
    )
    res = {}
    res["t"] = []
    res["flim_frames"] = []
    res["flim_times_offset"] = []
    res["flim_times"] = []
    for time_stamp in time_stamps:
        res["t"].append(time_stamp)
        res["flim_times_offset"].append(
            flim_timestamps[flim_timestamps < time_stamp][-1] - offset_flim
        )
        res["flim_times"].append(flim_timestamps[flim_timestamps < time_stamp][-1])
        res["flim_frames"].append(
            len(flim_timestamps[flim_timestamps < time_stamp]) - 1
        )
    return res


def affine_transform_pixel2extent(geom, geom_extent: List[float], extent: List[float]):
    matrix = [
        (extent[1] - extent[0]) / (geom_extent[1] - geom_extent[0]),
        0,
        0,
        (extent[2] - extent[3]) / (geom_extent[3] - geom_extent[2]),
        0,
        extent[3],
    ]
    return affine_transform(geom, matrix)


if __name__ == "__main__":
    # process_raw_flim(r"D:\[Code]\paper01\data\current.NPZ")
    print(open_flim(r"D:\[Code]\paper01\data\current_p.npz").keys())
