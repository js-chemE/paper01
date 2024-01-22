import math
from typing import List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.affinity import affine_transform
from shapely.geometry import LinearRing, LineString, Point
from skimage import draw, measure, segmentation

MORPH_TYPE = {
    "ellipse": cv2.MORPH_ELLIPSE,
    "rectangle": cv2.MORPH_RECT,
    "cross": cv2.MORPH_CROSS,
}

"""Feature and Image Analysis"""


def extract_feature(
    gray_scale_frame: np.ndarray,
    adaptive_threshold: int = 101,
    kernel_shape: Tuple[int, int] = (10, 10),
    kernel_type: str = "ellipse",
    first: str = "close",
    nth_element: int = 0,
    snake_alpha: float = 0.0015,
    snake_beta: float = 1,
    snake_gamma: float = 0.5,
    snake_w_line: int = 0,
    snake_w_edge: int = 0,
    plot_process: bool = True,
):
    """https://stackoverflow.com/questions/61432335/blob-detection-in-python"""
    region_properties = [
        "label",
        "bbox",
        "area",
        "centroid",
        "centroid_weighted",
        "coords",
        "eccentricity",
        "axis_major_length",
        "axis_minor_length",
        "equivalent_diameter_area",
        "extent",
        "orientation",
        "image",
    ]
    # normalize
    process_properties = {}
    process_properties["image"] = np.array(
        gray_scale_frame / np.max(gray_scale_frame) * 255, dtype=np.uint8
    )
    # do adaptive threshold on gray image
    process_properties["thresh"] = cv2.adaptiveThreshold(
        process_properties["image"],
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        adaptive_threshold,
        3,
    )
    process_properties["first"] = np.zeros(process_properties["thresh"].shape)
    process_properties["second"] = np.zeros(process_properties["thresh"].shape)
    if first == "open":
        kernel = cv2.getStructuringElement(MORPH_TYPE[kernel_type], kernel_shape)
        process_properties["first"] = cv2.morphologyEx(
            process_properties["thresh"], cv2.MORPH_OPEN, kernel
        )
        kernel = cv2.getStructuringElement(MORPH_TYPE[kernel_type], kernel_shape)
        process_properties["second"] = cv2.morphologyEx(
            process_properties["first"], cv2.MORPH_CLOSE, kernel
        )
    elif first == "close":
        kernel = cv2.getStructuringElement(MORPH_TYPE[kernel_type], kernel_shape)
        process_properties["first"] = cv2.morphologyEx(
            process_properties["thresh"], cv2.MORPH_CLOSE, kernel
        )
        kernel = cv2.getStructuringElement(MORPH_TYPE[kernel_type], kernel_shape)
        process_properties["second"] = cv2.morphologyEx(
            process_properties["first"], cv2.MORPH_OPEN, kernel
        )
    else:
        print("Choose 'open' or 'close'!")
    process_properties["labeled"], numbers = measure.label(label_image=np.asarray(process_properties["second"]), background=0, return_num=True, connectivity=2)  # type: ignore
    process_properties["segmentation_properties"] = measure.regionprops_table(
        process_properties["labeled"],
        process_properties["image"],
        properties=region_properties,
    )
    process_properties["contours"] = measure.find_contours(
        process_properties["labeled"], 0.8, fully_connected="high"
    )

    process_properties["snakes"] = []
    for contour in process_properties["contours"]:
        snake = segmentation.active_contour(
            gray_scale_frame,
            contour,
            alpha=snake_alpha,
            beta=snake_beta,
            gamma=snake_gamma,
            w_line=snake_w_line,
            w_edge=snake_w_edge,
        )
        process_properties["snakes"].append(snake)

    final_label = np.where(
        process_properties["segmentation_properties"]["area"]
        == np.sort(process_properties["segmentation_properties"]["area"])[nth_element]
    )[0][0]
    final_index = final_label - 1
    final_contour = process_properties["contours"][int(final_index)]
    final_snake = process_properties["snakes"][int(final_index)]
    rr, cc = draw.polygon(
        process_properties["snakes"][final_index][:, 0],
        process_properties["snakes"][final_index][:, 1],
        gray_scale_frame.shape,
    )
    final_mask = np.zeros(gray_scale_frame.shape, dtype=np.uint8)
    final_mask[rr, cc] = 1
    final_properties = pd.DataFrame.from_dict(
        measure.regionprops_table(
            final_mask.astype(np.uint8),
            process_properties["image"],
            properties=region_properties,
        )
    )

    if plot_process:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
        flat_axes = np.ravel(axes)  # type:ignore
        ax = flat_axes[0]
        ax.imshow(process_properties["image"], cmap="gray")
        ax = flat_axes[1]
        ax.imshow(process_properties["thresh"], cmap="gray")
        ax = flat_axes[2]
        ax.imshow(process_properties["first"], cmap="gray")
        ax = flat_axes[3]
        ax.imshow(process_properties["second"], cmap="gray")
        ax = flat_axes[4]
        ax.imshow(process_properties["labeled"])
        ax = flat_axes[5]
        ax.imshow(gray_scale_frame, cmap="gray")
        for contour in process_properties["contours"]:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        ax = flat_axes[6]
        ax.imshow(process_properties["image"], cmap="gray")
        for snake in process_properties["snakes"]:
            ax.plot(snake[:, 1], snake[:, 0], linewidth=2)
        ax = flat_axes[7]
        ax.imshow(final_mask, cmap="gray")
        ax.plot(final_contour[:, 1], final_contour[:, 0], linewidth=2)
        ax.plot(final_snake[:, 1], final_snake[:, 0], linewidth=2)
        plt.show()
    return final_mask, final_index, process_properties, final_properties


def get_extent(image: np.ndarray, pixel_size: float, factor: float = 1):
    extent = np.array([0, image.shape[1] * pixel_size, 0, image.shape[0] * pixel_size])
    return extent * factor


def get_corners_of_array(array: np.ndarray) -> List[Tuple[int, int]]:
    xmin, ymin = 0, 0
    ymax, xmax = array.shape[0] - 1, array.shape[1] - 1
    return [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]


def angle_image2flow(image_angle):
    flow_angle = image_angle + 90
    if flow_angle > 180:
        flow_angle = -(360 - flow_angle)
    return flow_angle


def angle_flow2image(flow_angle):
    image_angle = flow_angle - 90
    if image_angle < 0:
        image_angle = 360 + image_angle
    return image_angle


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


"""Rays and Points"""


def get_angle(pt1, pt2):
    x_diff = pt2.x - pt1.x
    y_diff = pt2.y - pt1.y
    return math.degrees(math.atan2(y_diff, x_diff))


def get_point_from_angle(pt: Point, direction: float, dist: float = 20):
    x = pt.x + dist * math.cos(math.radians(direction))
    y = pt.y + dist * math.sin(math.radians(direction))
    return Point(x, y)


def get_points_along_line_distance(
    line: Union[LineString, LinearRing], distance_delta: float = 20
) -> List[Point]:
    distances = np.arange(0, line.length, distance_delta)
    points = [Point(LineString(line).interpolate(distance)) for distance in distances]
    return points


def get_points_along_line_number(
    line: Union[LineString, LinearRing], number_of_points: int
):
    return get_points_along_line_distance(line, line.length / number_of_points)


def get_sorted_point_list_distance(points, point):
    points.sort(key=lambda p: (p.x - point.x) ** 2 + (p.y - point.y) ** 2)
    return points


"""Rolling Averages"""


def linear(x, m, n):
    return m * x + n


def linear_n1(x, m):
    return linear(x, m, 1)


def rollavg_convolve_edges(a, n):
    "scipy.convolve, edge handling"
    assert n % 2 == 1
    return np.convolve(a, np.ones(n, dtype="float"), "same") / np.convolve(
        np.ones(len(a)), np.ones(n), "same"
    )


def rollavg_convolve_circle(a, n):
    "scipy.convolve, edge handling"
    assert n % 2 == 1
    new_a = np.concatenate([a, a, a], axis=0)
    res = np.convolve(new_a, np.ones(n, dtype="float"), "same") / np.convolve(
        np.ones(len(new_a)), np.ones(n), "same"
    )
    return res[len(a) : 2 * len(a)]
