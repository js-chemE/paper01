import math
import os
from dataclasses import dataclass, field
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import scipy.stats as stats
import skimage.io as io
import tifffile
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from skimage import draw, measure

import flim_analysis as fa


@dataclass
class Feature:
    name: str
    raster: np.ndarray
    geometry: Polygon

    @property
    def transparent_mask(self) -> np.ndarray:
        mask = np.zeros(self.raster.shape)
        mask[self.raster > 0] = 1
        mask[mask == 0] = np.nan
        return mask

    @property
    def transparent_mask_boundary(self) -> np.ndarray:
        sobelx = ndi.sobel(self.raster, axis=0) ** 2  # type:ignore
        sobely = ndi.sobel(self.raster, axis=1) ** 2  # type:ignore
        sobel = sobelx + sobely
        boundary = np.zeros(sobel.shape)
        boundary[sobel > 0] = 1
        boundary[boundary == 0] = np.nan
        return boundary


def save_features(
    folder: str, features: List[Feature], feature_base: str = "features"
) -> None:
    features_shp = gpd.GeoDataFrame(
        data={"name": [f.name for f in features]},
        geometry=[f.geometry for f in features],
    )  # type:ignore
    features_shp.to_file(os.path.join(folder, f"{feature_base}_features.shp"))
    np.savez_compressed(
        os.path.join(folder, feature_base + "_features.npz"),
        **{
            k: v
            for k, v in zip([f.name for f in features], [f.raster for f in features])
        },
    )


def get_feature_from_contour(
    contour, array: np.ndarray, name: str, value: float = 1
) -> Feature:
    ys = contour[:, 1]
    xs = contour[:, 0]
    rr, cc = draw.polygon(xs, ys, array.shape)
    array = array.copy()
    array[rr, cc] = value
    return Feature(
        name=name,
        raster=np.asarray(array, dtype=np.uint8),
        geometry=Polygon(np.vstack((ys, xs)).T),
    )


def get_rest_as_feature(features: List[Feature], name: str) -> Feature:
    mask = np.ones(features[0].raster.shape, dtype=np.uint8) - np.sum(
        np.array([feature.raster for feature in features]), axis=0
    )
    padded_mask = np.zeros(
        mask.shape, dtype=np.uint8
    )  # np.pad(mask, pad_width=1, mode='constant')
    padded_mask[1:-1, 1:-1] = mask[1:-1, 1:-1]
    contour = measure.find_contours(padded_mask, 0.8, fully_connected="high")[0]
    ys = contour[:, 1]
    xs = contour[:, 0]
    geom = Polygon(np.vstack((ys, xs)).T.astype("int"))
    for feature in features:
        geom -= feature.geometry
    return Feature(name=name, raster=mask, geometry=geom)  # type:ignore


def get_feature_from_mask(mask: np.ndarray, name: str, value: float = 1) -> Feature:
    padded_mask = np.zeros(mask.shape)  # np.pad(mask, pad_width=1, mode='constant')
    padded_mask[1:-1, 1:-1] = mask[1:-1, 1:-1]
    contour = measure.find_contours(padded_mask, 0.8, fully_connected="high")[0]
    ys = contour[:, 1]
    xs = contour[:, 0]
    return Feature(
        name=name, raster=mask, geometry=Polygon(np.vstack((ys, xs)).T.astype("int"))
    )


def load_features(folder: str, shp_name: str, npz_name: str) -> List[Feature]:
    files = os.listdir(folder)
    features_shp = gpd.read_file(os.path.join(folder, shp_name))
    features_npz = np.load(os.path.join(folder, npz_name))
    features = []
    for name in features_npz.keys():
        geometry = features_shp[features_shp["name"] == name].dissolve()["geometry"][0]
        raster = features_npz[name]
        feature = Feature(name=name, raster=raster, geometry=geometry)
        features.append(feature)
    return features


def get_rays_of_sphere_like_base_points(
    base_points, bearing_pulling_factor=1, base_offset=0
):
    rays = []
    for i, p0 in enumerate(base_points):
        if i == 0:
            p2m = base_points[-2]
            p1m = base_points[-1]
            p1 = base_points[i + 1]
            p2 = base_points[i + 2]
        elif i == 1:
            p2m = base_points[-1]
            p1m = base_points[i - 1]
            p1 = base_points[i + 1]
            p2 = base_points[i + 2]
        elif i == len(base_points) - 2:
            p2m = base_points[i - 2]
            p1m = base_points[i - 1]
            p1 = base_points[i + 1]
            p2 = base_points[0]
        elif i == len(base_points) - 1:
            p2m = base_points[i - 2]
            p1m = base_points[i - 1]
            p1 = base_points[0]
            p2 = base_points[1]
        else:
            p2m = base_points[i - 2]
            p1m = base_points[i - 1]
            p1 = base_points[i + 1]
            p2 = base_points[i + 2]

        bearing_left = fa.get_angle(p2m, p1m)
        bearing_right = fa.get_angle(p1, p2)
        bearing_1 = fa.get_angle(p1m, p1)
        bearing_2 = fa.get_angle(p2m, p2)

        angle_should = 90 - i * len(base_points) / 360
        if angle_should < 0:
            angle_should += 360
        bearing_should = angle_should - 90
        bearings = [bearing_left, bearing_right, bearing_1, bearing_2]
        for i in range(bearing_pulling_factor):
            bearings.append(bearing_should)
        bearing = stats.circmean([bearings], high=360, low=0)
        angle = bearing + 90
        if angle > 360:
            angle -= 360
        ray = Ray(base_point=p0, angle=angle)  # type: ignore
        print(angle_should, ray.angle)
        ray.move_point(point_name="base_point", distance=base_offset)

        rays.append(ray)
    return rays


@dataclass
class Ray:
    base_point: Point
    angle: float = field(default=0)
    end_point: Point = field(default_factory=Point)
    end_point_image: Point = field(default_factory=Point)
    end_point_volume: Point = field(default_factory=Point)
    evaluation_distance: float = field(default=1)

    """Properties"""

    @property
    def direction(self):
        return np.array(
            [math.cos(math.radians(self.angle)), math.sin(math.radians(self.angle))]
        )

    @property
    def base_line(self):
        return LineString([self.base_point, self.end_point_image])

    @property
    def evaluation_line(self):
        return LineString([self.base_point, self.end_point])

    @property
    def base_ray(self):
        return LineString(
            [
                self.base_point,
                fa.get_point_from_angle(self.base_point, self.angle, dist=5000),
            ]
        )

    @property
    def length(self):
        return self.base_line.length

    @property
    def evaluation_points(self):
        points = fa.get_points_along_line_distance(
            self.evaluation_line, self.evaluation_distance
        )
        return fa.get_sorted_point_list_distance(points, self.base_point)

    """Endpoints"""

    def set_end_point_by_distance(self, distance):
        setattr(
            self,
            "end_point",
            fa.get_point_from_angle(self.base_point, self.angle, dist=distance),
        )

    def set_evaluation_end_point(self, point: Point):
        setattr(self, "end_point", point)

    def set_end_point(self, distance: float, bound: str = "image", maximum_length=1e6):
        temp_end_point = fa.get_point_from_angle(
            self.base_point, self.angle, dist=distance
        )
        temp_line = LineString([self.base_point, temp_end_point])
        volume_line = LineString([self.base_point, self.end_point_volume])
        image_line = LineString([self.base_point, self.end_point_image])
        temp_length = maximum_length
        closest_point = self.base_point
        for point in [temp_end_point, self.end_point_volume, self.end_point_image]:
            line = LineString([self.base_point, point])
            length = line.length
            if length < temp_length:
                closest_point = point
                temp_length = length
        setattr(self, "end_point", closest_point)

    def set_end_point_image(self, image: np.ndarray, offset: float = 0):
        image_vectorized = Polygon(fa.get_corners_of_array(image))
        end_point_temp = fa.get_point_from_angle(
            self.base_point,
            self.angle,
            dist=int(math.hypot(image.shape[1], image.shape[0])),
        )
        helper_line = LineString((self.base_point, end_point_temp))
        intersect = helper_line.intersection(image_vectorized.boundary)
        if type(intersect) != Point:
            print("ERROR - more than on point")
        else:
            self.end_point_image = Point(intersect)
        self.move_point("end_point_image", offset)

    def set_end_point_volume(self, volume, offset: float = 0):
        helper_line = LineString((self.base_point, self.end_point_image))
        intersect = helper_line.intersection(volume.exterior)

        end_point_volume = self.base_point

        if type(intersect) == Point:
            end_point_volume = Point(intersect)

        elif type(intersect) == MultiPoint:
            intersects = [pt for pt in intersect.geoms]  # type:ignore
            min_distance = float("inf")
            closest_point = self.base_point
            for p in intersects:
                distance = self.base_point.distance(p)
                if distance < min_distance:
                    closest_point = p
                    min_distance = distance

            end_point_volume = closest_point
        setattr(self, "end_point_volume", end_point_volume)
        self.move_point("end_point_volume", offset)

    def move_point(self, point_name: str, distance: float):
        setattr(
            self,
            point_name,
            fa.get_point_from_angle(getattr(self, point_name), self.angle, distance),
        )

    """Values"""

    def get_values_along_ray(
        self, image: np.ndarray, pixel_size: float = 1, rolling_window: float = 1
    ):
        res = {}
        res["values"] = []
        res["xs"] = []
        res["ys"] = []
        res["ray"] = (
            np.linspace(
                0, np.round(self.evaluation_line.length), len(self.evaluation_points)
            )
            * self.evaluation_distance
            * pixel_size
        )

        for i_point, point in enumerate(self.evaluation_points):
            res["xs"].append(int(point.x))
            res["ys"].append(int(point.y))
            try:
                res["values"].append(image[int(point.y), int(point.x)])

            except IndexError:
                res["values"].append(np.nan)
        res["xs"] = np.array(res["xs"])
        res["ys"] = np.array(res["ys"])
        res["values"] = np.array(res["values"])
        if rolling_window > 1:
            res["values"] = fa.rollavg_convolve_edges(res["values"], rolling_window)
        return res


"""
@dataclass
class Ray:
    base_point: Point
    angle: float = field(default=0)
    end_point: Point = field(default_factory=Point)
    end_point_image: Point = field(default_factory=Point)
    end_point_volume: Point = field(default_factory=Point)
    width: float = field(default=10)
    evaluation_distance: float = field(default=1)

    def set_end_point_by_distance(self, distance):
        setattr(
            self,
            "end_point",
            mep.analysis.get_point_from_angle(
                self.base_point, self.angle, dist=distance
            ),
        )

    def set_end_point(self, distance: float, bound: str = "image", maximum_length=1e6):
        temp_end_point = mep.analysis.get_point_from_angle(
            self.base_point, self.angle, dist=distance
        )
        temp_line = LineString([self.base_point, temp_end_point])
        volume_line = LineString([self.base_point, self.end_point_volume])
        image_line = LineString([self.base_point, self.end_point_image])
        temp_length = maximum_length
        closest_point = self.base_point
        for point in [temp_end_point, self.end_point_volume, self.end_point_image]:
            line = LineString([self.base_point, point])
            length = line.length
            if length < temp_length:
                closest_point = point
                temp_length = length
        setattr(self, "end_point", closest_point)

    def set_end_point_image(self, image: np.ndarray, offset: float = 0):
        image_vectorized = Polygon(mep.analysis.get_corners_of_array(image))
        end_point_temp = mep.analysis.get_point_from_angle(
            self.base_point,
            self.angle,
            dist=int(math.hypot(image.shape[1], image.shape[0])),
        )
        helper_line = LineString((self.base_point, end_point_temp))
        intersect = helper_line.intersection(image_vectorized.boundary)
        if type(intersect) != Point:
            print("ERROR - more than on point")
        else:
            self.end_point_image = Point(intersect)
        self.move_point("end_point_image", offset)

    def set_end_point_volume(self, volume, offset: float = 0):
        helper_line = LineString((self.base_point, self.end_point_image))
        intersect = helper_line.intersection(volume.exterior)

        end_point_volume = self.base_point

        if type(intersect) == Point:
            end_point_volume = Point(intersect)

        elif type(intersect) == MultiPoint:
            intersects = [pt for pt in intersect]  # type:ignore
            min_distance = float("inf")
            closest_point = self.base_point
            for p in intersects:
                distance = self.base_point.distance(p)
                if distance < min_distance:
                    closest_point = p
                    min_distance = distance

            end_point_volume = closest_point
        setattr(self, "end_point_volume", end_point_volume)
        self.move_point("end_point_volume", offset)

    @property
    def direction(self):
        return np.array(
            [math.cos(math.radians(self.angle)), math.sin(math.radians(self.angle))]
        )

    @property
    def base_line(self):
        return LineString([self.base_point, self.end_point_image])

    def set_evaluation_end_point(self, point: Point):
        setattr(self, "end_point", point)

    @property
    def evaluation_line(self):
        return LineString([self.base_point, self.end_point])

    @property
    def length(self):
        return self.base_line.length

    def set_width(self, width: float):
        self.width = width

    def move_point(self, point_name: str, distance: float):
        setattr(
            self,
            point_name,
            mep.analysis.get_point_from_angle(
                getattr(self, point_name), self.angle, distance
            ),
        )

    @property
    def evaluation_points(self):
        points = mep.analysis.get_points_along_line_distance(
            self.evaluation_line, self.evaluation_distance
        )
        return mep.analysis.get_sorted_point_list_distance(points, self.base_point)

    def get_values_along_ray(
        self, image: np.ndarray, pixel_size: float = 1, rolling_window: float = 1
    ):
        res = {}
        res["values"] = []
        res["xs"] = []
        res["ys"] = []
        res["ray"] = (
            np.linspace(
                0, np.round(self.evaluation_line.length), len(self.evaluation_points)
            )
            * self.evaluation_distance
            * pixel_size
        )

        for i_point, point in enumerate(self.evaluation_points):
            res["xs"].append(int(point.x))
            res["ys"].append(int(point.y))
            try:
                res["values"].append(image[int(point.y), int(point.x)])

            except IndexError:
                res["values"].append(np.nan)
        res["xs"] = np.array(res["xs"])
        res["ys"] = np.array(res["ys"])
        res["values"] = np.array(res["values"])
        if rolling_window > 1:
            res["values"] = mep.analysis.rollavg_convolve_edges(
                res["values"], rolling_window
            )
        return res
"""
