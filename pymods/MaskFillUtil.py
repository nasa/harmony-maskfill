""" Utility functions to support MaskFill processing """
from collections import namedtuple
from os.path import join as path_join
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4
from warnings import catch_warnings, simplefilter
import hashlib
import json
import os

from affine import Affine
from geopandas import GeoDataFrame, GeoSeries
from h5py import Dataset, File as H5File, Group
from osgeo import gdal
from pyproj import Transformer, CRS
from rasterio.features import rasterize
from shapely.geometry import Polygon, shape
import geopandas as gpd
import numpy as np
import rasterio


BBox = namedtuple('BBox', ['west', 'south', 'east', 'north'])
Coordinates = Tuple[float]


def get_mask_array(shape_path: str, crs: CRS, out_shape: Tuple[int, int],
                   transform: Affine) -> np.ndarray:
    """ Rasterizes the intersection of the given shapes and the bounding box of
        the data to create a mask array.

        Args:
            shape_path (string): The path to the shape file
            crs (pyproj.CRS): The target CRS
            out_shape (tuple): The shape of the resultant mask array
            transform (affine.Affine): A transform mapping from image coordinates
            to world coordinates
        Returns:
            numpy.ndarray: A numpy array representing the rasterized shapes
    """
    bounded_shape_gdf = get_bounded_shape(shape_path, crs, out_shape,
                                          transform)

    # Project data frame to new coordinate reference system
    projected_gdf = bounded_shape_gdf.to_crs(crs)
    shapes = projected_gdf['geometry']

    # Rasterize the bounded and projected shapes into the mask array
    if shapes.is_empty.empty:
        return np.ones(out_shape)

    return rasterize(shapes=shapes, default_value=0, fill=1,
                     out_shape=out_shape, dtype=np.uint8, transform=transform,
                     all_touched=True)


def get_bounded_shape(shape_path: str, crs: CRS, out_shape: Tuple[int, int],
                      transform: Affine) -> gpd.GeoDataFrame:
    """ Creates a geodataframe (in geographic coordinates) for the shapes in
        the shape file. Bounds the shapes by the geographic extent of the data.

            Args:
                shape_path (string): The path to the shape file
                crs (pyproj.CRS): CRS of the target grid.
                out_shape (tuple): The shape of the resultant mask array
                transform (affine.Affine): A transform mapping from image coordinates
                to world coordinates
            Returns:
                geodataframe: The bounded shape geodataframe

        `pyproj.CRS` will not necessarily populate an Area Of Use for non-EPSG
        defined CRS objects, so if a CRS has an EPSG code, this will need to be
        used to instantiate a new CRS with an area of use.

    """
    epsg = crs.to_epsg()

    if epsg is not None and not should_ignore_pyproj_bounds(crs):
        # Get geographic extent of data using the EPSG code
        minx, miny, maxx, maxy = CRS(epsg).area_of_use.bounds

    else:
        # Transform all indices in the data array to geographic coordinates
        # and get min/max lat/lon values
        latitudes, longitudes = get_grid_lat_lons(transform, crs, out_shape)
        minx, maxx = np.nanmin(longitudes), np.nanmax(longitudes)
        miny, maxy = np.nanmin(latitudes), np.nanmax(latitudes)

    # Create bounding box in geographic coordinates
    bbox = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
    bbox_gdf = GeoDataFrame(geometry=GeoSeries(Polygon(bbox), crs='EPSG:4326'))

    if crs.is_geographic:
        shape_gdf = gpd.read_file(shape_path)
    else:
        shape_gdf = get_resolved_dataframe(shape_path, transform, crs,
                                           out_shape)

    # Intersect shapes with bounding box
    bounded_shape_gdf = gpd.overlay(shape_gdf, bbox_gdf, how='intersection')
    return bounded_shape_gdf


def should_ignore_pyproj_bounds(crs: CRS) -> bool:
    """ A function to check if the CRS is a projection that should not use the
        bounds defined by the `pyproj.CRS` object. For example, strict UTM
        zones are constrained to 6 degree-wide longitudinal regions, but users
        may have shapes that transcend these regions.

        `pyproj` gives a warning when converting CRS objects to a Proj4 string,
        however, we do not require the additional information that may be lost,
        so those warnings are suppressed in this function.

    """
    with catch_warnings():
        simplefilter('ignore')
        proj4_string = crs.to_proj4()

    return '+proj=utm' in proj4_string


def mask_fill_array(raster_arr: np.ndarray, mask_array: np.ndarray,
                    fill_value: float) -> np.ndarray:
    """ Performs a mask fill on raster_arr using the mask mask_array
        Args:
            raster_arr: The array to be masked.
            mask_array: The mask array which will be applied to raster_arr
            fill_value (float): Value used to fill in the masked values when necessary
        Returns:
            numpy.ndarray: The mask filled array
    """

    out_image = np.ma.array(raster_arr, mask=mask_array, fill_value=fill_value)
    return out_image.filled()


def get_masked_file_path(original_file_path: str, output_dir: str) -> str:
    """ Returns the path to the mask filled output file.
        Args:
            original_file_path (str): The original file which is to be mask filled
            output_dir (str): The directory to which the output file will be written
        Returns:
            str: The path to the mask filled version of the original file

    """
    base_name = os.path.basename(original_file_path)
    file_name, extension = os.path.splitext(base_name)
    return os.path.join(output_dir, f'{file_name}_mf{extension}')


def get_h5_mask_array_id(h5_dataset: Dataset, crs: CRS,
                         shape_path: str) -> str:
    """ Creates an ID corresponding to the given shape file, projection
        information, pixel-to-projected coordinates Affine transformation
        inputs, and shape of a dataset, which determine the mask array for
        the dataset.

        Returns:
            str: A string ID generated via a hashing algorithm, based upon the
                a combined input string of the shape file path, dataset shape,
                dataset projection and Affine transformation.
    """
    transform_info = get_transform_information(h5_dataset)
    dataset_shape = h5_dataset[:].shape

    return create_mask_array_id(crs, transform_info, dataset_shape, shape_path)


def get_transform_information(h5_dataset: Dataset) -> str:
    """ Determine the attributes of an HDF-5 dataset that will be used to
        determine the Affine transformation between pixel indices and
        projected coordinates. This function doesn't actually derive the
        tranform itself, in an effort to minimise computationally intensive
        operations.

    """
    from pymods.H5GridProjectionInfo import get_dimension_datasets

    if get_dimension_datasets(h5_dataset):
        dimension_list = get_decoded_attribute(h5_dataset, 'DIMENSION_LIST')
        h5_file = h5_dataset.file
        dimension_names = ', '.join([h5_file[reference[0]].name
                                     for reference in dimension_list])
        output_string = f'DIMENSION_LIST: {dimension_names}'
    else:
        output_string = f'coords: {get_decoded_attribute(h5_dataset, "coordinates")}'

    return output_string


def get_decoded_attribute(h5_dataset: Dataset, attribute_key: str,
                          default: Optional[Any] = None) -> Any:
    """ Ensure that any Byte type attributes are decoded to a string. Otherwise
        return the metadata attribute as stored in the H5 file.

        With some OPeNDAP output, `h5py` will consider single floating point
        values to be 1-element arrays of floating point values. If a `numpy`
        array is detected with only one value (that is not itself an object),
        that metadata attribute is determined to be just the element itself,
        not an array. `numpy` arrays with `dtype='object'` are left as arrays
        as these are most commonly `h5py.References` and these are typically
        stored as arrays.

    """
    attribute_value = h5_dataset.attrs.get(attribute_key, default)

    if isinstance(attribute_value, (bytes, np.bytes_)):
        attribute_value = attribute_value.decode()
    elif (
        isinstance(attribute_value, np.ndarray) and attribute_value.size == 1
        and not attribute_value.dtype == 'object'
    ):
        attribute_value = attribute_value[0]

    return attribute_value


def get_geotiff_mask_array_id(geotiff_path: str, shape_path: str) -> str:
    """ Creates an ID corresponding to the given shape file, projection
    information and shape of a dataset, which determine the mask array for the
    dataset.

    Args:
        geotiff_path (str): Path to the GeoTIFF dataset.
        shape_path (str): Path to a shape file used to create the mask array
            for the mask fill.

        Returns:
            str: A string ID generated via a hashing algorithm, based upon the
                a combined input string of the shape file path, dataset shape,
                dataset projection and Affine transformation.
    """
    # The mask array is determined by the CRS of the dataset, the dataset's
    # transform, the shape of the dataset, and the shapes used in the mask.
    crs = get_geotiff_crs(geotiff_path)
    dataset_shape, transform = get_geotiff_info(geotiff_path)

    return create_mask_array_id(crs, transform, dataset_shape, shape_path)


def create_mask_array_id(crs: CRS, transform: Union[str, Affine],
                         dataset_shape: Tuple[int],
                         shape_file_path: str) -> str:
    """ Creates an ID corresponding to the given shapefile, projection
        information (CRS and affine transformation), and shape of a dataset.

        Args:
            crs: A `pyproj.CRS` object describing the projection of the data.
            transform:
              - GeoTIFF (affine.Affine): The affine transform for the grid.
              - HDF-5 (str): A string describing the affine transform.
            dataset_shape (tuple): The shape of the data array.
            shape_file_path (str): The path to the given shapefile.
        Returns:
            str: A string ID generated via a hashing algorithm, based upon the
                a combined input string of the shape file path, dataset shape,
                dataset projection and Affine transformation.
    """
    mask_id = (f'{crs.to_string()}{str(transform)}{str(dataset_shape)}'
               f'{shape_file_path}')

    return hashlib.sha224(mask_id.encode()).hexdigest()


def get_geotiff_crs(geotiff_path: str) -> CRS:
    """ Returns a `pyproj.crs.Crs` object that is constructed from the Well
        Known Text (WKT) representation of the GeoTIFF projection information.

    """
    data = gdal.Open(geotiff_path)
    wkt_string = data.GetProjection()
    return CRS.from_wkt(wkt_string)


def get_geotiff_info(geotiff_path: str) -> Tuple[Tuple[int], Affine]:
    """ Retuns the shape and transform of the given GeoTIFF.

    Args:
        geotiff_path (str): The path to the GeoTIFF

    Returns:
        tuple: The shape (tuple) and transform (affine.Affine) corresponding to
            the GeoTIFF.
    """
    with rasterio.open(geotiff_path) as raster:
        shape = raster.read(1).shape
        transform = raster.transform

    return shape, transform


def process_h5_file(file_path, process, *args):
    """ Performs the given process on all datasets in the HDF5 file.
        Args:
            file_path (str): The path to the input HDF5 file
            process (function): The process to be performed on the datasets in the file
            *args: The arguments passed to the process
    """

    def process_children(obj, process, *args):
        for child in obj.values():
            # Process the children of a group
            if isinstance(child, Group):
                process_children(child, process, *args)
            # Process datasets
            elif isinstance(child, Dataset):
                process(child, *args)

    with H5File(file_path, mode='r+') as file:
        process_children(file, process, *args)


def apply_2d(data, process, *args):  # , name = ""  ??? - unused!
    """ Recursively applies a 2D process to datasets with two or more dimensions,
        Always applies the process to the last 2 dimensions of the dataset,
        iterating through any lower dimensions and processing up through n-2 dimensions.
        E.g., a dataset with coordinates dimensions: (time, lat, lon),
            will apply to lat, lon dimensions, for each time entry.
        Args:
            data (numpy.ndarray): The data array to be processed
            process: The process to be applied to data
            *args: tuple of parameters being passed to process
        Returns:
            numpy.ndarray: The processed array
    """
    # 2D Case
    if len(data.shape) == 2:
        return process(data, *args)

    # For more than two dimensions, mask fill each dimension recursively
    for i in range(len(data)):
        data[i] = apply_2d(data[i], process, *args)

    return data


def apply_2d_yxz(data, process, *args):  # , name = ""  ??? - unused!
    """ Applies a 2D process to datasets with three dimensions,
        Always applies the process to the first 2 dimensions of the dataset,
        iterating through any lower dimensions and processing up through dimensions.
        E.g., a dataset with coordinates dimensions: (lat, lon, land_cover_type),
            will apply to lat, lon dimensions, for each land cover type.
        Args:
            data (numpy.ndarray): The data array to be processed
            process: The process to be applied to data
            *args: tuple of parameters being passed to process
        Returns:
            numpy.ndarray: The processed array
    """

    # For more than two non nominal dimensions(yxz), copy the first 2 dimensions
    # get the masked array for each and combine them all together
    for i in range(data.shape[-1]):
        data[:, :, i] = process(data[:, :, i], *args)
    return data


def create_bounding_box_shape_file(bounding_box: List[float],
                                   working_directory: str) -> str:
    """ Take a bounding box in the format of [W, S, E, N] and create a GeoJSON
        polygon. Then write that polygon to a temporary file. Return the
        location of that temporary file.

        The coordinates list should be in an anticlockwise order, beginning and
        ending at the same point.

    """
    bounding_box = BBox(*bounding_box)
    coordinates = [[[bounding_box.west, bounding_box.south],
                    [bounding_box.east, bounding_box.south],
                    [bounding_box.east, bounding_box.north],
                    [bounding_box.west, bounding_box.north],
                    [bounding_box.west, bounding_box.south]]]

    bbox_geojson = {'type': 'FeatureCollection',
                    'features': [{'type': 'Feature',
                                  'geometry': {'type': 'Polygon',
                                               'coordinates': coordinates},
                                  'properties': {'name': 'Harmony bbox'}}]}
    shape_file_path = path_join(working_directory, f'{uuid4()}.geo.json')

    with open(shape_file_path, 'w', encoding='utf-8') as file_handler:
        json.dump(bbox_geojson, file_handler, indent=4)

    return shape_file_path


def get_resolved_dataframe(shape_file_path: str, transform: Affine, crs: CRS,
                           out_shape: Tuple[int, int]) -> gpd.GeoDataFrame:
    """ When data are projected, first derive the longitude and latitude
        values for all points on the grid. From this determine the smallest
        separation between diagonally adjacent points. Use this as the
        resolution of the grid in geographic space. Finally, take the input
        GeoJSON shape file and ensure that all polygon edges have points at
        the geographic resolution of the grid. The `explode` method is used
        on the input `geopandas.GeoDataFrame` to split MultiPolygon features
        into separate Polygons.

    """
    latitudes, longitudes = get_grid_lat_lons(transform, crs, out_shape)
    geographic_resolution = get_geographic_resolution(longitudes, latitudes)
    initial_gpd = gpd.read_file(shape_file_path).explode(index_parts=True)
    return get_resolved_shape(initial_gpd, geographic_resolution)


def get_geographic_resolution(longitudes: np.ndarray,
                              latitudes: np.ndarray) -> float:
    """ Calculate the distance between diagonally adjacent cells in both
        longitude and latitude. Combined those differences in quadrature to
        obtain Euclidean distances. Return the minimum of these Euclidean
        distances. Over the typical distances being considered, differences
        between the Euclidean and geodesic distance between points should be
        minimal, with Euclidean distances being slightly shorter.

    """
    lon_square_diffs = np.square(np.subtract(longitudes[1:, 1:],
                                             longitudes[:-1, :-1]))
    lat_square_diffs = np.square(np.subtract(latitudes[1:, 1:],
                                             latitudes[:-1, :-1]))
    return np.nanmin(np.sqrt(np.add(lon_square_diffs, lat_square_diffs)))


def get_grid_lat_lons(transform: Affine, crs: CRS,
                      out_shape: Tuple[int, int]) -> Tuple[np.ndarray]:
    """ Use the components of the Affine transformation matrix to perform an
        inverse transformation from array indices to projected x and y
        coordinates for all points on the input data grid. Then transform those
        points to longitudes and latitudes.

    """
    projected_y, projected_x = np.indices(out_shape)
    projected_x = np.add(np.multiply(transform.a, projected_x), transform.c)
    projected_y = np.add(np.multiply(transform.e, projected_y), transform.f)
    to_geo_transformer = Transformer.from_crs(crs, 4326)
    return to_geo_transformer.transform(projected_x, projected_y)


def get_resolved_shape(geo_dataframe: gpd.GeoDataFrame,
                       resolution: float) -> gpd.GeoDataFrame:
    """ Iterate through all features in the `geopandas.GeoDataFrame`. At this
        point, the features will all be Polygons as MultiPolygons will have
        been split in `get_resolved_dataframe`. `get_resolved_polygon` will
        return a new Polygon for each input shape, with each exterior and
        interior edge being resolved at the requested resolution.

        Finally return a new `geopandas.GeoDataFrame` at with points at the
        specified resolution.

    """
    polygons = [get_resolved_polygon(feature, resolution)
                for feature in geo_dataframe.iterfeatures()]

    return gpd.GeoDataFrame(crs='epsg:4326', geometry=polygons)


def get_resolved_polygon(feature: Dict, resolution: float) -> Polygon:
    """ Populate points around the exterior and interior rings of the supplied
        geopandas feature. These additional points will be at the resolution of
        the gridded data. Return a `shapely.geometry.Polygon` object to be used
        to create the final `geopandas.GeoDataFrame` that will define the
        `numpy` mask object.

    """
    feature_shape = shape(feature['geometry'])

    exterior_ring = get_resolved_ring(list(feature_shape.exterior.coords),
                                      resolution)

    interior_rings = [get_resolved_ring(list(interior.coords), resolution)
                      for interior in feature_shape.interiors]

    return Polygon(exterior_ring, holes=interior_rings)


def get_resolved_ring(ring_points: List[Coordinates],
                      resolution: float) -> List[Coordinates]:
    """ Iterate through all pairs of consecutive points and ensure that, if
        those points are further apart than the resolution of the input data,
        additional points are placed along that edge at regular intervals. Each
        line segment will have regular spacing, and will remain anchored at the
        original start and end of the line segment. This means the spacing of
        the points will have an upper bound of the supplied resolution, but may
        be a shorter distance to account for non-integer multiples of the
        resolution along the line.

        To avoid duplication of points, the last point of each line segment is
        not retained, as this will match the first point of the next line
        segment. `shapely` will close the output ring, meaning the first point
        does not have to be repeated in this list.

    """
    new_points = [get_resolved_line(point_one,
                                    ring_points[point_one_index + 1],
                                    resolution)[:-1]
                  for point_one_index, point_one
                  in enumerate(ring_points[:-1])]

    return [point for segment_points in new_points for point in segment_points]


def get_resolved_line(point_one: Coordinates, point_two: Coordinates,
                      resolution: float) -> List[Coordinates]:
    """ A function that takes two consecutive points from either an exterior
        or interior ring of a `shapely.geometry.Polygon` object and places
        equally spaced points along that line determined by the supplied
        geographic resolution. That resolution will be determined by the
        gridded input data.

        The resulting points will be appended to the rest of the ring,
        ensuring the ring has points at a resolution of the gridded data.

    """
    length = np.linalg.norm(np.array(point_two[:2]) - np.array(point_one[:2]))
    n_points = np.ceil(length / resolution) + 1
    new_x = np.linspace(point_one[0], point_two[0], int(n_points))
    new_y = np.linspace(point_one[1], point_two[1], int(n_points))
    return list(zip(new_x, new_y))


def get_default_fill_for_data_type(variable_type: Union[str, None]) -> Any:
    """ Retrieve a default value for filling as defined in the
        DEFAULT_FILL_VALUES dictionary. This will only be used if there is no
        _FillValue (HDF-5/NetCDF-4) or NoData (GeoTIFF) in-file metadata, no
        configuration file setting for fill value, or no user-supplied default
        fill value (SDPS implementation only).

        If the type is not recognised, the returned default value will be
        -9999.0

    """
    default_fill_values = {
        'float16': -9999.0,
        'float32': -9999.0,
        'float64': -9999.0,
        'float128': -9999.0,
        'int8': 127,
        'int16': 32767,
        'int32': 2147483647,
        'str32': '',
        'uint8': 254,
        'uint16': 65534,
        'uint32': 4294967294,
        'uint64': 18446744073709551614
    }
    return default_fill_values.get(variable_type, -9999.0)
