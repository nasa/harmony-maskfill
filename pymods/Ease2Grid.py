""" Class file for EASE-2 Grid objects, providing the essential corner-point coordinates
    (projected meters), pixel-size, AFFINE transformation matrix and CRS transformation
    object for the standard set of EASE-2 grids.  Initialized and defined in terms of the
    base resolution: 36K, 12K, 9K, 3K, 1K.
"""
from enum import Enum

import affine  # Affine Transform Matrix support, from Index to Projected-Meters math
from pyproj import CRS, Transformer  # Coordinate Reference System, Projected-Meters to/from degrees

GLOBAL_PROJ = CRS.from_proj4("+ellps=WGS84 +datum=WGS84 +proj=cea  +lat_ts=30")
POLAR_PROJ = CRS.from_proj4("+ellps=WGS84 +datum=WGS84 +proj=laea +lat_0=90")
LAT_LONG_PROJ = CRS.from_epsg(4326)

# Transform from cea to lat/long coordinates
GLOBAL_CRS_TRANSFORM = Transformer.from_crs(GLOBAL_PROJ, LAT_LONG_PROJ)
# Transform from laea to lat/long coordinates
POLAR_CRS_TRANSFORM = Transformer.from_crs(POLAR_PROJ, LAT_LONG_PROJ)


class Ease2GridResolution(Enum):
    r_3K = 3000
    r_5K = 5000
    r_6_25K = 6250
    r_9K = 9000
    r_10K = 10_000
    r_12_5K = 12_500
    r_25K = 25_000
    r_36K = 36_000
    r_100K = 100_000


class Ease2GridType(Enum):
    t_polar = 'LAEA'
    t_global = 'CEA'


EASE2_GRIDS = {
    Ease2GridType.t_global: {
        Ease2GridResolution.r_36K:
            {'cols': 964, 'rows': 406,
             'ul_x': -17_367_530.4, 'ul_y': 7_314_540.83,
             'cell_width': 36_032.2208, 'cell_height': -36_032.2208},
        Ease2GridResolution.r_9K:
            {'cols': 3_856, 'rows': 1_624,
             'ul_x': -17_367_530.4, 'ul_y': 7_314_540.83,
             'cell_width': 9_008.05521, 'cell_height': -9_008.05521},
        Ease2GridResolution.r_3K:
            {'cols': 11_568, 'rows': 1_624,
             'ul_x': -17_367_530.4, 'ul_y': 7_314_540.83,
             'cell_width': 3_002.68507, 'cell_height': -3_002.68507}
    },
    Ease2GridType.t_polar: {
        Ease2GridResolution.r_36K:
            {'cols': 500, 'rows': 500,
             'ul_x': -900_000, 'ul_y': 900_000,
             'cell_width': 36_000, 'cell_height': -36_000},
        Ease2GridResolution.r_9K:
            {'cols': 2_000, 'rows': 2_000,
             'ul_x': -900_000, 'ul_y': 900_000,
             'cell_width': 9_000, 'cell_height': -9_000},
        Ease2GridResolution.r_3K:
            {'cols': 6_000, 'rows': 6_000,
             'ul_x': -900_000, 'ul_y': 900_000,
             'cell_width': 3_000, 'cell_height': -3_000}
    }
}


class Ease2Grid:
    def __init__(self, resolution: Ease2GridResolution,
                 grid_type: Ease2GridType):
        self.resolution = resolution
        self.grid_type = grid_type

        try:
            grid_dictionary = EASE2_GRIDS[grid_type][resolution]
        except Exception as e:
            raise ValueError(f"Grid Type and Resolution: {grid_type}, {resolution}"
                             " - are not yet supported \n"
                             f"{e}")

        # set in sub-functions:
        self.cols, self.rows = grid_dictionary['cols'], grid_dictionary['rows']
        self.ul_x, self.ul_y = grid_dictionary['ul_x'], grid_dictionary['ul_y']
        self.cell_width, self.cell_height = grid_dictionary['cell_width'], grid_dictionary['cell_height']

        self.crs_transform = GLOBAL_CRS_TRANSFORM \
            if grid_type == Ease2GridType.t_global \
            else POLAR_CRS_TRANSFORM

        self.affine_transform = \
            affine.Affine(self.cell_width, 0, self.ul_x, 0, self.cell_height, self.ul_y)


# process main
if __name__ == '__main__':
    def print_vars(obj: object):
        attrs = vars(obj)
        print('\n '.join("%s: %s" % item for item in attrs.items()))

    print_vars(Ease2Grid(Ease2GridResolution.r_36K, Ease2GridType.t_global))
    print_vars(Ease2Grid(Ease2GridResolution.r_9K, Ease2GridType.t_global))
    print_vars(Ease2Grid(Ease2GridResolution.r_3K, Ease2GridType.t_global))

    print_vars(Ease2Grid(Ease2GridResolution.r_36K, Ease2GridType.t_polar))
    print_vars(Ease2Grid(Ease2GridResolution.r_9K, Ease2GridType.t_polar))
    print_vars(Ease2Grid(Ease2GridResolution.r_3K, Ease2GridType.t_polar))
