## v0.1.7
### 2025-02-20

This version of MaskFill adds a coordinate override rule to the configuration
file and implements use of this rule to point SPL3SMP and SPL3SMP_E PM
variables from AM coordinates to the coordinate variables in their group.

## v0.1.6
### 2024-11-11

This version of MaskFill adds additional prefix to shortname mappings for
SMAP L3 collections and adds handling for fake dimensions that are
created by OPeNDAP/NetCDF-Library when anonymous dimensions exist.

## v0.1.5
### 2024-08-05

This version of MaskFill adds rules to the configuration file to ensure the
service does not attempt to mask the `/EASE2_global_projection` variable in
SMAP L4 products. Additional file path prefixes are also added to ensure the
service can recognise output from OPeNDAP (via HOSS) as belonging to those
collections.

## v0.1.4
### 2024-02-05

This version of MaskFill contains a bug fix to ensure that if there is no
configuration file, in-file metadata, or user-specified fill value, the default
fill value applied will vary based on the variable data type. Prior to this
change a global default of -9999.0 was applied. However, this is not an
appropriate value for some integer types, where -9999.0 will be cast to the
correct data type and become an incorrect value (e.g., 241 instead of 254).

## v0.1.3
### 2022-11-10

This version of MaskFill contains a bug fix to ensure that if the input array
dimensions for x and y are equal, the same single dimension will not be
retrieved for both. In this case, it is assumed that dimensions adhere to the
standard Python indexing system of (row, column).

## v0.1.2
### 2022-09-07

This version of MaskFill updates the configuration file to make the service
compatible with ABoVE collections. These collections will invoke MaskFill via
the HOSS Projection-Gridded service chain, when spatial subsetting is required.

## v0.1.1
### 2022-08-03

This version of MaskFill updates the configuration file to ensure the service
can recognise coordinate variables for SPL3FTP granules that contain their
native 3-D bands, instead of flattened output.

## v0.1.0
### 2022-07-29

This version of MaskFill allows the inbound Harmony message to specify a
bounding box. Within the service, this bounding box will be converted to a
polygon and points will be placed along the edges of that polygon as determined
by the minimum geographic resolution of the spatial grid. This feature is of
use for projection-gridded collections, and allows them to be spatially
subsetted with a bounding box, via a combination of the Harmony OPeNDAP
SubSetter (HOSS) and MaskFill.

## v0.0.4
### 2022-07-12

This version of MaskFill updates a number of dependencies primarily resulting
from a Snyk vulnerability reported within the version of GDAL previously being
used by the service. This also updates the on-premises environment to
"MaskFill_0.0.2".

## v0.0.3
### 2022-05-30

This version of MaskFill updates the configuration file to include settings for
RSSMIF16D, ensuring the correct fill values will be applied to each variable.

## v0.0.2
### 2022-05-23

This version of MaskFill updates the types of input file that can be processed
to include NetCDF-4 outputs. NetCDF-4 files will be processed by the HDF-5
branch of the code and are the expected output format from the Harmony OPeNDAP
SubSetter (HOSS).

## v0.0.1
### 2022-01-06

This version of MaskFill adds semantic version number tagging to the Harmony
service Docker image.
