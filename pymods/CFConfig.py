''' Utilities for reading and interpreting the CF configuration file
    Allows processing of hdf-5 files that do not fully follow the CF conventions
    Where the configuration file provides the missing information.
'''
import json
import re

import h5py


def readConfigFile(configFile):
    """ Read the config json file
        Args:
            configFile(string): config file path
    """
    global config
    configString = open(configFile).read()
    configStringWoComments = removeComments(configString)
    config = json.loads(configStringWoComments)


def removeComments(text):
    """ Remove c-style comments.
        Args:
            txt(string): blob of text with comments (can include newlines)
        Return:
            text with comments removed
    """
    pattern = r"""
                        ##  --------- COMMENT ---------
       /\*              ##  Start of /* ... */ comment
       [^*]*\*+         ##  Non-* followed by 1-or-more *'s
       (                ##
         [^/*][^*]*\*+  ##
       )*               ##  0-or-more things which don't start with /
                        ##    but do end with '*'
       /                ##  End of /* ... */ comment
     |                  ##  -OR-  various things which aren't comments:
       (                ##
                        ##  ------ " ... " STRING ------
         "              ##  Start of " ... " string
         (              ##
           \\.          ##  Escaped char
         |              ##  -OR-
           [^"\\]       ##  Non "\ characters
         )*             ##
         "              ##  End of " ... " string
       |                ##  -OR-
                        ##
                        ##  ------ ' ... ' STRING ------
         '              ##  Start of ' ... ' string
         (              ##
           \\.          ##  Escaped char
         |              ##  -OR-
           [^'\\]       ##  Non '\ characters
         )*             ##
         '              ##  End of ' ... ' string
       |                ##  -OR-
                        ##
                        ##  ------ ANYTHING ELSE -------
         .              ##  Anything other char
         [^/"'\\]*      ##  Chars which doesn't start a comment, string
       )                ##    or escape
    """
    regex = re.compile(pattern, re.VERBOSE | re.MULTILINE | re.DOTALL)
    noncomments = [m.group(2) for m in regex.finditer(text) if m.group(2)]
    return "".join(noncomments)


def getShortName(input_file):
    """ Get product short name using config json file
        Args:
            input_file(string): input file path
        Returns:
            shortname(string): product short name
    """
    shortnamePaths = config["ShortNamePath"]
    if isinstance(input_file, str):
        inf = h5py.File(input_file, 'r')
    else:
        inf = input_file

    for path in shortnamePaths:
        if path.endswith("/"):
            path = path[:-1]

        shortnamePath = path.rpartition("/")[0]
        label = path.rpartition("/")[2]
        if shortnamePath in inf:
            if label in inf[shortnamePath].attrs:
                shortName = inf[shortnamePath].attrs[label]
                break
    return shortName


def getGridMappingGroup(shortName, datasetName):
    """ Get grid mapping group projection for CF-Compliance
        Args:
            shortName(string): product short name
        Return:
            grid mapping group projection
    """
    gridMappingGroups = config["Grid_Mapping_Group"]
    mappingGroup = ""
    for i, (key, value) in enumerate(gridMappingGroups.items()):
        if not isinstance(shortName, str):
            shortName = shortName.decode()

        if re.match(key, shortName):
            for j, (key2, value2) in enumerate(value.items()):
                if re.match(key2, datasetName):
                    mappingGroup = value2
                    break
            break
    return mappingGroup


def getGridMappingData(mappingGroup):
    """ Get grip mapping data for CF-Compliance
        Args:
            mappingGroup(string): mapping group projection
        Return:
            grid mapping information
    """
    gridMappingData = config["Grid_Mapping_Data"]
    for i, (key, value) in enumerate(gridMappingData.items()):
        if re.match(key, mappingGroup):
            return value
