#!/bin/bash
##################################################################################
#
# Extract release notes for the latest version of Harmony MaskFill service
#
# 2025-09-15: Updated from the Harmony Metadata Annotator
#
##################################################################################

CHANGELOG_FILE="CHANGELOG.md"

## captures versions
## >## v1.0.0
## >## [v1.0.0]
VERSION_PATTERN="^## [\[]v"

## captures url links
## [v1.0.0]:https://github.com/nasa/harmony-maskfill/releases/tag/1.0.0
LINK_PATTERN="^\[.*\].*releases/tag/.*"

# Read the file and extract text between the first two occurrences of the
# VERSION_PATTERN
result=$(awk "/$VERSION_PATTERN/{c++; if(c==2) exit;} c==1" "$CHANGELOG_FILE")

# Print the result
echo "$result" |  grep -v "$VERSION_PATTERN" | grep -v "$LINK_PATTERN"
