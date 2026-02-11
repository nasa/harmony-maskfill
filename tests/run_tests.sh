#!/bin/sh

#####################################
#
# A script invoked by the Dockerfile to run unit tests.
#
#####################################
# Exit status used to report back to caller
STATUS=0

coverage run -m xmlrunner discover tests -o reports/tests-reports

RESULT=$?
if [ "$RESULT" -ne "0" ]; then
    STATUS=1
    echo "ERROR: unittest generated errors"
fi

echo "\n\n"

echo "Test Coverage Estimates"
coverage report --omit="*tests/*"
coverage html --omit="*tests/*" -d reports/coverage

exit $STATUS
