#!/bin/sh

#####################################
#
# A script invoked by the Dockerfile to run unit tests.
#
#####################################

coverage run -m xmlrunner discover tests -o reports/tests-reports

echo "\n\n"

echo "Test Coverage Estimates"
coverage report --omit="*tests/*"
coverage html --omit="*tests/*" -d reports/coverage
