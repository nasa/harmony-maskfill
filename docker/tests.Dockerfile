#
# Test image for the Harmony MaskFill service. This image uses the main service
# image, ghcr.io/nasa/harmony-maskfill, as a base layer for the tests. This
# ensures that the contents of the service image are tested, preventing
# discrepancies between the service and test environments.
#
# The results of the test run will be saved to tests/reports, which should be
# mounted as a shared volume with the host.
#
# Commands to use this file locally:
#
# docker build -f docker/tests.Dockerfile -t ghcr.io/nasa/harmony-maskfill-test .
# docker run -v /full/path/to/host/directory/test-reports:/home/tests/reports ghcr.io/nasa/harmony-maskfill-test:latest
#
# 2021-06-25: Updated
# 2025-09-15: Updated for migration to GitHub and GHCR images.
# 2025-09-16: Updated to install test dependencies.
# 2026-09-24: Removed conda environment, as the service image is now Pip-only.
#
FROM ghcr.io/nasa/harmony-maskfill

# Install additional Pip requirements (for testing)
COPY tests/pip_test_requirements.txt .
RUN pip install --no-input --no-cache-dir -r pip_test_requirements.txt

# Copy test directory containing Python unittest suite, test data and utilities
COPY ./tests tests

# An environment variable used by BaseHarmonyAdapter uses to not stage files
ENV ENV=test

# Configure a container to be executable via the `docker run` command.
ENTRYPOINT ["/home/tests/run_tests.sh"]
