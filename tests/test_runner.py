''' test_runner - created to assist in debugging unit tests
'''
from tests.python.unit.test_H5MaskFill import TestH5MaskFill
from pymods import CFConfig

tester = TestH5MaskFill()

tester.setUp()
CFConfig.readConfigFile('tests/data/MaskFillConfig.json')

tester.test_get_coordinates()
tester.test_get_exclusions()
tester.test_no_exclusions()

tester.tearDown()
