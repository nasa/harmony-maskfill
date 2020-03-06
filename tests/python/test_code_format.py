from pathlib import Path
from unittest import TestCase

import pycodestyle


class TestCodeFormat(TestCase):

    @classmethod
    def setUpClass(test_class):
        test_class.python_files = [file_name
                                   for file_name
                                   in Path('.').rglob('*.py')]

    def test_pep8_conformance(self):
        """Test that Python files conform to PEP8.

        E501 is ignored - this is the line length, which defaults to 80
        characters.

        """
        pep8style = pycodestyle.StyleGuide(ignore='E501,W504')
        result = pep8style.check_files(self.python_files)
        self.assertEqual(result.total_errors, 0,
                         'Found code style errors (and warnings).')
