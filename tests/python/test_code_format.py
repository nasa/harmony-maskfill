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

        Ignored errors and warnings:

        * E501: Line length, which defaults to 80 characters.
        * E722: `try`/`except` catching bare exceptions. Better to not do, but
            not always clear which exceptions third-party packages will raise.
        * W503: Break before binary operator. Have to ignore one of W503 or W504
            to allow for breaking of some long lines. PEP8 suggests breaking the
            line before a binary operator is more "Pythonic".

        """
        pep8style = pycodestyle.StyleGuide(ignore=['E501', 'E722', 'W503'])
        result = pep8style.check_files(self.python_files)
        self.assertEqual(result.total_errors, 0,
                         'Found code style errors (and warnings).')
