import os.path
import pathlib

import pytest

from land_cover.data_utils.file_utils import uncompress_zip
import zipfile
import tempfile


@pytest.mark.parametrize("keep_zip_name", [(True,), (False,)])
def test_uncompress_zip(keep_zip_name):

    # Create temporary directory for the test
    with tempfile.TemporaryDirectory() as dirpath:

        # Create a dummy file
        dummy_file = os.path.join(dirpath, "dummy.txt")
        with open(dummy_file, "w") as dummy:
            dummy.write("I am dummy")

        # Create a zip file and add the dummy file to it
        file_name = os.path.join(dirpath, "sample_file.zip")
        with zipfile.ZipFile(file_name, "a") as file:
            file.write(dummy_file, os.path.basename(dummy_file))

        # Remove the dummy txt file
        os.remove(dummy_file)

        # Try to uncompress the zip file in the temporary directory
        result = uncompress_zip(file_name, dirpath, keep_zip_name=keep_zip_name)
        assert os.path.isdir(result)

        if keep_zip_name:
            assert pathlib.Path(result).name == "sample_file"
        else:
            assert pathlib.Path(result).name == pathlib.Path(dirpath).name
