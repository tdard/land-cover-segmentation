import os.path
import pathlib
import zipfile
from os import PathLike
from typing import Union, Optional


def uncompress_zip(
    file_path: Union[str, PathLike],
    target_dir: Optional[Union[str, PathLike]] = None,
    keep_zip_name: bool = True,
) -> str:
    """
    Uncompress a zip file located at file_path in the specified target directory.
    If keep_zip_name is specified, create an additional folder at the target directory that keeps the base name of the
    zip file (without the extension) and in which the data will be placed.

    :param file_path:
    :param target_dir:
    :param keep_zip_name:
    :return: the path to the uncompressed folder
    """

    # For a zip file 'archive.zip' and a target directory '/target/', build the desired target directory:
    # '/target/archive/' in which the unzipped data will be put in
    if keep_zip_name:
        if isinstance(file_path, str):
            name_without_extension = pathlib.Path(file_path).stem
        else:  # file_path is a Path
            assert isinstance(file_path, pathlib.Path)
            name_without_extension = file_path.stem

        if target_dir is not None:
            target_dir = os.path.join(target_dir, name_without_extension)
        else:
            target_dir = name_without_extension

    # If the directory does not exist, create it
    if not os.path.isdir(target_dir):
        if ".." in target_dir:
            # Create and explicit path removes the double dots. It is good as it creates problem with makedirs function.
            target_dir = pathlib.Path(target_dir)
        os.makedirs(target_dir, exist_ok=True)

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        print(
            f"Extract content of '{file_path}' in '{'cwd' if target_dir is None else target_dir}'"
        )
        zip_ref.extractall(target_dir)

    return target_dir
