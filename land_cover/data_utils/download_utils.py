import requests
import os
import cgi
from os import PathLike
import logging
from tqdm import tqdm


logger = logging.getLogger(__file__)


def login_with_csrf_form(
    url: str,
    username: str,
    password: str,
    username_form_name: str,
    password_form_name: str,
) -> requests.Session:
    """
    Opens a session using a form that has CSRF protection.

    :param url: The url of the login page
    :param username: The user name
    :param password: The user password
    :param username_form_name: The 'name' field of the form for the username input
    :param password_form_name: The 'name' field of the form for the password input
    :return: the corresponding Session object

    """

    # Opens a session at the given URL
    session = requests.session()
    session.get(url)

    # Fill the form data with user credentials
    login_data = {
        username_form_name: username,
        password_form_name: password,
        "next": "/",
    }

    # Handle the case where the form is protected by a CSRF Middleware Token
    if (csrftoken := session.cookies.get("csrftoken", None)) is not None:
        login_data["csrfmiddlewaretoken"] = csrftoken

    # Login & raise an exception if a problem occurs
    response = session.post(url, data=login_data, headers=dict(Referer=url))
    response.raise_for_status()

    return session


def download_file(
    url: str,
    session: requests.Session,
    target_directory: PathLike,
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Download the file located at a given url using a web session. Returns the corresponding file path on the local
    system

    :param url: The URL corresponding to the file to download
    :param session: The bounded session
    :param target_directory: The local directory in which we want to download the files
    :param chunk_size: The block size in bytes for incoming streaming data
    :return: The path to the downloaded file
    """
    # First get the filename using header only
    response = session.head(url)
    response.raise_for_status()

    # Parse the headers
    try:
        header = response.headers["Content-Disposition"]
        _, params = cgi.parse_header(header)
        filename = params["filename"]
    except KeyError:
        logger.exception(
            "Could not find the right header or the filename. Make sure you are logged"
        )
        raise

    # Build up the file path
    file_path = os.path.join(target_directory, filename)

    # Don't download if the file already exist
    if not os.path.isfile(file_path):

        # Get a streaming response, as the downloaded file can be huge
        with session.get(url, stream=True) as response:
            response.raise_for_status()

            print(f"Download '{filename}' at '{url}'")
            print(f"You can find this file at: '{file_path}'")

            # # Setup progress bar
            total_size_in_bytes = int(response.headers.get("content-length", 0))

            with tqdm(
                total=total_size_in_bytes,
                unit="iB",
                unit_scale=True,
                position=0,
                leave=False,
            ) as progress_bar:
                # Dump the dataset in the data folder
                with open(file_path, "wb") as file:
                    # Iterate on chunks of the response
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        progress_bar.update(len(chunk))
                        file.write(chunk)

            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                logger.error(
                    "ERROR, mismatch between downloaded file size and expected file size"
                )

    return file_path
