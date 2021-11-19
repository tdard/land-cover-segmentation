import os
import tempfile

from land_cover.data_utils.download_utils import login_with_csrf_form, download_file

USER_NAME = os.environ["CHALLENGE_USERNAME"]
USER_PWD = os.environ["CHALLENGE_PWD"]
CHALLENGE_LOGIN_URL = "https://challengedata.ens.fr/login/"
CHALLENGE_DATA_URL = "https://challengedata.ens.fr/participants/challenges/48/download/custom-metric-file"  # 2.3 Kb


def test_login_with_csrf_form():
    # If no exception is raised, then it means a session can be established
    session = login_with_csrf_form(CHALLENGE_LOGIN_URL, USER_NAME, USER_PWD, "username", "password")


def test_download_file():
    with tempfile.TemporaryDirectory() as dirpath:
        session = login_with_csrf_form(CHALLENGE_LOGIN_URL, USER_NAME, USER_PWD, "username", "password")

        file_name = download_file(CHALLENGE_DATA_URL, session, target_directory=dirpath)

        assert file_name.endswith(".py")
