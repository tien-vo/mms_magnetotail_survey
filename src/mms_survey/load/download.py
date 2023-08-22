__all__ = ["download_file"]

import os

import urllib.request
from tempfile import NamedTemporaryFile

from .base import server


def download_file(remote_file_name: str):
    with NamedTemporaryFile(delete=False, mode="wb") as temp_file:
        url = f"{server}/download/science?file={remote_file_name}"
        urllib.request.urlretrieve(url, file_name := temp_file.name)

    return file_name


if __name__ == "__main__":
    file = "mms1_fgm_srvy_l2_20170728_v5.98.0.cdf"
    temp_file = download_file(file)
