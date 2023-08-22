__all__ = ["download_file"]

import os

import requests
from tempfile import NamedTemporaryFile

from .base import server


def download_file(remote_file_name: str):
    url = f"{server}/download/science?file={remote_file_name}"
    with NamedTemporaryFile(delete=False, mode="wb") as temp_file:
        response = requests.get(url)
        file_size = int(response.headers.get("content-length"))
        temp_file.write(response.content)
        download_size = os.path.getsize(file_name := temp_file.name)

        if file_size == download_size:
            return file_name
        else:
            return None


if __name__ == "__main__":
    file = "mms1_fgm_srvy_l2_20170728_v5.98.0.cdf"
    temp_file = download_file(file)
