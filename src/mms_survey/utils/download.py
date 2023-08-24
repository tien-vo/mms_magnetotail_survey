__all__ = ["download"]

import os
from tempfile import NamedTemporaryFile

import requests


def download(url: str):
    with NamedTemporaryFile(delete=False, mode="wb") as temp_file:
        response = requests.get(url)
        file_size = int(response.headers.get("content-length"))
        temp_file.write(response.content)
        download_size = os.path.getsize(file_name := temp_file.name)

    if file_size == download_size:
        return file_name
    else:
        return None
