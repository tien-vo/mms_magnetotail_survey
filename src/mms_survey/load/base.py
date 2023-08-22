__all__ = ["BaseDownload"]

import warnings
from abc import ABC, abstractmethod

import psutil
import requests
import pandas as pd
from tqdm import tqdm
from pathos.pools import ProcessPool as Pool

BYTE_TO_GIGABYTE = 9.313e-10
server = "https://lasp.colorado.edu/mms/sdc/public/files/api/v1"


class BaseDownload(ABC):

    def __init__(
        self,
        start_date: str = "2017-07-26",
        end_date: str = "2017-07-26",
        probe: str | list = "mms1",
        instrument: str | list = "fgm",
        data_rate: str | list = "srvy",
        data_type: None | str | list = None,
        data_level: str | list = "l2",
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.probe = probe
        self.instrument = instrument
        self.data_rate = data_rate
        self.data_type = data_type
        self.data_level = data_level

    @abstractmethod
    def process(self, file: str):
        raise NotImplementedError()

    def download(self, parallel=True, dry_run=False):
        files = self.get_remote_files()
        if dry_run:
            return

        pool = Pool() if parallel else Pool(1)
        with tqdm(total=len(files)) as process_bar:
            for _ in pool.uimap(self.process, files):
                process_bar.update()

        pool.close()

    def get_remote_files(self) -> list:
        url = f"{server}/file_info/science?{self._query_string}"
        response = requests.get(url)
        assert response.ok, "Query error!"

        files = response.json()["files"]
        remote_file_names = list(map(lambda x: x["file_name"], files))
        download_size = sum(list(map(lambda x: x["file_size"], files)))
        free_disk_space = psutil.disk_usage("/home").free
        print(
            f"{self._id}: Query found {len(files)} files with total size = "
            f"{BYTE_TO_GIGABYTE * download_size:.4f} GB"
        )
        assert download_size < 0.95 * free_disk_space, "Download size too large"

        return remote_file_names

    @property
    def _id(self) -> str:
        return f"({self.probe},{self.instrument},{self.data_rate})"

    @property
    def _query_string(self) -> str:
        s = (
            f"start_date={self.start_date}"
            f"&end_date={self.end_date}"
            f"&sc_ids={self.probe}"
            f"&instrument_ids={self.instrument}"
            f"&data_rate_modes={self.data_rate}"
            f"&data_levels={self.data_level}"
        )
        if self.data_type is not None:
            s += f"&descriptor={self.data_type}"

        return s

    @property
    def start_date(self) -> str:
        if ("srvy" in self._data_rate) or ("fast" in self._data_rate):
            return self._start_date.strftime("%Y-%m-%d")
        else:
            return self._start_date.strftime("%Y-%m-%d-%H-%M-%S")

    @start_date.setter
    def start_date(self, date: str):
        assert isinstance(date, str), "Incorrect type for date input"

        self._start_date = pd.to_datetime(date)

    @property
    def end_date(self) -> str:
        if ("srvy" in self._data_rate) or ("fast" in self._data_rate):
            return self._end_date.strftime("%Y-%m-%d")
        else:
            return self._end_date.strftime("%Y-%m-%d-%H-%M-%S")

    @end_date.setter
    def end_date(self, date: str):
        assert isinstance(date, str), "Incorrect type for date input"

        self._end_date = pd.to_datetime(date)

    @property
    def probe(self) -> str:
        return ",".join(self._probe)

    @probe.setter
    def probe(self, probe: str | list):
        assert isinstance(probe, (str, list)), "Incorrect type for probe input"
        if isinstance(probe, str):
            probe = [probe]

        self._probe = probe

    @property
    def instrument(self) -> str:
        return ",".join(self._instrument)

    @instrument.setter
    def instrument(self, instrument: str | list):
        assert isinstance(
            instrument, (str, list)
        ), "Incorrect type for instrument input"
        if isinstance(instrument, str):
            instrument = [instrument]

        self._instrument = instrument

    @property
    def data_rate(self) -> str:
        return ",".join(self._data_rate)

    @data_rate.setter
    def data_rate(self, data_rate: str | list):
        assert isinstance(
            data_rate, (str, list)
        ), "Incorrect type for data rate input"
        if isinstance(data_rate, str):
            data_rate = [data_rate]
        elif ("srvy" in data_rate) or ("fast" in data_rate):
            warnings.warn(
                "Requesting both fast survey and burst data."
                " Time range will be truncated to %Y-%m-%d format"
                " to prioritize survey data. It is recommended to"
                " query fast survey and burst data separately."
            )

        self._data_rate = data_rate

    @property
    def data_type(self) -> None | str:
        if self._data_type is None:
            return None
        else:
            return ",".join(self._data_type)

    @data_type.setter
    def data_type(self, data_type: None | str | list):
        assert data_type is None or isinstance(
            data_type, (str, list)
        ), "Incorrect type for data type input"
        if isinstance(data_type, str):
            data_type = [data_type]

        self._data_type = data_type

    @property
    def data_level(self) -> str:
        return ",".join(self._data_level)

    @data_level.setter
    def data_level(self, data_level: str | list):
        assert isinstance(
            data_level, (str, list)
        ), "Incorrect type for data level input"
        if isinstance(data_level, str):
            data_level = [data_level]

        self._data_level = data_level
