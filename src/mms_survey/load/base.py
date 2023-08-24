__all__ = ["BaseLoader"]

from abc import ABC, abstractmethod

import pandas as pd
import requests
from pathos.pools import ProcessPool as Pool
from tqdm import tqdm

BYTE_TO_GIGABYTE = 9.313e-10
server = "https://lasp.colorado.edu/mms/sdc/public/files/api/v1"


class BaseLoader(ABC):
    server = server

    def __init__(
        self,
        start_date: str = "2017-07-26",
        end_date: str = "2017-07-26",
        probe: None | str | list = "mms1",
        instrument: None | str | list = "fgm",
        data_rate: None | str | list = "srvy",
        data_type: None | str | list = None,
        data_level: None | str | list = "l2",
        query_type: str = "science",
        skip_ok_dataset: bool = False,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.probe = probe
        self.instrument = instrument
        self.data_rate = data_rate
        self.data_type = data_type
        self.data_level = data_level
        self.query_type = query_type
        self.skip_ok_dataset = skip_ok_dataset

    @staticmethod
    @abstractmethod
    def process(file: str):
        raise NotImplementedError()

    def download(self, parallel=True, dry_run=False):
        files = self.get_remote_files()
        if dry_run:
            return

        with Pool(8 if parallel else 1) as pool:
            with tqdm(total=len(files)) as progress_bar:
                for msg in pool.uimap(self.process, files):
                    progress_bar.write(msg)
                    progress_bar.update()

    def get_remote_files(self) -> list:
        url = f"{self.server}/file_info/{self.query_type}?{self.query_string}"
        response = requests.get(url)
        assert response.ok, "Query error!"

        files = response.json()["files"]
        remote_file_names = list(map(lambda x: x["file_name"], files))
        download_size = sum(list(map(lambda x: x["file_size"], files)))
        print(
            f"{self._id}: Query found {len(files)} files with"
            f" total size = {BYTE_TO_GIGABYTE * download_size:.4f} GB",
            flush=True,
        )
        return remote_file_names

    @property
    def _id(self) -> str:
        return (
            f"({self.probe},{self.instrument},{self.data_rate},"
            f"{self.data_type},{self.data_level})"
        )

    @property
    def query_string(self) -> str:
        s = f"start_date={self.start_date}&end_date={self.end_date}"
        if self.probe is not None:
            s += f"&sc_ids={self.probe}"
        if self.instrument is not None:
            s += f"&instrument_ids={self.instrument}"
        if self.data_rate is not None:
            s += f"&data_rate_modes={self.data_rate}"
        if self.data_level is not None:
            s += f"&data_levels={self.data_level}"
        if self.data_type is not None:
            s += f"&descriptor={self.data_type}"

        return s

    @property
    def start_date(self) -> str:
        if self._data_rate is None or (
            ("srvy" not in self._data_rate) and ("fast" not in self._data_rate)
        ):
            return self._start_date.strftime("%Y-%m-%d-%H-%M-%S")
        else:
            return self._start_date.strftime("%Y-%m-%d")

    @start_date.setter
    def start_date(self, date: str):
        assert isinstance(date, str), "Incorrect type for `date` input!"

        self._start_date = pd.to_datetime(date)

    @property
    def end_date(self) -> str:
        if self._data_rate is None or (
            ("srvy" not in self._data_rate) and ("fast" not in self._data_rate)
        ):
            return self._end_date.strftime("%Y-%m-%d-%H-%M-%S")
        else:
            return self._end_date.strftime("%Y-%m-%d")

    @end_date.setter
    def end_date(self, date: str):
        assert isinstance(date, str), "Incorrect type for `date` input"

        self._end_date = pd.to_datetime(date)

    @property
    def probe(self) -> None | str:
        if self._probe is None:
            return None
        else:
            return ",".join(self._probe)

    @probe.setter
    def probe(self, probe: None | str | list):
        assert probe is None or isinstance(
            probe, (str, list)
        ), "Incorrect type for `probe` input!"
        if isinstance(probe, str):
            probe = [probe]

        self._probe = probe

    @property
    def instrument(self) -> None | str:
        if self._instrument is None:
            return None
        else:
            return ",".join(self._instrument)

    @instrument.setter
    def instrument(self, instrument: None | str | list):
        assert instrument is None or isinstance(
            instrument, (str, list)
        ), "Incorrect type for `instrument` input"
        if isinstance(instrument, str):
            instrument = [instrument]

        self._instrument = instrument

    @property
    def data_rate(self) -> None | str:
        if self._data_rate is None:
            return None
        else:
            return ",".join(self._data_rate)

    @data_rate.setter
    def data_rate(self, data_rate: None | str | list):
        assert data_rate is None or isinstance(
            data_rate, (str, list)
        ), "Incorrect type for `data rate` input"
        if isinstance(data_rate, str):
            data_rate = [data_rate]

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
    def data_level(self) -> None | str:
        if self._data_level is None:
            return None
        else:
            return ",".join(self._data_level)

    @data_level.setter
    def data_level(self, data_level: None | str | list):
        assert data_level is None or isinstance(
            data_level, (str, list)
        ), "Incorrect type for `data level` input"
        if isinstance(data_level, str):
            data_level = [data_level]

        self._data_level = data_level
