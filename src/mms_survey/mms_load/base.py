__all__ = ["BaseLoader"]

import os
import logging
from abc import ABC, abstractmethod
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import requests
from pathos.threading import ThreadPool
from tqdm.contrib.logging import tqdm_logging_redirect as tqdm

from mms_survey.utils.units import u_
from mms_survey.utils.io import dataset_is_processed


class BaseLoader(ABC):
    r"""Base class for querying data from LASP MMS Science Data Center"""
    server = "https://lasp.colorado.edu/mms/sdc/public/files/api/v1"

    def __init__(
        self,
        start_date: str = "2017-07-26",
        end_date: str = "2017-07-26",
        probe: None | str | list = None,
        instrument: None | str | list = None,
        data_rate: None | str | list = None,
        data_type: None | str | list = None,
        data_level: None | str | list = None,
        product: None | str | list = None,
        query_type: str = "science",
        skip_processed_data: bool = False,
    ):
        r"""
        Class instantiation requires MMS SDC query parameters
        (see https://lasp.colorado.edu/mms/sdc/public/about/how-to/)

        Parameters
        ----------
        start_date: str
            Query start date (MMS SDC API equivalence: start_date)
        end_date: str
            Query end date (MMS SDC API equivalence: end_date)
        probe: None (default), str, or list of str
            MMS probe name (MMS SDC API equivalence: sc_id)
        instrument: None (default), str, or list of str
            MMS instrument name (MMS SDC API equivalence: instrument_id)
        data_rate: None (default), str, or list of str
            Data rate mode (MMS SDC API equivalence: data_rate_mode)
        data_type: None (default), str, or list of str
            Data descriptor (MMS SDC API equivalence: descriptor)
        data_level: None (default), str, or list of str
            Data level (MMS SDC API equivalence: data_level)
        product: None (default), str, or list of str
            Ancillary product (MMS SDC API equivalence: product)
        query_type: str
            Type of query ("science" or "ancillary")
        skip_processed_data: bool
            Tag for skipping processed data
        """
        self.start_date = start_date
        self.end_date = end_date
        self.probe = probe
        self.instrument = instrument
        self.data_rate = data_rate
        self.data_type = data_type
        self.data_level = data_level
        self.product = product
        self.query_type = query_type
        self.skip_processed_data = skip_processed_data

    def get_payload(self, file: None | str = None) -> dict:
        r"""
        Construct HTTP payload from class properties, which are
        ignored if a particular file name is provided.

        Parameter
        ---------
        file: None (default) or str
            File name (MMS SDC API equivalence: file)

        Return
        ------
        payload: dict
            Dictionary for HTTP request containing payload information
        """
        if file is None:
            payload = {
                "start_date": self.start_date,
                "end_date": self.end_date,
                "sc_id": self.probe,
                "instrument_id": self.instrument,
                "data_rate_mode": self.data_rate,
                "descriptor": self.data_type,
                "data_level": self.data_level,
                "product": self.product,
            }
        else:
            payload = {"file": file}
        return payload

    def get_file_list(self) -> list:
        r"""
        Return list of file names relevant to payload information

        Return
        ------
        file_list: list
            List of file names
        """
        response = requests.get(
            url=f"{self.server}/file_info/{self.query_type}",
            params=self.get_payload(),
            timeout=10.0,
        )
        response.raise_for_status()

        file_info = response.json()["files"]
        file_list = list(map(lambda x: x["file_name"], file_info))
        file_size = sum(list(map(lambda x: x["file_size"], file_info))) * u_.B
        logging.info(
            f"{self.string_id} Found {len(file_list)} files,"
            f" total size = {file_size.to(u_.GB):.4f}."
        )
        return file_list

    def download_file(self, file: str) -> str:
        local_file_name = None

        # `local_file_name` is set here if `file` downloads successfully
        with NamedTemporaryFile(delete=False, mode="wb") as temp_file:
            temp_file_name = temp_file.name
            for _ in range(3):
                try:
                    response = requests.get(
                        url=f"{self.server}/download/{self.query_type}",
                        params=self.get_payload(file=file),
                        timeout=60.0,
                    )
                    file_size = int(response.headers.get("content-length"))
                    temp_file.write(response.content)
                    download_size = os.path.getsize(temp_file_name)
                    if file_size == download_size:
                        local_file_name = temp_file_name
                        break
                except (
                    requests.ConnectionError,
                    requests.HTTPError,
                    requests.Timeout,
                ):
                    pass
            else:
                logging.warning(f"Giving up downloading {file}!")

        if local_file_name is None:
            os.unlink(temp_file_name)
        return local_file_name

    def should_skip(self, group: str) -> bool:
        """Called before download to determine if dataset is present"""
        return self.skip_processed_data and dataset_is_processed(group)

    @abstractmethod
    def get_metadata(self, file: str) -> dict:
        raise NotImplementedError()

    @abstractmethod
    def process_file(self, file: str, metadata: dict):
        """Called after download to process data"""
        raise NotImplementedError()

    def download(self, parallel: int = 1, dry_run: bool = False):
        def _helper(file: str):
            metadata = self.get_metadata(file)
            if self.should_skip(metadata["group"]):
                logging.info(f"{file} already processed. Skipping...")
                return

            temp_file = self.download_file(file)
            if temp_file is not None:
                self.process_file(temp_file, metadata)
                os.unlink(temp_file)
                logging.info(f"Processed {file}")

        file_list = self.get_file_list()
        if dry_run:
            return

        with ThreadPool(nodes=parallel) as pool:
            with tqdm(total=len(file_list), dynamic_ncols=True) as pbar:
                for _ in pool.uimap(_helper, file_list):
                    pbar.update()

    @property
    def string_id(self) -> str:
        if self.query_type == "science":
            return f"({self.probe}-{self.instrument} query)"
        else:
            return f"({self.product} query)"

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

    @property
    def product(self) -> None | str:
        if self._product is None:
            return None
        else:
            return ",".join(self._product)

    @product.setter
    def product(self, product: None | str | list):
        assert product is None or isinstance(
            product, (str, list)
        ), "Incorrect type for `product` input"
        if isinstance(product, str):
            product = [product]

        self._product = product
