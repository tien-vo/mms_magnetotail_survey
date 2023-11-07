import logging
import os
from abc import ABC, abstractmethod
from tempfile import NamedTemporaryFile

import astropy.units as u
import numpy as np
import pandas as pd
import requests
import zarr
from numcodecs.abc import Codec
from pathos.threading import ThreadPool
from tqdm.contrib.logging import tqdm_logging_redirect as tqdm

from mms_survey.utils.io import default_compressor, default_store


class BaseSynchronizer(ABC):
    r"""
    Base class for synchronizing data with LASP MMS Science Data Center
    """
    _api_link = "https://lasp.colorado.edu/mms/sdc/public/files/api/v1"

    def __init__(
        self,
        start_date: str = "2017-07-26",
        end_date: str = "2017-07-26",
        probe: str | list | None = None,
        instrument: str | list | None = None,
        data_rate: str | list | None = None,
        data_type: str | list | None = None,
        data_level: str | list | None = None,
        product: str | list | None = None,
        query_type: str = "science",
        update_local: bool = False,
        store: zarr._storage.store.Store = default_store,
        compressor: Codec = default_compressor,
    ):
        r"""
        Instantiation of synchronizer requires MMS API query parameters
        (see https://lasp.colorado.edu/mms/sdc/public/about/how-to/)

        Parameters
        ----------
        start_date: str
            Query start date (API equivalence: start_date)
        end_date: str
            Query end date (API equivalence: end_date)
        probe: str, list of str, or None (default)
            MMS probe name (API equivalence: sc_id)
        instrument: str, list of str, or None (default)
            MMS instrument name (API equivalence: instrument_id)
        data_rate: str, list of str, or None (default)
            Data rate mode (API equivalence: data_rate_mode)
        data_type: str, list of str, or None (default)
            Data descriptor (API equivalence: descriptor)
        data_level: str, list of str, or None (default)
            Data level (API equivalence: data_level)
        product: str, list of str, or None (default)
            Ancillary product (API equivalence: product)
        query_type: str
            Type of query ("science" or "ancillary")
        update_local: bool
            Toggle to force updating local data
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
        self.update_local = update_local
        self.store = store
        self.compressor = compressor
        self._compression_factor = 1.0

    def get_payload(self, cdf_file_name: str | None = None) -> dict:
        r"""
        Constructs HTTP payload from class properties, which are
        ignored if a particular file name is provided.

        Parameter
        ---------
        cdf_file_name: str or None (default)
            CDF file name (API equivalence: file)

        Returns
        -------
        payload: dict
            Dictionary for HTTP request containing payload information
        """
        if cdf_file_name is None:
            return {
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
            return {"file": cdf_file_name}

    def get_cdf_file_list(self) -> list[str]:
        r"""
        Get list of CDF file names relevant to payload information

        Returns
        -------
        cdf_file_list: list[str]
            List of file names
        """
        response = requests.get(
            url=f"{self._api_link}/file_info/{self.query_type}",
            params=self.get_payload(),
            timeout=10.0,
        )
        response.raise_for_status()

        cdf_file_info = response.json()["files"]
        cdf_file_list = list(map(lambda x: x["file_name"], cdf_file_info))
        cdf_file_size = (
            sum(
                list(
                    map(
                        lambda x: x["file_size"],
                        cdf_file_info,
                    )
                )
            )
            * u.B
        ).to(u.GB)
        logging.info(
            f"Downloading {len(cdf_file_list)} files with total size ="
            f" {cdf_file_size:.4f}, will be compressed to"
            f" {cdf_file_size * self._compression_factor:.4f}"
        )
        return cdf_file_list

    def download_file(self, cdf_file_name: str) -> str | None:
        r"""
        Download content of CDF file from LASP SDC into temporary file

        Parameter
        ---------
        cdf_file_name: str
            Name of CDF file

        Returns
        -------
        temp_file_name: str | None
            Name of temporary file (None if download failed)
        """
        with NamedTemporaryFile(delete=False, mode="wb") as temp_file:
            for _ in range(3):
                try:
                    response = requests.get(
                        url=f"{self._api_link}/download/{self.query_type}",
                        params=self.get_payload(cdf_file_name=cdf_file_name),
                        timeout=60.0,
                    )
                    cdf_file_size = int(response.headers.get("content-length"))
                    temp_file.write(response.content)
                    download_size = os.path.getsize(temp_file.name)
                    if cdf_file_size == download_size:
                        return temp_file.name
                except (
                    requests.ConnectionError,
                    requests.HTTPError,
                    requests.Timeout,
                ):
                    pass
            else:
                logging.warning(f"Giving up downloading {cdf_file_name}!")
                os.unlink(temp_file.name)
                return None

    def dataset_is_updated(self, file_metadata: dict) -> bool:
        r"""
        Called before downloading to determine if local dataset is updated
        """
        try:
            ds = zarr.open(self.store)
            group = file_metadata["group"]
            local_version = ds[group].attrs["Data_version"].replace("v", "")
            remote_version = file_metadata["version"].replace("v", "")
            return local_version == remote_version
        except KeyError:
            return False

    @abstractmethod
    def get_file_metadata(self, cdf_file_name: str) -> dict:
        raise NotImplementedError()

    @abstractmethod
    def process_file(self, temp_file_name: str, file_metadata: dict):
        r"""
        Called after downloading to process and save data to local storage
        """
        raise NotImplementedError()

    def sync(self, parallel: int = 1, dry_run: bool = False):
        def _helper(cdf_file_name: str):
            file_metadata = self.get_file_metadata(cdf_file_name)
            if not self.update_local and self.dataset_is_updated(
                file_metadata
            ):
                logging.info(f"Data from {cdf_file_name} is up-to-date")
                return

            temp_file_name = self.download_file(cdf_file_name)
            if temp_file_name is not None:
                self.process_file(temp_file_name, file_metadata)
                os.unlink(temp_file_name)
                logging.info(f"Processed {cdf_file_name}")

        cdf_file_list = self.get_cdf_file_list()
        if dry_run:
            return

        with ThreadPool(nodes=parallel) as pool:
            with tqdm(total=len(cdf_file_list), dynamic_ncols=True) as bar:
                for _ in pool.uimap(_helper, cdf_file_list):
                    bar.update()

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
