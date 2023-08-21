import os
import requests
import numpy as np
import pandas as pd

from bunch import Bunch
from tqdm import tqdm

# Consult LASP SDC's MMS API at
#   https://lasp.colorado.edu/mms/sdc/public/about/how-to/
#   for query instructions
_default_config = Bunch(
    server="https://lasp.colorado.edu/mms/sdc/public/files/api/v1",
    start_date="2017-07-26T07:10",
    end_date="2017-07-26T07:40",
    sc_ids=[
        "mms1",
    ],
    instrument_ids=[
        "fgm",
    ],
    data_rate_modes=[
        "srvy",
    ],
    data_levels=[
        "l2",
    ],
)


class Downloader(object):
    def __init__(self, **kwargs):
        self.config = Bunch({**_default_config, **kwargs})

    @property
    def start_date(self):
        return self._start_date

    @start_date.setter
    def start_date(self, date: np.datetime64):
        assert isinstance(date, np.datetime64), "Date must be np.datetime64 datatype"

    @property
    def trange(self):
        return self._trange

    @trange.setter
    def trange(
        self,
        trange=np.array(
            ["2017-07-26T07:10", "2017-07-26T07:40"], dtype="datetime64[ns]"
        ),
    ):
        pass
        # return self._start_date

    # def _get_file_names(self):
    #    return = (
    #        f"{self.config.server}/file_names/science?"
    #        + f"start_date={self.config.start_date}"
    #        + f"&end_date={self.config.end_date}"
    #        + f"&sc_ids={','.join(self.config.sc_ids)}"
    #        + f"&instrument_ids={','.join(self.config.instrument_ids)}"
    #        + f"&data_rate_modes={','.join(self.config.data_rate_modes)}"
    #        + f"&data_levels={','.join(self.config.data_levels)}"
    #    )

    # def _get_download_string(self):
    #    return (
    #        f"{self.config.server}/download/science?"
    #        + f"start_date={self.config.start_date}"
    #        + f"&end_date={self.config.end_date}"
    #        + f"&sc_ids={','.join(self.config.sc_ids)}"
    #        + f"&instrument_ids={','.join(self.config.instrument_ids)}"
    #        + f"&data_rate_modes={','.join(self.config.data_rate_modes)}"
    #        + f"&data_levels={','.join(self.config.data_levels)}"
    #    )
