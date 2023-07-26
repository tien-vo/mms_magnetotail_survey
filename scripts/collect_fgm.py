import os
import lib
import h5py as h5
from pathos.pools import ProcessPool as Pool
from itertools import product
from lib.utils import write_data, read_num_intervals
from lib.load import fgm


def helper(probe, interval):
    data = fgm(probe, interval, drate="srvy")
    write_data(probe, interval, instrument, data)
    print(f"MMS{probe}: Saved FGM data for interval {interval}", flush=True)


probes = range(1, 5)
intervals = range(read_num_intervals())
instrument = "fgm"
for probe in probes:
    where = lib.h5_dir / f"mms{probe}" / instrument
    where.mkdir(parents=True, exist_ok=True)
    for intv in intervals:
        h5.File(where / f"interval_{intv}.h5", "w")

#with Pool(8) as pool:
#    for _ in pool.uimap(lambda args: helper(*args), product(probes, intervals)):
#        pass
