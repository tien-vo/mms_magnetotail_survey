import os
from itertools import product

import h5py as h5
from pathos.pools import ProcessPool as Pool

import lib
from lib.load import fpi_moms
from lib.utils import read_num_intervals, write_data


def helper(probe, interval):
    print(
        f"MMS{probe}: Saving FPI moment data for interval {interval}",
        flush=True,
    )
    for species in ["ion", "elc"]:
        data = fpi_moms(probe, interval, drate="fast", species=species)
        if data is not None:
            write_data(probe, interval, f"{species}-fpi-moms", data)
        else:
            with open("log", "a") as f:
                f.write(
                    f"MMS{probe},intv{interval},{species} data not loaded.\n"
                )


probes = range(1, 5)
intervals = range(read_num_intervals())
for probe in probes:
    for species in ["ion", "elc"]:
        where = lib.h5_dir / f"mms{probe}" / f"{species}-fpi-moms"
        where.mkdir(parents=True, exist_ok=True)
        for intv in intervals:
            h5.File(where / f"interval_{intv}.h5", "w")

# with Pool(8) as pool:
#    for _ in pool.uimap(lambda args: helper(*args), product(probes, intervals)):
#        pass
