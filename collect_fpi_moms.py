from lib.utils import write_dataset, read_num_intervals
from pathos.pools import ProcessPool as Pool
from lib.load.fpi_moms import load_fpi_moms
from itertools import product
import lib
import os


def helper(probe, interval):
    print(f"MMS{probe}: Saving FPI moment data for interval {interval}", flush=True)
    for species in ["ion", "elc"]:
        data = load_fpi_moms(probe, interval, drate="fast", species=species)
        if data is not None:
            write_dataset(probe, interval, f"{species}-fpi-moms", data)
        else:
            with open("log", "a") as f:
                f.write(f"MMS{probe}, interval {interval}, {species} data not available\n")


#probes = [1,]
probes = [2, 3]
intervals = range(read_num_intervals())

for probe in probes:
    for species in ["ion", "elc"]:
        os.makedirs(f"{lib.h5_dir}/mms{probe}/{species}-fpi-moms", exist_ok=True)

with Pool(8) as pool:
    for _ in pool.uimap(lambda args: helper(*args), product(probes, intervals)):
        pass
