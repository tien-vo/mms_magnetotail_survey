from lib.utils import write_dataset, read_num_intervals
from pathos.pools import ProcessPool as Pool
from lib.load.feeps import load_feeps
from itertools import product
import lib
import os


def helper(probe, interval):
    for species in ["ion", "elc"]:
        data = load_feeps(probe, interval, drate="srvy", species=species)
        write_dataset(probe, interval, f"{species}-feeps", data)

    print(f"MMS{probe}: Saved {species} FEEPS data for interval {interval}", flush=True)


probes = [4,]
intervals = range(250, read_num_intervals())

for probe in probes:
    for species in ["ion", "elc"]:
        os.makedirs(f"{lib.h5_dir}/mms{probe}/{species}-feeps", exist_ok=True)

with Pool(8) as pool:
    for _ in pool.uimap(lambda args: helper(*args), product(probes, intervals)):
        pass
