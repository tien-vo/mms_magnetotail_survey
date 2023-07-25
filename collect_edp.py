from lib.utils import write_dataset, read_num_intervals
from pathos.pools import ProcessPool as Pool
from lib.load.edp import load_edp
from itertools import product
import lib
import os


def helper(probe, interval):
    data = load_edp(probe, interval, drate="fast")
    write_dataset(probe, interval, instrument, data)
    print(f"MMS{probe}: Saved EDP data for interval {interval}", flush=True)


probes = [3, 4]
intervals = range(read_num_intervals())
instrument = "edp"
for probe in probes:
    os.makedirs(f"{lib.h5_dir}/mms{probe}/{instrument}", exist_ok=True)

with Pool(8) as pool:
    for _ in pool.uimap(lambda args: helper(*args), product(probes, intervals)):
        pass
