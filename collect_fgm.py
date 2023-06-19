from lib.utils import write_dataset, get_num_intervals
from pathos.pools import ProcessPool as Pool
from lib.load.fgm import load_fgm
from itertools import product
import lib


def helper(probe, interval):
    data = load_fgm(probe, interval, drate="srvy")
    write_dataset(probe, interval, "/fgm", data)
    print(f"MMS{probe}: Saved FGM data for interval {interval}", flush=True)


with Pool(8) as pool:
    for _ in pool.uimap(lambda args: helper(*args), product(range(1, 5), range(get_num_intervals()))):
        pass
