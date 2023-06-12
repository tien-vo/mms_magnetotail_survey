import astropy.units as u
import h5py as h5
import lib

__all__ = ["write_dataset"]


def write_dataset(probe, interval, where, data):
    fname = f"{lib.data_dir}/h5/mms{probe}/interval_{interval}.h5"
    with h5.File(fname, "a") as h5f:
        for key, value in data.items():
            if (write_where := f"{where}/{key}") in h5f:
                del h5f[write_where]

            if isinstance(value, u.Quantity):
                h5d = h5f.create_dataset(write_where, data=value.value)
                h5d.attrs["unit"] = str(value.unit)
            else:
                h5f.create_dataset(write_where, data=value)
