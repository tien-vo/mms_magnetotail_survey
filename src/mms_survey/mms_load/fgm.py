__all__ = ["LoadFluxGateMagnetometer"]

import os

from cdflib.xarray import cdf_to_xarray

from mms_survey.utils.io import compressor, fix_epoch_metadata, store

from .base import BaseLoader


class LoadFluxGateMagnetometer(BaseLoader):
    def __init__(
        self,
        start_date: str = "2017-07-26",
        end_date: str = "2017-07-26",
        probe: str = "mms1",
        data_rate: str = "srvy",
        data_level: str = "l2",
        skip_ok_dataset: bool = False,
    ):
        super().__init__(
            instrument="fgm",
            start_date=start_date,
            end_date=end_date,
            probe=probe,
            data_rate=data_rate,
            data_type=None,
            data_level=data_level,
            product=None,
            query_type="science",
            skip_ok_dataset=skip_ok_dataset,
        )

    def get_metadata(self, file: str) -> dict:
        name = os.path.splitext(file)[0]
        probe, instrument, data_rate, data_level, tid, _ = name.split("_")
        return {
            "file": file,
            "probe": probe,
            "instrument": instrument,
            "data_rate": data_rate,
            "data_level": data_level,
            "group": f"/{data_rate}/{instrument}/{probe}/{tid}",
        }

    def process_file(self, file: str, metadata: dict):
        if metadata["instrument"] != self.instrument:
            return f"{file} is not in FGM dataset!"

        # Load file and fix metadata
        ds = cdf_to_xarray(file, to_datetime=True, fillval_to_nan=True)
        ds = fix_epoch_metadata(
            ds, vars=["Epoch", "Epoch_state"]
        ).reset_coords()
        ds = ds.rename_dims(dict(dim0="space"))
        ds = ds.assign_coords({"space": ["x", "y", "z", "mag"]})

        # Rename variables and remove unwanted variables
        pfx = f"{metadata['probe']}_{metadata['instrument']}"
        sfx = f"{metadata['data_rate']}_{metadata['data_level']}"
        ds = ds.rename(
            vars := {
                "Epoch": "time",
                "Epoch_state": "ephemeris_time",
                f"{pfx}_b_gse_{sfx}": "B_gse",
                f"{pfx}_b_gsm_{sfx}": "B_gsm",
                f"{pfx}_r_gse_{sfx}": "R_gse",
                f"{pfx}_r_gsm_{sfx}": "R_gsm",
                f"{pfx}_flag_{sfx}": "flag",
            }
        )
        ds = ds[list(vars.values())]
        ds.attrs["ok"] = True

        # Save
        encoding = {x: {"compressor": compressor} for x in ds}
        ds.to_zarr(
            mode="w",
            store=store,
            group=metadata["group"],
            encoding=encoding,
            consolidated=False,
        )


if __name__ == "__main__":
    d = LoadFluxGateMagnetometer()
    d.download()
