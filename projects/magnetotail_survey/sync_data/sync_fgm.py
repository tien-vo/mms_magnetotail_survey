import zarr

from mms_survey.sync.fgm import SyncFluxGateMagnetometer

d = SyncFluxGateMagnetometer(
    start_date="2017-01-01",
    end_date="2021-01-01",
    probe=["mms1", "mms2", "mms3", "mms4"],
    data_rate="srvy",
    update=False,
    store=zarr.DirectoryStore("/Volumes/MMS_DATA/mms_survey/raw"),
)
d.download(parallel=8, dry_run=False)