import zarr

from mms_survey.sync import SyncMEC
from mms_survey.utils.io import data_dir

s = SyncMEC(
    start_date="2017-07-26",
    end_date="2017-07-26",
    probe=["mms1", "mms2", "mms3", "mms4"],
    data_rate="srvy",
    update=True,
    store=zarr.DirectoryStore(data_dir / "20170726_iaw_event"),
)
s.download(dry_run=False)
