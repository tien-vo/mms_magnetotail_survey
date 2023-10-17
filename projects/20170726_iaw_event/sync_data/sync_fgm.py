import zarr

from mms_survey.sync import SyncFGM
from mms_survey.utils.io import data_dir

s = SyncFGM(
    start_date="2017-07-26/07:26:00",
    end_date="2017-07-26/07:30:00",
    probe=["mms1", "mms2", "mms3", "mms4"],
    data_rate="brst",
    update=True,
    store=zarr.DirectoryStore(data_dir / "20170726_iaw_event"),
)
s.download(dry_run=False)
