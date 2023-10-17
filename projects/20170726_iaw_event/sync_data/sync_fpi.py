import zarr

from mms_survey.sync import SyncFPID, SyncFPIM, SyncFPIPM
from mms_survey.utils.io import data_dir

kw = dict(
    start_date="2017-07-26/07:26:00",
    end_date="2017-07-26/07:30:00",
    probe=["mms2"],
    data_type=["ion", "elc"],
    data_rate="brst",
    update=True,
    store=zarr.DirectoryStore(data_dir / "20170726_iaw_event"),
)

#s = SyncFPID(**kw)
#s.download(parallel=8, dry_run=False)

#s = SyncFPIM(**kw)
#s.download(parallel=8, dry_run=False)

s = SyncFPIPM(**kw)
s.download(parallel=8, dry_run=False)
