import zarr

from mms_survey.sync import SyncFPID, SyncFPIM, SyncFPIPM
from mms_survey.utils.io import data_dir

#s = SyncFPID(
#    data_type=["ion",],
#    update=True,
#    store=zarr.DirectoryStore(data_dir / "test"),
#)
#s.download()

#s = SyncFPIM(
#    data_type=["ion",],
#    update=True,
#    store=zarr.DirectoryStore(data_dir / "test"),
#)
#s.download()

s = SyncFPIPM(
    data_type=["ion",],
    update=True,
    store=zarr.DirectoryStore(data_dir / "test"),
)
s.download()
