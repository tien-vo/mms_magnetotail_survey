import zarr

from mms_survey.sync import SyncEDPDC, SyncEDPP
from mms_survey.utils.io import data_dir

#s = SyncEDPDC(
#    update=True,
#    store=zarr.DirectoryStore(data_dir / "test"),
#)
#s.download()

s = SyncEDPP(
    update=True,
    store=zarr.DirectoryStore(data_dir / "test"),
)
s.download()
