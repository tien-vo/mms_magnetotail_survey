from mms_survey.sync import SyncFgm
from mms_survey.utils.io import data_dir

SyncFgm(update_local=True, store=data_dir / "test").sync()
