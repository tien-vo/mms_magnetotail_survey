from mms_survey.sync import SyncMec
from mms_survey.utils.io import data_dir

SyncMec(update_local=True, store=data_dir / "test").sync()
