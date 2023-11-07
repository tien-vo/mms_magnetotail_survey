from mms_survey.sync import SyncEdpDce, SyncEdpScpot
from mms_survey.utils.io import data_dir

SyncEdpDce(update_local=True, store=data_dir / "test").sync()
SyncEdpScpot(update_local=True, store=data_dir / "test").sync()
