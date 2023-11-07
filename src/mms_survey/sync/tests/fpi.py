from mms_survey.sync import (
    SyncFpiDistribution,
    SyncFpiMoments,
    SyncFpiPartialMoments,
)
from mms_survey.utils.io import data_dir

kw = dict(data_type=["ion", "elc"], update_local=True, store=data_dir / "test")
SyncFpiDistribution(**kw).sync()
SyncFpiMoments(**kw).sync()
SyncFpiPartialMoments(**kw).sync()
