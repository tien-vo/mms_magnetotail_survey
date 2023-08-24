from mms_survey.load.ancillary import LoadAncillary
from mms_survey.load.fgm import LoadFGM
from mms_survey.load.edp import LoadEDP

start_date = "2017-01-01"
end_date = "2021-01-01"
data_rate = "srvy"
dry_run = False

d = LoadAncillary(
    start_date=start_date,
    end_date=end_date,
    skip_ok_dataset=True,
)
d.download(dry_run=dry_run)

d = LoadFGM(
    start_date=start_date,
    end_date=end_date,
    probe=["mms1", "mms2", "mms3", "mms4"],
    data_rate=data_rate,
    skip_ok_dataset=True,
)
d.download(dry_run=dry_run)

d = LoadEDP(
    start_date=start_date,
    end_date=end_date,
    probe=["mms1", "mms2", "mms3", "mms4"],
    data_rate=data_rate,
    data_type="dce,scpot",
    skip_ok_dataset=True,
)
d.download(dry_run=dry_run)
