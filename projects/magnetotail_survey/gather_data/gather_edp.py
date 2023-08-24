from mms_survey.load.edp import LoadEDP

d = LoadEDP(
    start_date="2017-01-01",
    end_date="2021-01-01",
    probe=["mms1", "mms2", "mms3", "mms4"],
    data_rate="srvy",
    data_type="dce,scpot",
    data_level="l2",
    skip_ok_dataset=True,
)
d.download(dry_run=False)
