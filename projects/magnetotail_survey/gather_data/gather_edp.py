from mms_survey.load.edp import LoadEDP

for year in range(2017, 2021):
    d = LoadEDP(
        start_date=f"{year}-04-01",
        end_date=f"{year}-11-01",
        probe=["mms1", "mms2", "mms3", "mms4"],
        data_rate="srvy",
        data_type="dce,scpot",
        data_level="l2",
    )
    d.download(dry_run=False)
