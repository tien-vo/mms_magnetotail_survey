from mms_survey.load.feeps import LoadFEEPS

for year in range(2017, 2021):
    d = LoadFEEPS(
        start_date=f"{year}-04-01",
        end_date=f"{year}-11-01",
        probe="mms1",
        data_rate="srvy",
        data_type="ion",
        data_level="l2",
    )
    d.download(dry_run=True)

    d = LoadFEEPS(
        start_date=f"{year}-04-01",
        end_date=f"{year}-11-01",
        probe="mms1",
        data_rate="srvy",
        data_type="electron",
        data_level="l2",
    )
    d.download(dry_run=True)
