from mms_survey.mms_load.edp import LoadElectricDoubleProbes

d = LoadElectricDoubleProbes(
    start_date="2017-01-01",
    end_date="2021-01-01",
    probe=["mms1", "mms2", "mms3", "mms4"],
    data_rate="srvy",
    skip_processed_data=True,
)
d.download(parallel=8, dry_run=True)
