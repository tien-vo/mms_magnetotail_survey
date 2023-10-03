from mms_survey.mms_load.fgm import LoadFluxGateMagnetometer

d = LoadFluxGateMagnetometer(
    start_date="2017-01-01",
    end_date="2021-01-01",
    probe=["mms1", "mms2", "mms3", "mms4"],
    data_rate="srvy",
    skip_up_to_date_dataset=True,
)
d.download(parallel=8, dry_run=False)
