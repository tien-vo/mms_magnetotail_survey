from mms_survey.mms_load.ancillary import LoadTetrahedronQualityFactor

d = LoadTetrahedronQualityFactor(
    start_date="2017-01-01",
    end_date="2021-01-01",
    skip_processed_data=True,
)
d.download(parallel=8, dry_run=False)
