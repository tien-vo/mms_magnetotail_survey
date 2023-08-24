from mms_survey.load.ancillary import LoadAncillary

d = LoadAncillary(
    start_date="2017-01-01",
    end_date="2021-01-01",
    skip_ok_dataset=True,
)
d.download(dry_run=False)
