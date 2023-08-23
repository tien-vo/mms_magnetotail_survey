from mms_survey.load.ancillary import DownloadANCIL

d = DownloadANCIL(start_date="2017-01-01", end_date="2021-01-01")
d.download(dry_run=False)
