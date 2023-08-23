from mms_survey.load.ancillary import DownloadANCIL

for year in range(2017, 2021):
    d = DownloadANCIL(start_date=f"{year}-04-01", end_date=f"{year}-11-01")
    d.download(dry_run=False)
