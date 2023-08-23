from mms_survey.load.fpi import DownloadFPI

#for year in range(2017, 2021):
for year in range(2017, 2018):
    d = DownloadFPI(
        start_date=f"{year}-04-01",
        end_date=f"{year}-11-01",
        probe="mms1",
        data_rate="srvy",
        data_type="dis-dist",
        data_level="l2",
    )
    d.download(dry_run=True)
