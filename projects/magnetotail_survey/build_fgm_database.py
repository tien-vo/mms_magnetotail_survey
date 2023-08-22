from mms_survey.load.fgm import DownloadFGM

d = DownloadFGM(
    #start_date="2017-04-01",
    #end_date="2017-11-01",
    start_date="2017-07-26",
    end_date="2017-07-28",
    probe=["mms1", "mms2", "mms3", "mms4"],
    data_rate="srvy",
    data_level="l2",
)

d.download(dry_run=False)
