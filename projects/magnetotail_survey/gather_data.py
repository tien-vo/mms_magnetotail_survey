from mms_survey.mms_load.ancillary import LoadTetrahedronQualityFactor
from mms_survey.mms_load.mec import LoadMagneticEphemerisCoordinates
from mms_survey.mms_load.fgm import LoadFluxGateMagnetometer
from mms_survey.mms_load.edp import LoadElectricDoubleProbes

#start_date = "2017-01-01"
#end_date = "2021-01-01"
start_date = "2017-07-26"
end_date = "2017-07-27"
data_rate = "srvy"
dry_run = False
parallel = 8
skip_ok_dataset = True

d = LoadTetrahedronQualityFactor(
    start_date=start_date,
    end_date=end_date,
    skip_ok_dataset=skip_ok_dataset,
)
d.download(parallel=parallel, dry_run=dry_run)

d = LoadMagneticEphemerisCoordinates(
    start_date=start_date,
    end_date=end_date,
    probe=["mms1", "mms2", "mms3", "mms4"],
    data_rate=data_rate,
    skip_ok_dataset=skip_ok_dataset,
)
d.download(parallel=parallel, dry_run=dry_run)

d = LoadFluxGateMagnetometer(
    start_date=start_date,
    end_date=end_date,
    probe=["mms1", "mms2", "mms3", "mms4"],
    data_rate=data_rate,
    skip_ok_dataset=skip_ok_dataset,
)
d.download(parallel=parallel, dry_run=dry_run)

d = LoadElectricDoubleProbes(
    start_date=start_date,
    end_date=end_date,
    probe=["mms1", "mms2", "mms3", "mms4"],
    data_rate=data_rate,
    data_type="dce,scpot",
    skip_ok_dataset=skip_ok_dataset,
)
d.download(parallel=parallel, dry_run=dry_run)
