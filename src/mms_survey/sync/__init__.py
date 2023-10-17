import warnings

from .fgm._fgm import SyncFluxGateMagnetometer
from .edp._dc import SyncElectricDoubleProbesDC
from .edp._potential import SyncElectricDoubleProbesPotential
from .fpi._distribution import SyncFastPlasmaInvestigationDistribution
from .fpi._moments import SyncFastPlasmaInvestigationMoments
from .fpi._partial_moments import SyncFastPlasmaInvestigationPartialMoments

# Ignore cdflib.xarray.cdf_to_xarray timestamp precision warning
warnings.filterwarnings(
    "ignore",
    message=(
        "Converting non-nanosecond precision datetime values "
        "to nanosecond precision."
    ),
)

# Aliases
SyncFGM = SyncFluxGateMagnetometer
SyncEDPDC = SyncElectricDoubleProbesDC
SyncEDPP = SyncElectricDoubleProbesPotential
SyncFPID = SyncFastPlasmaInvestigationDistribution
SyncFPIM = SyncFastPlasmaInvestigationMoments
SyncFPIPM = SyncFastPlasmaInvestigationPartialMoments
