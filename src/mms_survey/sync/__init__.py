import warnings

from .edp._dce import SyncElectricDoubleProbesDce
from .edp._scpot import SyncElectricDoubleProbesScpot
from .fgm._fgm import SyncFluxGateMagnetometer
from .fpi._distribution import SyncFastPlasmaInvestigationDistribution
from .fpi._moments import SyncFastPlasmaInvestigationMoments
from .fpi._partial_moments import SyncFastPlasmaInvestigationPartialMoments
from .mec._mec import SyncMagneticEphemerisCoordinates

# Ignore cdflib.xarray.cdf_to_xarray timestamp precision warning
warnings.filterwarnings(
    "ignore",
    message=(
        "Converting non-nanosecond precision datetime values "
        "to nanosecond precision."
    ),
)

# Aliases
SyncMec = SyncMagneticEphemerisCoordinates
SyncFgm = SyncFluxGateMagnetometer
SyncEdpDce = SyncElectricDoubleProbesDce
SyncEdpScpot = SyncElectricDoubleProbesScpot
SyncFpiDistribution = SyncFastPlasmaInvestigationDistribution
SyncFpiMoments = SyncFastPlasmaInvestigationMoments
SyncFpiPartialMoments = SyncFastPlasmaInvestigationPartialMoments
