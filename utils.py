
## Lisa tools
import lisaconstants

##
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline



def fast_response(freq, arm_length=2.5e9, tdi2=False):
    """Sky averaged response of the LISA constellation for TDI X.

    :param array freq: frequency range
    :param float arm_length: arm length in meter
    :param bool tdi2: TDI1.5 or 2nd generation
    :return array R: LISA TDI X response
    """
    lisaLT = arm_length / lisaconstants.SPEED_OF_LIGHT
    x = 2.0 * np.pi * lisaLT * freq
    r = np.absolute(9 / 20 / (1 + (3 * x / 4) ** 2) * ((16 * x**2 * np.sin(x) ** 2)))
    if tdi2:
        r *= 4 * np.sin(2 * x) ** 2
    return r / 1.5 / 2



def psd2sh(freq, SX, arm_length=2.5e9, tdi2=False, sky_averaging=False):
    """Return sensitivity curve from noise psd.

    :param array freq: frequency range
    :param array SX: noise PSD
    :param bool tdi2: TDI1.5 or 2nd generation
    :param float arm_length: arm length in meter
    :param bool sky_averaging: apply sky averaging factor
    :return array Sh: sensitivity
    """
    lisa_arm_t = arm_length / lisaconstants.SPEED_OF_LIGHT
    if tdi2:
        fctr = (
            8.0
            * np.sin(2.0 * np.pi * freq * lisa_arm_t)
            * np.sin(4.0 * np.pi * freq * lisa_arm_t)
        ) ** 2
    else:
        fctr = (4.0 * np.sin(2.0 * np.pi * freq * lisa_arm_t)) ** 2

    f_star = 2.0 * np.pi * lisa_arm_t * freq
    R = 1.0 / (1.0 + 0.6 * (f_star) ** 2)
    fctr = fctr * (2.0 * np.pi * freq * lisa_arm_t) ** 2 * R
    if sky_averaging:
        fctr *= 3 / 20.0
    Sh = spline(freq, SX / fctr)
    return Sh



def compute_snr(XYZ_, SXX_, SXY_):
    """SNR from TDI XYZ

    :param 3xN array XYZ: signal TDI X,Y,Z
    :param 1xN array SXX: noise PSD auto term
    :param 1xN array SXY: noise PSD cross term
    :return float snr: total snr from X,Y,Z combination.
    """
    Efact = SXX * SXX + SXX * SXY - 2 * SXY * SXY
    Efact[Efact == 0] = np.inf
    Efact = 1 / Efact
    EXX = (SXX + SXY) * Efact
    EXY = -SXY * Efact

    snr = 0
    for k in range(3):
        snr += np.sum(np.real(XYZ[k] * np.conj(XYZ[k]) * EXX))
    for k1, k2 in [(0, 1), (0, 2), (1, 2)]:
        snr += np.sum(np.real(XYZ[k1] * np.conj(XYZ[k2]) * EXY))
        snr += np.sum(np.real(XYZ[k2] * np.conj(XYZ[k1]) * EXY))
    snr *= 4.0 * df
    return np.sqrt(snr)
