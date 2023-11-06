##this work
import utils


## Lisa tools
import lisaconstants



##
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import matplotlib.pyplot as plt



class LISA_analytical_noise:
    def __init__(self,name_,level_):
        self.noise_init(name_,level_)

    def __str__(self):
        if self.initialized:
            display = self.name+f" configuration is : {self.level}"
        else:
            display = "Not initialized..."
        return display

    def noise_init(self,name_,level_):
        self.name  = name_
        self.level = level_
        self.initialized = True
        
    def set_noise_level(self,level_):
        self.level = level_

    def get_noise_level(self):
        return self.level



    
    def instru_noise_psd(self,freq_, option_="X", tdi2_=False, arm_length_=2.5e9):
        """Return noise PSD from acc and oms noise, at given freq. range.
        :param array freq: frequency range
        :param str option: TDI name can be X, XY, A, E, T
        :param bool tdi2: TDI1.5 or 2nd generation
        :param float arm_length: arm length in meter
        :return array s_n: noise PSD
        """
        clight = lisaconstants.SPEED_OF_LIGHT
        #print("DEBUG : instru_noise_psd : ",tdi2)
        # LISA noise
        # Acceleration
        sa_a = (
            (3e-15) ** 2 * (1.0 + (0.4e-3 / freq_) ** 2) * (1.0 + (freq_ / 8e-3) ** 4)
        )  # in acceleration
        sa_d = sa_a * (2.0 * np.pi * freq_) ** (-4.0)  # in displacement
        s_pm = sa_d * (2.0 * np.pi * freq_ / clight) ** 2  # in rel freq unit

        # Optical Metrology System
        psd_oms_d = (15.0e-12) ** 2  # in displacement
        s_op = psd_oms_d * (2.0 * np.pi * freq_ / clight) ** 2  # in rel freq unit
    
        # Light travel time
        lisa_lt = arm_length_ / clight

        # Angular frequency
        omega = 2.0 * np.pi * freq_
        x = omega * lisa_lt

        if option_ == "X":
            s_n = 16.0 * np.sin(x) ** 2 * (2.0 * (1.0 + np.cos(x) ** 2) * s_pm + s_op)
        elif option_ == "XY":
            s_n = -4.0 * np.sin(2 * x) * np.sin(x) * (s_op + 4.0 * s_pm)
        elif option_ in ["A", "E"]:
            s_n = (
                8.0
                * np.sin(x) ** 2
                * (
                    2.0 * s_pm * (3.0 + 2.0 * np.cos(x) + np.cos(2 * x))
                    + s_op * (2.0 + np.cos(x))
                )
            )
        elif option_ == "T":
            s_n = (
                16.0 * s_op * (1.0 - np.cos(x)) * np.sin(x) ** 2
                + 128.0 * s_pm * np.sin(x) ** 2 * np.sin(0.5 * x) ** 4
            )
        else:
            print("PSD option should be in [X, XY, A, E, T] (%s)" % option)
            return None
        if tdi2_:
            factor_tdi2 = 4 * np.sin(2 * x) ** 2
            s_n *= factor_tdi2

        return s_n


    
    def confusion_noise_psd(self,freq_, duration_=4.5, option_="X", tdi2_=False, arm_length_=2.5e9):
        """Return noise PSD from GB confusion noise, at given freq. range.
        :param array freq: frequency range
        :param float nyears: number of years of observation
        :param str option: TDI name can be X, XY, A, E, T
        :param bool tdi2: TDI1.5 or 2nd generation
        :param float arm_length: arm length in meter
        :return array s_n: noise PSD
        """
        clight = lisaconstants.SPEED_OF_LIGHT
        lisaLT = arm_length_ / clight
        x = 2.0 * np.pi * lisaLT * freq_
        t = 4.0 * x**2 * np.sin(x) ** 2
        
        # confusion noise model for snr>7
        ampl = 1.28265531e-44
        alpha = 1.62966700e00
        fr2 = 4.81078093e-04
        af1 = -2.23499956e-01
        bf1 = -2.70408439e00
        afk = -3.60976122e-01
        bfk = -2.37822436e00
        
        tobs = duration_
        fr1 = 10.0 ** (af1 * np.log10(tobs) + bf1)
        fknee = 10.0 ** (afk * np.log10(tobs) + bfk)
        sg_sens = (
            ampl
            * np.exp(-((freq_ / fr1) ** alpha))
            * (freq_ ** (-7.0 / 3.0))
            * 0.5
            * (1.0 + np.tanh(-(freq_ - fknee) / fr2))
        )
        
        sgx = t * sg_sens
        if tdi2_ == True:
            factor_tdi2 = 4 * np.sin(2 * x) ** 2
            sgx *= factor_tdi2
        if option_ in ["A", "E"]:
            return 1.5 * sgx
        elif option_ == "XY":
            return -0.5 * sgx
        else:
            return sgx

    

    
    def reset(self):
        self.name  = None
        self.level = None
        self.initialized = False



if __name__ == "__main__":
    
    test0 = LISA_analytical_noise("dummy", 42)
    print(test0)
    
    
    test0.set_noise_level(666)
    print(test0)
    
    test0.reset()
    print(test0)
    
    
    test0.noise_init("red book",12)
    print(test0)
    
    
    freq = np.logspace(-5, 0, 9990)
    duration = 4.5  # years
    tdi2 = True
    
    ax.loglog(freq, np.sqrt(freq) * np.sqrt(sh(freq)), label="instrumental noise")
    ax.loglog(freq, np.sqrt(freq) * np.sqrt(20 / 3) * np.sqrt(sh_wd(freq)), color="k", ls="--",label="+confusion noise")


    # graph to publish
    fig, ax = plt.subplots(1, figsize=(12, 8))



    ax.loglog(freq, np.sqrt(freq) * np.sqrt(sh(freq)), label="instrumental noise")
    ax.loglog(freq, np.sqrt(freq) * np.sqrt(20 / 3) * np.sqrt(sh_wd(freq)), color="k", ls="--",label="+confusion noise")
    
    ax.set_ylabel("ASD (to check)")
    ax.set_xlabel("Frequnecy (Hz)")
    
    plt.legend()
    plt.grid()
    plt.show()

