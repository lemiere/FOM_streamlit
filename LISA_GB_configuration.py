
import math as m
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import matplotlib.pyplot as plt


import utils
from fastgb.fastgb import FastGB
#import fastGB as FastGB

import lisaorbits
import lisaconstants


class LISA_GB_source:
    def __init__(self,name_,params_):
        self.source_init(name_,params_)

    def __str__(self):
        if self.initialized:
            display = "source "+self.name+" OK"
        else:
            display = "Not initialized..."
        return display

    def source_init(self,name_,params_):
        self.name  = name_
        self.params = params_
        self.set_source_position(params_[3],params_[4])
        self.initialized = True
        
    def set_source_position(self,beta_,lambda_):
        self.source_beta   = beta_
        self.source_lambda = lambda_

    def get_name(self):
        if self.initialized == True : 
            return self.name
        else:
            return None
        
    def get_source_parameters(self):
        if self.initialized == True:
            return self.params.reshape(1,-1)          
        else:
            return None

        
    def get_source_position(self):
        if self.initialized == True:
            position = [self.source_beta, self.source_lambda]
        else:
            position = None
        return position

    # def compute(self,tdi2_):
    #     X, Y, Z, kmin = GB.get_fd_tdixyz(params.reshape(1, -1), tdi2=True)
    #     X_f = df * np.arange(kmin, kmin + len(X.flatten()))



    
    def reset(self):
        self.name          = None
        self.params        = None
        self.source_beta   = None
        self.source_lambda = None
        self.initialized = False

    def display(self):
        if self.initialized == True:
            print(self.name ," Source parametrization : ")
            print("   |-> params : ",self.params)
            print("   |-> position : ",self.source_beta," ",self.source_lambda)
        else:
            print("Not initialized ...")
            

        
if __name__ == "__main__":

    # parametres = [0,0,0,m.pi/4,m.pi/2]
    # test0 = LISA_GB_source("source 1", parametres)
    # print(test0)
    # test0.reset()
    # print(test0)
    
    
    duration = 1
    tobs = duration * lisaconstants.SIDEREALYEAR_J2000DAY * 24 * 60 * 60
    lisa_orbits = lisaorbits.EqualArmlengthOrbits(dt=8640, size=(tobs + 10000) // 8640) # to control the +10000

    GB = FastGB(delta_t=15, T=tobs, orbits=lisa_orbits, N=1024)
    df = 1 / tobs
    
    
    # verification GB reader
    input_gb_filename = "data/VGB.npy"

    
    gb_config_file = np.load(input_gb_filename)
    nb_of_sources = len(gb_config_file)
    GB_out = np.rec.fromarrays(
        [np.zeros((nb_of_sources, 1)), np.zeros((nb_of_sources, 1)), np.zeros((nb_of_sources, 1))],
        names=["freq", "sh", "snr"],
    )
    list_of_sources = []
    list_of_amplitude = []
    for j, s in enumerate(gb_config_file):
     
        pGW = dict(zip(gb_config_file.dtype.names, s))
        params = np.array( [pGW["Frequency"],
                            pGW["FrequencyDerivative"],
                            pGW["Amplitude"],
                            pGW["EclipticLatitude"],
                            pGW["EclipticLongitude"],
                            pGW["Polarization"],
                            pGW["Inclination"],
                            pGW["InitialPhase"] ])

        ### probably due to the FastGB interface ###
        source_tmp = LISA_GB_source(pGW["Name"],params)
        list_of_sources.append(source_tmp)

        list_of_amplitude.append( source_tmp.get_source_parameters()[0][2]/(1e-23))
        #source_tmp.display()
        X, Y, Z, kmin = GB.get_fd_tdixyz(source_tmp.get_source_parameters(), tdi2=True) 
        X_f = df * np.arange(kmin, kmin + len(X.flatten()))

        freq = np.logspace(-5, 0, 9990)
        R_ = utils.fast_response(freq, tdi2=True)
        R = spline(freq, R_)

        
        h0 = np.sqrt(4 * df * float(np.sum(np.abs(X) ** 2 / R(X_f))))
        h0 *= np.sqrt(2)
        GB_out["sh"][j] = h0**2
        GB_out["freq"][j] = pGW["Frequency"]
        #GB_out["snr"][j] = utils.compute_snr(np.vstack([X, Y, Z]), SXX(X_f), SXY(X_f))

    fig, ax = plt.subplots(1, figsize=(12, 8))
        
    vf= []
    vy = []

    
    for vgb in GB_out:
        vf.append(vgb["freq"])
        vy.append(np.sqrt(vgb["freq"] * vgb["sh"]))

    ax.scatter(vf,vy,s=list_of_amplitude,label="sources")
    ax.legend(loc="upper center", fontsize=8)
    plt.grid()
    plt.legend()
    plt.show()
    
