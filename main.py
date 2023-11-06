#!/usr/bin/python3

#homemade 
import LISA_GB_configuration as myGB
import LISA_noise_configuration as NOISE
import utils
#lisa
from fastgb.fastgb import FastGB
import lisaorbits
import lisaconstants
#display module
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
#common
import math as m
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline



#### main init
apptitle = 'FOM display facility'
im = Image.open("images/lisa.ico")
st.set_page_config(page_title=apptitle,
                   page_icon=im,
                   layout="wide")


### data init
# verification GB reader
input_gb_filename = "data/VGB.npy"

gb_config_file = np.load(input_gb_filename)
nb_of_sources = len(gb_config_file)
GB_out = np.rec.fromarrays(
    [np.zeros((nb_of_sources, 1)),
     np.zeros((nb_of_sources, 1)),
     np.zeros((nb_of_sources, 1))],
    names=["freq", "sh", "snr"]
)


list_of_names = gb_config_file["Name"]

list_of_names_opt = list_of_names
list_of_names_opt = np.append("select all",list_of_names_opt)




list_of_sources = []
list_of_amplitude = []

    

st.sidebar.header('Sources selection')

list_of_GB = st.sidebar.multiselect('Galactic Binaries', list_of_names_opt,placeholder='Select GB')

if  "select all" in list_of_GB:
    list_of_GB = list_of_names


mission_duration = st.sidebar.slider('duration in year ?', min_value=1., max_value=10., step=0.5)

#nb_of_GB = st.sidebar.slider('number of GB ?', min_value=0, max_value=max_nb_of_sources, step=1)

tdi_type = st.sidebar.radio("select your TDI version",
   ["tdi1.5","tdi2.0"])#horizontal=True,)

if tdi_type == "tdi1.5":
    tdi2 = False
elif tdi_type == "tdi2.0":
    tdi2 = True
else:
    tdi2 = False


#display mode
display_mode = st.sidebar.radio(
    "select your display mode",
    ["x unified","x","closest"])
    
####### prepare the data
#noise
test0 = NOISE.LISA_analytical_noise("dummy", 42)

freq        = np.logspace(-5, 0, 9990)
duration    = mission_duration  # years
tobs        = duration * lisaconstants.SIDEREALYEAR_J2000DAY * 24 * 60 * 60
lisa_orbits = lisaorbits.EqualArmlengthOrbits(dt=8640, size=(tobs + 10000) // 8640)
# to control the +10000

# noise psd
SXX_noise_instru_only    = test0.instru_noise_psd(freq, tdi2_=tdi2, option_="X")
SXX_confusion_noise_only =  test0.confusion_noise_psd(freq, duration_=duration, tdi2_=tdi2, option_="X")
SXX_noise = SXX_noise_instru_only + SXX_confusion_noise_only

SXY_noise_instru_only    = test0.instru_noise_psd(freq, tdi2_=tdi2, option_="XY")
SXY_confusion_noise_only = test0.confusion_noise_psd(freq, duration_=duration, tdi2_=tdi2, option_="XY")
SXY_noise =  SXY_noise_instru_only + SXX_confusion_noise_only


SXX = spline(freq, SXX_noise)
SXY = spline(freq, SXY_noise)

# response
R_ = utils.fast_response(freq, tdi2=tdi2)
R = spline(freq, R_)
# NOISE SENSITIVITY
sh = spline(freq, SXX_noise_instru_only / R_)
sh_wd = utils.psd2sh(freq, SXX_noise, sky_averaging=False, tdi2=tdi2)

#signal

GB = FastGB(delta_t=15, T=tobs, orbits=lisa_orbits, N=1024)
df = 1 / tobs



for j, s in enumerate(gb_config_file):

    pGW = dict(zip(gb_config_file.dtype.names, s))


    if pGW["Name"] in list_of_GB:
        
        params = np.array( [pGW["Frequency"],
                            pGW["FrequencyDerivative"],
                            pGW["Amplitude"],
                            pGW["EclipticLatitude"],
                            pGW["EclipticLongitude"],
                            pGW["Polarization"],
                            pGW["Inclination"],
                            pGW["InitialPhase"] ])
        
        source_tmp = myGB.LISA_GB_source(pGW["Name"],params)
        list_of_sources.append(source_tmp)
        list_of_amplitude.append( source_tmp.get_source_parameters()[0][2]/(1e-23))

        X, Y, Z, kmin = GB.get_fd_tdixyz(source_tmp.get_source_parameters(), tdi2=True) 
        X_f = df * np.arange(kmin, kmin + len(X.flatten()))
        
        h0 = np.sqrt(4 * df * float(np.sum(np.abs(X) ** 2 / R(X_f))))
        h0 *= np.sqrt(2)
        GB_out["sh"][j] = h0**2
        GB_out["freq"][j] = pGW["Frequency"]

list_of_source = []
tab_sensitivity, tab_waterfall = st.tabs(["Sensitivity curve", "Waterfall plot"])

# st.error('Error message')
# st.warning('Warning message')
# st.info('Info message')
# st.success('Success message')



####### display the sensitivity curve
vf= []
vy = []

for vgb in GB_out:
    vf.append(float(vgb["freq"]))
    vy.append(float(np.sqrt(vgb["freq"] * vgb["sh"])))

## end of fake data

# col1, col2 = st.columns([3,1])
# col1.write('Figure')
# col2.write('Buttons')


fig = go.Figure()

tmp = list_of_names.tolist()#list_of_GB


fig.add_trace(go.Scatter(
    x=vf,
    y=vy,
    hovertext = tmp,
    #visible='legendonly',
    mode='markers',
    marker={'color':'red'},
    marker_symbol="hexagon",
    name="GBs",
    hovertemplate = "<b>%{hovertext}</b><br>f= %{x:.4f} Hz<br>h=%{y}",
))

fig.add_trace(go.Scatter(
    x=freq,
    y=np.sqrt(freq) * np.sqrt(sh(freq)),
    name="Instrumental Noise"
))


fig.add_trace(go.Scatter(
    x=freq,
    y=np.sqrt(freq) * np.sqrt(20 / 3) * np.sqrt(sh_wd(freq)),
    name="LISA Noise (Instru+Confusion)"
))


fig.update_xaxes(title_text="Frequency (Hz)", type="log",showgrid=True,showexponent = 'all',exponentformat='e' )
fig.update_yaxes(title_text="Characteristic Strain (TODO)", type="log",showgrid=True)
fig.update_layout(xaxis=dict(range=[-5,0]))
fig.update_layout(yaxis=dict(range=[-22,-15]))
fig.update_layout(template="ggplot2")

fig.update_layout(hovermode=display_mode)

fig.update_layout(legend=dict(orientation="h",
                              yanchor="bottom",
                              y=1.02,
                              xanchor="right",
                              x=1
                              )
                  )

fig.update_layout(height=600, width=1000)# , grid= {'rows': 7, 'columns': 6})
tab_sensitivity.plotly_chart(fig, theme=None, use_container_width=True)



fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=freq,
    y=SXX_noise_instru_only,
    name="instru"
))

fig2.add_trace(go.Scatter(
    x=freq,
    y=SXX_confusion_noise_only,
    #visible='legendonly',
    name="confusion"
))

fig2.update_xaxes(title_text="Frequency (Hz)",
                  type="log",showgrid=True,showexponent = 'all',exponentformat='e' )
fig2.update_yaxes(title_text="Characteristic Strain (TODO)", type="log",showgrid=True)
tab_sensitivity.plotly_chart(fig2, theme=None, use_container_width=True)
