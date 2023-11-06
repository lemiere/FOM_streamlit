
import time
import random as rnd
import math as m
import numpy as np


input_gb_filename = "VGB.npy"
gb_config_file = np.load(input_gb_filename)


print(gb_config_file["Name"])
list_of_names = gb_config_file["Name"]
max_nb_of_sources = len(list_of_names)

print(list_of_names)
print(max_nb_of_sources)
