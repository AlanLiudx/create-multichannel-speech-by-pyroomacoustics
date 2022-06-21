from __future__ import division, print_function
import pyroomacoustics
from pyroomacoustics import Room
import numpy as np


import math
import warnings

import numpy as np
import scipy.spatial as spatial

def reset_simulator_state(self):
    self.simulator_state["ism_needed"]=True
    self.simulator_state["rt_needed"]=False
    self.simulator_state["ism_done"]=False
    self.simulator_state["rt_done"]=False

Room.reset_simulator_state=reset_simulator_state