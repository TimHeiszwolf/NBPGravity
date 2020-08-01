import numpy as np
import pandas as pd
import time
import moviepy.editor as mpy
import matplotlib.pyplot as plt
import imageio#https://www.tutorialexample.com/python-create-gif-with-images-using-imageio-a-complete-guide-python-tutorial/
from scipy.optimize import fsolve

from body import Body
from space import Space
from controller import Controller
from helpers import *

bodies = get_test_Space_simple_solar()

controller = Controller([Space(bodies)], start_time=0.0, delta_time_settings=[[0, 7200, 3.0*10**6, 3.0*10**7, 3.0*10**8], [1.0, 60.0, 3600.0, 8.64*10**4, 2.592*10**6], 10], split_settings=[1, 10], center=True, report=True)

controller.simulate_until(10*365.25*24*3600)

limit = au_to_m(35)

make_gif(controller.get_combined_space().bodies, 10*24*3600, tick_per_frame=30, frames_per_second=30, window=[[-limit, limit], [-limit, limit]], name='output', axis=[0, 1], labels=False)

print('Done')