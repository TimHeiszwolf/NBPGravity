import numpy as np
import pandas as pd
import time
import datetime

from body import Body
from space import Space
from helpers import *

class Controller:
    def __init__(self, spaces=[], start_time=0.0, delta_time_settings=[[0, 7200, 3.0*10**6, 3.0*10**7, 3.0*10**8], [1.0, 60.0, 3600.0, 8.64*10**4, 2.592*10**6], 10], split_settings=[1, 10], center=False, report=True):
        
        self.spaces = spaces
        print(spaces)
        self.time = start_time
        self.tick = 0
        self.delta_time_settings = delta_time_settings
        self.report = report
        self.set_delta_time()
        self.center = center
        self.split_settings = split_settings
        
        """
        data = {'time': self.time}
        names = []
        
        
        for body in self.get_combined_space.get_real_bodies():
            if names.count(body.name)==1:
                raise ValueError('Two bodies have the same name: ' + body.name)
            names.append(body.name)
            
            data[body.name] = body.position
        
        self.data = DataFrame(data)"""
    
    
    def get_combined_space(self):
        total_bodies = []
        
        for space in self.spaces:
            total_bodies = total_bodies + space.get_real_bodies()# Does this work? https://stackoverflow.com/questions/1720421/how-do-i-concatenate-two-lists-in-python
        
        return Space(total_bodies, self.spaces[0].delta_time, self.spaces[0].time)
    
    def set_delta_time(self):
        minimal_time_distance = self.get_combined_space().get_minimal_time_distance()
        
        for i in range (len(self.delta_time_settings)):
            if not (self.delta_time_settings[0][i] >= minimal_time_distance):
                continue
            else:
                break
        
        self.delta_time = self.delta_time_settings[1][i]
        
        for space in self.spaces:
            space.delta_time = self.delta_time
        
        return self.delta_time# Return just for lols?
    
    def proceed_time(self, delta_time):
        
        for space in self.spaces:
            space.proceed_time(delta_time)
            
        self.time = self.time + delta_time
        self.tick = self.tick + 1
    
    def simulate_until(self, end_time):
        #start_time = datetime.now()
        
        start_time_simulation = self.time
        
        while True:
            message = ''
            
            if (self.tick%self.delta_time_settings[2] == 0) and (len(self.delta_time_settings[0]) > 1):
                self.set_delta_time()
                message = message + ' new dt:' + str(self.delta_time)
            
            if type(self.center) == str:
                for body in self.get_combined_space():
                    if body.name == self.center:
                        position_translation = body.position
                        velocity_translation = body.velocity
                
                for space in self.spaces:
                    space.translate(position_translation, velocity_translation)
            
            if (self.tick%self.split_settings[1] == 0) and (self.split_settings[0] > 1):
                
                message = message + ' split not yet implimented'
            
            if self.report:
                print(str(round(100*(self.time - start_time_simulation)/(end_time - start_time_simulation), 2)).rjust(6), '% tick', self.tick, 'time:',self.time, 's message:', message)
                # print(str(round(100*(self.time - start_time_simulation)/(end_time - start_time_simulation), 2)).rjust(6), 'tick', self.tick, 'time:', (datetime.fromtimestamp(self.time)-datetime.fromtimestamp(0)), ', took', (datetime.now() - start_time), 'message:', message)
                
            self.proceed_time(self.delta_time)
            
            if (end_time - self.time) <= self.delta_time:
                self.proceed_time(end_time - self.time)
                break




