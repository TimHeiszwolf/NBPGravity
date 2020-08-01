import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from body import Body
from helpers import *

class Space:
    def __init__(self, bodies=[], delta_time=0, start_time=0.0):
        """
        Description
        
        
        """
        self.bodies = bodies
        self.delta_time = delta_time
        self.time = start_time
        
        self.G = 6.67430 * 10**(-11)# The value of the gravitational constant in m^3/kgs^2
    
    
    def calculate_gravity(self, body1, body2):
        """
        Calculate the force of gravity that body 1 feels from body 2.
        
        """
        distance = body2.position - body1.position# The distance from body 1 to body 2.
        
        return self.G * body1.mass * body2.mass * distance / (np.linalg.norm(distance)**3)#Apply Newtons law of gravitation and create a unit vector.
    
    def calculate_acceleration_of_bodies(self):
        """
        Calculates the acceleration between the bodies. Currently only two body interaction allowed.
        
        """
        
        for body in self.bodies:
            body.acceleration = np.array([0.0 for i in body.velocity])# Reset the acceleration to zero.
        
        for i in range(len(self.bodies)):
            for j in range(i+1, len(self.bodies)):
                #The force between two bodies is only calculates once. The rest is done using Newtons third law.
                #if i==j:
                #    continue
                
                #print(i, j)
                body1 = self.bodies[i]
                body2 = self.bodies[j]
                force = self.calculate_gravity(body1, body2)
                
                body1.acceleration = body1.acceleration + force / body1.mass
                body2.acceleration = body2.acceleration - force / body2.mass
    
    def proceed_time(self, delta_time):
        """
        Proceeds the time by one delta time/tick. Includes calculating the acceleration and movement. The delta_time input should nearly always be self.delta_time unless needed otherwise (such as in proceed_time_until)
        """
        self.calculate_acceleration_of_bodies()
        
        for body in self.bodies:
            #print(body.name, body.position)
            body.calculate_movement(delta_time)
        
        self.time = self.time + delta_time
    
    def proceed_time_until(self, end_time):
        """
        Proceeds time until exactly the end time.
        
        >>> bodies = get_test_Space_simple_solar()
        >>> space = Space(bodies, 60.0, 0.0, 0)
        >>> [body.name for body in space.bodies]
        ['Sun', 'Earth', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
        >>> space.bodies[1].position
        array([-1.49900613e+11, -2.55535228e+10, -2.22996554e+04])
        
        >>> space.proceed_time_until(500.0)
        >>> space.time
        500.0
        
        >>> space.bodies[1].position
        array([-1.49898103e+11, -2.55679560e+10, -2.23121796e+04])
        
        >>> space.delta_time = 3600.0
        >>> space.proceed_time_until(1*24*3600)
        >>> space.time
        86400.0
        
        >>> space.bodies[1].position
        array([-1.49445702e+11, -2.80439964e+10, -2.30294090e+04])
        
        >>> space.proceed_time_until(365.25636*24*3600)
        >>> space.time
        31558149.504
        
        >>> space.bodies[1].position
        array([-1.51001466e+11, -2.27784293e+10, -1.27139388e+06])
        
        
        >>> np.linalg.norm(space.bodies[1].position - space.bodies[0].position)/1000
        152761080.89085364
        """
        
        while True:
            self.proceed_time(self.delta_time)
            
            if (end_time - self.time) <= self.delta_time:
                self.proceed_time(end_time - self.time)
                break
    
    def get_real_bodies(self):
        """
        Gets all the bodies which are real and returns them as a list.
        """
        real_bodies = []
        
        for body in self.bodies:
            if body.real:
                real_bodies.append(body)
        
        return real_bodies
    
    def translate(self, position_translation, velocity_translation):
        """
        Translates the bodies by a certain positional (and velocity) translation.
        """
        for body in self.bodies:
            body.position = body.position - position_translation
            body.velocity = body.velocity - velocity_translation
    
    def get_minimal_time_distance(self):
        time_distances = []
        
        for i in range(len(self.bodies)):
            for j in range(i+1, len(self.bodies)):
                body1 = self.bodies[i]
                body2 = self.bodies[j]
                
                time_distances.append(np.linalg.norm(body2.position - body1.position) / np.linalg.norm(body2.velocity - body1.velocity))
                
        return min(time_distances)
                
                
    
    def get_potential(self, position):
        """
        Calculates the gravitational potential at a certain point in the simulation.
        
        >>> bodies = get_test_Space_simple_solar()
        >>> space = Space(bodies, 3600.0, 0.0, 0)
        >>> space.get_potential(np.array([10**25, 0.0, 0.0]))
        -1.3289214696861148e-05
        
        >>> space.get_potential(space.bodies[1].position + np.array([space.bodies[1].radius, 0.0, 0.0]))
        -935531634.7636553
        """
        potential = 0.0
        
        for body in self.bodies:
            potential = potential - self.G * body.mass / np.linalg.norm(body.position - position)
        
        return potential

"""
bodies = get_test_Space_simple_solar()
space = Space(bodies, 3600.0, 0.0, 0)
simple_plotter(space, 365*24*3600, 24*3600)

#"""


if __name__ == "__main__":
    import doctest
    doctest.testmod()