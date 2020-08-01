import numpy as np
import random
from scipy.optimize import fsolve

#from helpers import *

class Body:
    def __init__(self, position=np.array([0.0, 0.0, 0.0]), velocity=np.array([0.0, 0.0, 0.0]), mass=1.0, radius=1.0, name=str(random.randint(0, 1000000)), real=True, color='b', start_time=0.0):
        """
        The body is a single particle in the simulation (it could be a plannet, astroid, spacecraft et cetera).
        
        >>> body = Body(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 100.0, 1.0, 'test name', True)
        """
        self.name = name
        self.color = color
        self.position = position
        self.velocity = velocity
        self.acceleration = np.array([0 for i in self.velocity])# Acceleration starts as zero with same dimension as velocity (and hopefully position).
        self.mass = mass
        self.radius = radius
        self.real = real
        self.time = start_time
        
        self.history = {'time':[], 'position':[], 'velocity':[]}#{'time':[self.time], 'position':[self.position], 'velocity':[self.velocity]}
        
        ## Validate the input of the body
        if len(self.position) != len(self.velocity):
            raise ValueError('Position and velocity vector of body', self.name, ' don\'t have same dimension', len(self.position), 'and', len(self.velocity))
        
        if type(self.mass) != int and type(self.mass) != float:
            raise ValueError('Mass of body', self.name, 'is a', type(self.mass), 'instead of a int or float.')
        
        if type(self.radius) != int and type(self.radius) != float:
            raise ValueError('Radius of body', self.name, 'is a', type(self.radius), 'instead of a int or float.')
        
        if type(self.real) != bool :
            raise ValueError('Real of body', self.name, 'is a', type(self.radius), 'instead of a bool.')
        
    
    def calculate_movement(self, delta_time):
        """
        Description
        
        >>> body = Body(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 100.0, 1.0, 'test name', True)
        >>> body.calculate_movement(1.0)
        >>> body.position
        array([0., 0., 0.])
        >>> body.velocity
        array([0., 0., 0.])
        
        >>> body.velocity = np.array([1.0, -1.0, 0])
        >>> body.calculate_movement(1.0)
        >>> body.position
        array([ 1., -1.,  0.])
        >>> body.velocity
        array([ 1., -1.,  0.])
        
        >>> body = Body(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 100.0, 1.0, 'test name', True)
        >>> body.acceleration = np.array([2.0, -1.5, 0.1])
        >>> body.calculate_movement(1.0)
        >>> body.position
        array([ 1.  , -0.75,  0.05])
        >>> body.velocity
        array([ 2. , -1.5,  0.1])
        
        >>> body.calculate_movement(1.0)
        >>> body.position
        array([ 4. , -3. ,  0.2])
        """
        self.history['time'].append(self.time)
        self.history['position'].append(self.position)
        self.history['velocity'].append(self.velocity)
        
        self.position = self.position + self.velocity * delta_time + (1/2) * self.acceleration * delta_time**2# The higher order term add accuraty but can be removed if delta_time is very small.
        self.velocity = self.velocity + self.acceleration * delta_time
        
        self.time = self.time + delta_time
        #self.acceleration = np.array([0.0 for i in self.velocity])# Reset the acceleration to zero.


if __name__ == "__main__":
    import doctest
    doctest.testmod()