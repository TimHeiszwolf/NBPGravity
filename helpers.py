import numpy as np
import time
import matplotlib.pyplot as plt
import imageio
from scipy.optimize import fsolve

from body import Body

def get_position_from_Kepler(semimajor_axis, eccentricity, inclination, ascending_node, argument_of_periapsis, mean_anomaly, mass_orbit, G=6.67430 * 10**(-11)):
    """
    Get the position vectors from the Keplerian coordinates
    
    First part from https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf
    Second part from https://space.stackexchange.com/questions/19322/converting-orbital-elements-to-cartesian-state-vectors
    
    >>> position = get_position_from_Kepler(1.5*10**8, 0.0167, (5*10**(-5))*np.pi/180, 1, 1, 190*np.pi/180, 1.988435 * (10**30))
    >>> position
    array([ 8.58449271e+07, -1.26004733e+08, -1.22449388e+02])
    >>> np.linalg.norm(position)
    152468174.39880842
    """
    
    mu = G * mass_orbit
    
    func = lambda EA: mean_anomaly - (EA - eccentricity * np.sin(EA))
    eccentric_anomaly = fsolve(func, np.pi)[0]
    
    true_anomaly = 2 * np.arctan2(np.sqrt(1 + eccentricity) * np.sin(eccentric_anomaly / 2), np.sqrt(1 - eccentricity) * np.cos(eccentric_anomaly / 2))
    radius = semimajor_axis * (1 - eccentricity * np.cos(eccentric_anomaly))
    
    h = np.sqrt(mu * semimajor_axis * (1 - eccentricity**2))
    p = semimajor_axis * (1 - eccentricity**2)
    
    Om = ascending_node
    w = argument_of_periapsis
    nu = true_anomaly
    r = radius
    i = inclination
    e = eccentricity
    
    x = r*(np.cos(Om)*np.cos(w+nu) - np.sin(Om)*np.sin(w+nu)*np.cos(i))
    y = r*(np.sin(Om)*np.cos(w+nu) + np.cos(Om)*np.sin(w+nu)*np.cos(i))
    z = r*(np.sin(i)*np.sin(w+nu))
    
    #print(x, r, Om, w, nu, i, e, eccentric_anomaly)
    
    position = np.array([x, y, z])
    
    xd = (x*h*e/(r*p))*np.sin(nu) - (h/r)*(np.cos(Om)*np.sin(w+nu) + np.sin(Om)*np.cos(w+nu)*np.cos(i))
    yd = (x*h*e/(r*p))*np.sin(nu) - (h/r)*(np.sin(Om)*np.sin(w+nu) - np.cos(Om)*np.cos(w+nu)*np.cos(i))
    zd = (x*h*e/(r*p))*np.sin(nu) - (h/r)*(np.cos(w+nu)*np.sin(i))
    
    velocity = np.array([xd, yd, zd])
    #print(velocity)
    
    return position


def get_coordinates_from_Kepler(semimajor_axis, eccentricity, inclination, ascending_node, argument_of_periapsis, mean_anomaly, current_velocity, mass_orbit, G=6.67430 * 10**(-11), delta=0.001):
    """
    Lol wtf pls kil me.
    
    >>> position, velocity = get_coordinates_from_Kepler(1.5*10**8, 0.0167, (5*10**(-5))*np.pi/180, 1, 1, 190*np.pi/180, 29300, 1.988435 * (10**30))
    >>> position
    array([ 8.58449271e+07, -1.26004733e+08, -1.22449388e+02])
    >>> velocity
    array([ 2.41591639e+04,  1.65778407e+04, -9.92410781e-03])
    >>> np.linalg.norm(position)
    152468174.39880842
    >>> np.linalg.norm(velocity)
    29299.999999999993
    """
    position = get_position_from_Kepler(semimajor_axis, eccentricity, inclination, ascending_node, argument_of_periapsis, mean_anomaly, mass_orbit, G)
    position_plus_delta = get_position_from_Kepler(semimajor_axis, eccentricity, inclination, ascending_node, argument_of_periapsis, mean_anomaly + delta, mass_orbit, G)
    
    delta_position = position_plus_delta - position
    
    direction_unit_vector = delta_position / np.linalg.norm(delta_position)
    
    return position, current_velocity * direction_unit_vector

def ld_to_m(ld):
    """
    Converts the input distance (or velocity) of the input from Lunar distances to meters.
    """
    return ld * 384402 * 10**3

def au_to_m(au):
    """
    Converts the input distance (or velocity) of the input from atronomical units to meters.
    """
    return au * 1.495978707 * 10**11

def ly_to_m(ly):
    """
    Converts the input distance (or velocity) of the input from light years to meters.
    """
    return ly * 9.4607 * 10**15

def pc_to_m(pc):
    """
    Converts the input distance (or velocity) of the input from parsec to meters.
    """
    return pc * 3.085677581 * 10**18

def make_gif(bodies, trail_length, tick_per_frame=10, frames_per_second=5, window=[[-1, 1], [-1, 1]], name='output', axis=[0, 1], labels=False):
    images = []
    min_trail = 0
    fig = plt.figure(figsize=(16, 16))
    
    for tick in range(0, len(bodies[0].history['time']), tick_per_frame):
        if bodies[0].history['time'][0] > (bodies[0].history['time'][tick] - trail_length):
            continue
        
        print('Rendering tick:', tick)
        
        current_time = bodies[0].history['time'][tick]
        
        
        x = [body.history['position'][tick][axis[0]] for body in bodies]
        y = [body.history['position'][tick][axis[1]] for body in bodies]
        colors = [body.color for body in bodies]
        
        plt.scatter(x, y, c=colors)
        plt.axis((window[0][0], window[0][1], window[1][0], window[1][1]))
        
        while bodies[0].history['time'][min_trail] + trail_length < current_time:
            min_trail = min_trail + 1
        
        for body in bodies:
            if labels:
                x_label = body.history['position'][tick][axis[0]]
                y_label = body.history['position'][tick][axis[1]]
                plt.text(x_label, y_label, body.name)
            
            #x_trail = [body.history['position'][i][axis[0]] for i in range(tick + 1) if ((body.history['time'][i] + trail_length) >= current_time)]
            #y_trail = [body.history['position'][i][axis[1]] for i in range(tick + 1) if ((body.history['time'][i] + trail_length) >= current_time)]
            x_trail = [body.history['position'][i][axis[0]] for i in range(min_trail, tick + 1)]
            y_trail = [body.history['position'][i][axis[1]] for i in range(min_trail, tick + 1)]
            plt.plot(x_trail, y_trail, c=body.color)
        
        plt.title(name + ' time ' + str(round(current_time, 0)))
        plt.xlabel('grgr')
        plt.ylabel('grgr')
        
        image_name = name + '/' + str(tick) + '.png'
        plt.savefig(image_name)
        images.append(imageio.imread(image_name))
        #plt.pause(0.0001)# Do we want this?
        plt.clf()
    
    print('Done rendering now saving gif.')
    imageio.mimwrite(name+'.gif', images, format='.gif', fps=frames_per_second)
    
    

def simple_plotter(space, end_time, time_per_second, updates_per_second=2):
    
    #plt.show()
    lim = 1.50*10**11#max([max([abs(pos) for pos in body.position]) for body in space.bodies])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    start_time = time.time()
    tick = 0
    #print(tick, space.time)
    while space.time<=end_time:
        print(tick, space.time, round(time.time()-start_time, 2), np.linalg.norm(space.bodies[1].position - space.bodies[2].position))
        time.sleep(max([0.001, tick - (time.time() - start_time)]))
        space.proceed_time_until(tick*time_per_second)
        
        x = []
        y = []
        
        for body in space.bodies:
            x.append(body.position[0])
            y.append(body.position[1])
        
        ax.clear()
        ax.scatter(x, y, marker='o', c='r')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        plt.pause(0.0001)
        tick = tick+1/updates_per_second
    
    plt.show()

def get_test_Space_simple_solar():
    """
    Generates a simple test Space object. It is filled with the 8 plannets of the solar system (and the moon). They are position in a way that doesn't 100% correspond to reality.
    """
    
    bodies = []
    
    mass_orbit = 1.988435 * (10**30)
    
    # The most important bodies.
    bodies.append(Body(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 1.988435 * (10**30), 695700000, 'Sun', True, 'tab:orange'))
    
    position_earth, velocity_earth = get_coordinates_from_Kepler(1.0*1.496*10**11, 0.01671, (5*10**(-5))*np.pi/180, 0, 0, 190*np.pi/180, 29300, mass_orbit)
    bodies.append(Body(position_earth, velocity_earth, 5.97 * (10**24), 6371009, 'Earth', True, 'tab:blue'))
    
    position, velocity = get_coordinates_from_Kepler(384400*1000, 0.0554, 5.16*np.pi/180, 125*np.pi/180, 318.15*np.pi/180, 213*np.pi/180, 1020, bodies[1].mass)
    position = position + position_earth
    velocity = velocity + velocity_earth
    bodies.append(Body(position,velocity, 7.349 * (10**22), 1737400, 'Moon', True, 'darkgrey'))
    
    # Other inner plannets.
    position, velocity = get_coordinates_from_Kepler(0.38709893*1.496*10**11, 0.20563069, 7.00487*np.pi/180, 48.33*np.pi/180, 29.12*np.pi/180, 269*np.pi/180, 45810, mass_orbit)
    bodies.append(Body(position, velocity, 3.301 * (10**23), 2440000, 'Mercury', True, 'lightsteelblue'))
    
    position, velocity = get_coordinates_from_Kepler(0.72333199*1.496*10**11, 0.00677, 3.39471*np.pi/180, 76.68069*np.pi/180, 54.85*np.pi/180, 187*np.pi/180, 34790, mass_orbit)
    bodies.append(Body(position, velocity, 4.867 * (10**24), 6050000, 'Venus', True, 'goldenrod'))
    
    position, velocity = get_coordinates_from_Kepler(1.52366*1.496*10**11, 0.09341, 1.85061*np.pi/180, 49.57*np.pi/180, 286*np.pi/180, 349*np.pi/180, 26450, mass_orbit)
    bodies.append(Body(position, velocity, 6.417 * (10**23), 3390000, 'Mars', True, 'sandybrown'))
    
    # Outer planets.
    position_jupiter, velocity_jupiter = get_coordinates_from_Kepler(5.2033*1.496*10**11, 0.04839, 1.3053*np.pi/180, 100.556*np.pi/180, -85.80*np.pi/180, 283*np.pi/180, 13170, mass_orbit)
    bodies.append(Body(position_jupiter, velocity_jupiter, 1.898 * (10**27), 69950000, 'Jupiter', True, 'darkorange'))
    
    position_saturn, velocity_saturn = get_coordinates_from_Kepler(9.537*1.496*10**11, 0.0541, 2.48446*np.pi/180, 113.715*np.pi/180, -21.2831*np.pi/180, 207*np.pi/180, 91590, mass_orbit)
    bodies.append(Body(position_saturn, velocity_saturn, 5.683 * (10**26), 58300000, 'Saturn', True, 'navajowhite'))
    
    position_uranus, velocity_uranus = get_coordinates_from_Kepler(19.1912*1.496*10**11, 0.0471771, 0.76986*np.pi/180, 74.22988*np.pi/180, 96.73436*np.pi/180, 229*np.pi/180, 6578, mass_orbit)
    bodies.append(Body(position_uranus, velocity_uranus, 8.681 * (10**25), 25360000, 'Uranus', True, 'powderblue'))
    
    position_neptune, velocity_neptune = get_coordinates_from_Kepler(30.06896*1.496*10**11, 0.00858587, 1.76917*np.pi/180, 131.72169*np.pi/180, -86.75*np.pi/180, 301*np.pi/180, 5449, mass_orbit)
    bodies.append(Body(position_neptune, velocity_neptune, 1.024 * (10**26), 24600000, 'Neptune', True, 'dodgerblue'))
    
    return bodies


if __name__ == "__main__":
    import doctest
    doctest.testmod()