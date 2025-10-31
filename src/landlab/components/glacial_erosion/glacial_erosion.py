from landlab import Component
from landlab import RasterModelGrid
from landlab.components import FlowAccumulator
from landlab.components import PriorityFloodFlowRouter
from landlab.utils.return_array import return_array_at_node
import numpy as np

PI = 3.14159265359
SECPERYEAR = 31556926

def _cosarctan(slope):
    ''' Calculates the cosine of a fractional slope. Uses cos(arctan(slope)) = 1/((1+slope**2)**0.5) 
    '''
    return 1/((1 + slope**2)*0.5)
    
def _sinarctan(slope):
    ''' Calculates the sine of a fractional slope. Uses sin(arctan(slope)) = slope/((1+slope**2)**0.5) 
    '''
    return slope/((1 + slope**2)*0.5)

class Swath():
    def __init__(
        self,
        AB3,
        discharge,
        slope_angle,
        width,
        max_ice_thickness,
        node_ids,
        node_elevations,
        node_distances,
        max_iteration_attempts = 100,
        velocity_error_margin = 0.01,
        ):

        if slope_angle == 0:
            print('Warning: slope_angle in swath is exactly 0, for which the equations break. Setting slope to be 0.001')
            slope_angle = 0.001
        
        self._AB3 = AB3
        self.discharge = discharge
        self.slope_angle = slope_angle
        self.width = width

        self.iteration_attempts = max_iteration_attempts
        self.velocity_error_margin = velocity_error_margin

        self.ice_level = node_elevations[0] + max_ice_thickness

        # Sort the swath nodes by distance, required to integrate over the swath
        sort_zip = sorted(zip(node_distances, node_elevations, node_ids))

        self.node_distances = [i for i,j,k in sort_zip]
        self.node_elevations = [j for i,j,k in sort_zip]
        self.node_ids = [k for i,j,k in sort_zip]

        # Calculate initial estimates for ice thickness, velocity at the glacier center and maximal radius
        self.ice_thicknesses = [max(0, (self.ice_level - elev)*_cosarctan(self.slope_angle)) for elev in self.node_elevations]
        self.velocity_0 = (self.discharge/(PI*2/3*(-1/self._AB3)**0.5))**(2/3) #The velocity at (0,0) which yeilds the discharge through a semicircular glacier with maximal radius.
        self._max_allowed_radius()
    
    def _numerical_velocity_integration(self):
        # Divide the distance between center and rim of glacier (halfwidth) into buckets, by calculating the distance between midpoints between nodes
        if len(self.node_distances) > 1:
            self._bucket_widths = [self.node_distances[1]/2] + [(self.node_distances[i+1] - self.node_distances[i-1])/2 for i in range(1, len(self.node_distances) - 1)] + [(self.node_distances[-1] - self.node_distances[-2])/2 + self.width/2 - self.node_distances[-1]]
        else:
            self._bucket_widths = [self.width/2]

        updated_ice_thicknesses = []
        basal_velocities = []

        for n, node in enumerate(self.node_ids):
            if self.node_distances[n] > self.max_radius: #Case where nodes are completely outside the max allowed radius
                updated_ice_thicknesses.append(0)
                basal_velocities.append(0)
            elif self.node_distances[n]**2 + self.ice_thicknesses[n]**2 > self.max_radius**2: #Case where node elevations are below the max allowed radius
                updated_ice_thicknesses.append((self.max_radius**2 - self.node_distances[n]**2)**0.5)
                basal_velocities.append(0)
            else:
                updated_ice_thicknesses.append(self.ice_thicknesses[n])
                basal_velocities.append(1)

        area_cross_sections = [self._bucket_widths[i]*updated_ice_thicknesses[i] for i in range(len(self._bucket_widths))]
        flows_cross_sections = [self._bucket_widths[i]*(1/5*updated_ice_thicknesses[i]**5 + 2/3*updated_ice_thicknesses[i]**3*self.node_distances[i]**2 + updated_ice_thicknesses[i]*self.node_distances[i]**4) for i in range(len(self.node_distances))]
        
        if sum(area_cross_sections) !=0:
            self.velocity_0 = (self.discharge - self._AB3/4*sum(flows_cross_sections)*2)/(sum(area_cross_sections)*2)
            self.basal_velocities = [basal_velocities[i]*(self._AB3/4*(updated_ice_thicknesses[i]**2 + self.node_distances[i]**2)**2 + self.velocity_0) for i in range(len(self.node_distances))]
        else:
            self.velocity_0 = 0
            self.basal_velocities = [0]*len(self.node_ids)
        return updated_ice_thicknesses
    
    def _max_allowed_radius(self):
        self.max_radius = (-4*self.velocity_0/self._AB3)**0.25
    
    def find_basal_velocities(self):
        radii = [(self.ice_thicknesses[i]**2 + self.node_distances[i]**2)**0.5 for i in range(len(self.node_distances))]

        if max(radii) <= self.max_radius:
            # Directly numerically integrate to obtain basal velocities.
            self._numerical_velocity_integration()

        elif min(radii) >= self.max_radius:
            # Bedrock surface is completely outside of the (positive) flow of the glacier, therefore basal velocities are zero.
            self.basal_velocities = [0]*len(self.node_ids)

        else:
            previous_velocity_0 = self.velocity_0
            # Here, iteratively calculate velocities such that the total discharge is solved but there are no nonnegative velocities.
            for i in range(self.iteration_attempts):
                # This is done by setting the ice_thicknesses to have radii at most at the radius where velocity = 0, then numerically integrating the velocities. 
                updated_ice_thicknesses = self._numerical_velocity_integration()
                # This yields a u_0 larger than before, which means the max radius increases. Repeat until u_0 is stable.
                if abs(previous_velocity_0 - self.velocity_0) <= self.velocity_error_margin:
                    self.ice_thicknesses = updated_ice_thicknesses
                    break
                else:
                    self._max_allowed_radius()
                    previous_velocity_0 = self.velocity_0

            if i == self.iteration_attempts - 1:
                print('Maximal iteration attempts reached for swath, continuing with last obtained solution.')

class GlacialErosion(Component):
    """

    """

    _name = "GlacialErosion"

    _unit_agnostic = True

    _info = {}

    def __init__(
        self,
        grid,
        equilibrium_line_altitude=None,
        full_ice_altitude=None,
        precipitation_rate=1.,
        width_scaling_exp=0.3,
        width_scaling_const=1, #150
        thickness_to_width_ratio=0.25,
        density_ice=920,
        grav_accel = 9.8,
        glen_exp = 3, #Not used at the moment, hardcoded. See if solutions to differential equations can handle varying this.
        erosion_exp = 2,
        erosion_const = 2.5*10**(-6),
        glen_const = 24*10**(-25),
        ):
        
        """Initialize the GlacialErosion model.

        Parameters
        ----------
        grid : ModelGrid
            Landlab ModelGrid object
        precipitation_rate : array or float
            Rate of precipitation [m/a].
        equilibrium_line_altitude : float
            Elevation of the equilibrium line, where ice accumulation == ablation [m]. If set to the standard value of None, assumes that all precipitation is converted to ice.
        full_ice_altitude : float
            Elevation of the line where all precipitation is converted to ice [m]. If set to the standard value of None, assumes that all precipitation is converted to ice.
        width_scaling_exp : float
            Discharge to glacier width power law exponent. Defaults to 0.3 (Hergarten, 2021).
        width_scaling_const: 
            Discharge to glacier width power law proportionality constant (units vary depending on the value of width_scaling_exp).
        thickness_to_width_ratio:
            Assumed thickness to width ratio for the glacier [-]. Should be <0.5 to prevent gaps in the glaciers. Defaults to 0.25 (Liebl et al 2023).
        density_ice : float
            Denisty of ice [kg/m^3]. Defaults to 920 kg/m^3.
        grav_accel : float
            Gravitation acceleration [m/s^2]. Defaults to 9.8 m/s^2.
        glen_exp : float
            Glen-Nye flow law exponent. Defaults to 3.
        erosion_exp : float
            Basal velocity to erosion rate power law exponent. Defaults to 2.
        erosion_const : float
            Basal velocity to erosion rate proportionality constant (units vary depending on the value of erosion_exp). Defaults to 2.5*10**(-6) a/m (Braedstrup et al., 2016).
        glen_const : float
            Glen-Nye flow law proportionality constant (units vary depending on the value of glen_exp). Defaults to 24*10**(-25) s^-1 Pa^-3 (Budd & Jacka 1989, Cuffey & Patterson: The Physics of Glaciers).
        """

        super().__init__(grid)

        if isinstance(grid, RasterModelGrid):
            self._link_lengths = grid.length_of_d8
        else:
            self._link_lengths = grid.length_of_link

        if isinstance(precipitation_rate, (int, float)):
            precipitation_rate = np.full(grid.number_of_nodes, precipitation_rate, dtype=np.float64)

        self._width_scaling_exp = width_scaling_exp
        self._width_scaling_const = width_scaling_const
        self._thickness_to_width_ratio = thickness_to_width_ratio
        self._density_ice = density_ice
        self._grav_accel = grav_accel
        self._glen_exp = glen_exp
        self._erosion_exp = erosion_exp
        self._erosion_const = erosion_const
        self._glen_const = glen_const

        self._precipitation_rate = precipitation_rate
        self._equilibrium_line_altitude = equilibrium_line_altitude
        self._full_ice_altitude = full_ice_altitude
        self._determine_flow()
        
    def _dist_two_nodes(self, node1, node2):
        '''Calculate the horizontal euclidian distance between two nodes on the modelgrid.
        node1 : int
            index of the first node
        node2 : int
            index of the second node
        '''
        if node1 == node2:
            distance = 0
        else: 
            distance = ((self._grid.node_x[node1] - self._grid.node_x[node2])**2 + (self._grid.node_y[node1] - self._grid.node_y[node2])**2)**0.5
        return distance
    
    def _donors(self, node):
        '''List indices of nodes which flow into the target node, exept itself.
        '''
        return [i for i in range(len(self._flow_receivers)) if self._flow_receivers[i] == node and i != node]

    def _cardinal_flowline(self, node):
        '''Find the cardinal flow line for the node (series of largest donors leading to the node)
        '''
        original_node = node
        cardinal_flowline = [node]
        end = False
        while not end:
            donors = self._donors(node)
            if not donors:
                end = True
            elif len(donors) == 1:
                cardinal_flowline.append(donors[0])
                node = donors[0]
            else: 
                # Find the donor with largest flow:
                largest_disch = self._ice_discharge[donors[0]]
                largest_donor = donors[0]
                for donor in donors[1:]:
                    if self._ice_discharge[donor] > largest_disch:
                        largest_donor = donor
                        largest_disch = self._ice_discharge[donor]
                cardinal_flowline.append(largest_donor)
                node = largest_donor

            if self._dist_two_nodes(original_node, node):
                end = True
                    
        return cardinal_flowline
    
    def _donors_in_swath(self, node, original_node, cardinal_flowline, width_swath):
        '''Recursive function that finds the donors of a node and the donors of those donors, etc. 
        Stops at a donor that is on the cardinal flowline or is outside of the swath width.
        '''
        # @Mats: keep track of the total discharge and update this when adding nodes?
        swath = []
        donors = self._donors(node)
        for donor in donors:
            if donor not in cardinal_flowline and self._dist_two_nodes(donor, original_node) < width_swath/2:
                swath.append(donor)
                swath += self._donors_in_swath(donor, original_node, cardinal_flowline, width_swath)
        return swath
    
    def _determine_flow(self):
        if self._equilibrium_line_altitude == None or self._full_ice_altitude == None:
            self._precipitation_rate_ice = self._precipitation_rate
        else:
            ice_multiplier = (self._grid.at_node["topographic__elevation"] - self._equilibrium_line_altitude)/(self._full_ice_altitude - self._equilibrium_line_altitude)
            self._precipitation_rate_ice = self._precipitation_rate * ice_multiplier.clip(max=1)

        fa = FlowAccumulator(self._grid, flow_director="D8", runoff_rate=self._precipitation_rate_ice, depression_finder="DepressionFinderAndRouter") #"D8", "DepressionFinderAndRouter"
        #fa = PriorityFloodFlowRouter(self._grid, runoff_rate=self._precipitation_rate_ice)
        fa.run_one_step()

        self._ice_discharge = return_array_at_node(self._grid, "surface_water__discharge")
        self._flow_receivers = self._grid.at_node["flow__receiver_node"]
        self._slope = self._grid.at_node["topographic__steepest_slope"]
        self._glacier_width = self._width_scaling_const*(self._ice_discharge*self._grid.dx)**self._width_scaling_exp
    
    def _update_ice_values(self, node, basal_velocity, ice_thickness, number_of_nodes_in_swath):
        if basal_velocity > self.sliding_velocities[node]:
            self.sliding_velocities[node] = basal_velocity
            self.ice_thickness[node] = ice_thickness
            self.erosion_rate[node] = self._erosion_const * basal_velocity**self._erosion_exp
            self.number_of_nodes_in_swath[node] = number_of_nodes_in_swath

    def erosion_rate_ice(self):
        self.sliding_velocities = np.zeros(self._grid.number_of_nodes)
        self.ice_thickness = np.zeros(self._grid.number_of_nodes)
        self.erosion_rate = np.zeros(self._grid.number_of_nodes)
        self.number_of_nodes_in_swath = np.zeros(self._grid.number_of_nodes)

        for center_node in range(self._grid.number_of_nodes):
            # Calculate glacier width and flow parameters
            slope_along_flow = max(self._slope[center_node], 0.0001) # Artifically set a lower bound to the slope to prevent it from becoming 0, which breaks equations in the swath routine.

            if self._glacier_width[center_node]/2 > self._grid.dx:
                # Find which nodes belong to the swath
                cardinal_flowline = self._cardinal_flowline(center_node)
                donor_indices = self._donors_in_swath(center_node, center_node, cardinal_flowline, self._glacier_width[center_node])

                swath_node_ids = [center_node] + donor_indices
                swath_node_distances = [self._dist_two_nodes(center_node, node) for node in swath_node_ids]
                swath_node_elevations = [self._grid.at_node["topographic__elevation"][node] for node in swath_node_ids]

                B = -0.5*self._density_ice*self._grav_accel*_sinarctan(slope_along_flow)
                AB3 = self._glen_const*B**3
                swath_object = Swath(AB3, self._ice_discharge[center_node]/SECPERYEAR, slope_along_flow, self._glacier_width[center_node], self._glacier_width[center_node]*self._thickness_to_width_ratio, swath_node_ids, swath_node_elevations, swath_node_distances)
                swath_object.find_basal_velocities()

                for n, node in enumerate(swath_object.node_ids):
                    self._update_ice_values(node, swath_object.basal_velocities[n]*SECPERYEAR, swath_object.ice_thicknesses[n], len(swath_object.node_ids))

    def erode_topography(self, timestep_size=100, num_timesteps=5, flowroute_recalc_interval = 1):
        for t in range(num_timesteps):
            if t % flowroute_recalc_interval == 0:
                self._determine_flow()
            self.erosion_rate_ice()
            self._grid.at_node["topographic__elevation"] -= self.erosion_rate*timestep_size
