from landlab import Component
from landlab import RasterModelGrid
from landlab.components import FlowAccumulator
from landlab.components import PriorityFloodFlowRouter
from landlab.utils.return_array import return_array_at_node
import numpy as np
from collections import deque

PI = 3.14159265359
SECPERYEAR = 31556926

def _cosarctan(slope):
    ''' Calculates the cosine of a fractional slope. Uses cos(arctan(slope)) = 1/((1+slope**2)**0.5) 
    '''
    return 1/((1 + slope**2)**0.5)
    
def _sinarctan(slope):
    ''' Calculates the sine of a fractional slope. Uses sin(arctan(slope)) = slope/((1+slope**2)**0.5) 
    '''
    return slope/((1 + slope**2)**0.5)

def _circular_velocity_0(discharge, AB3):
    ''' Calculates the velocity at (0,0) which satisfies the discharge through a semicircular glacier where vel=0 at the circumference.
    '''
    return (discharge/(PI*2/3*(-1/AB3)**0.5))**(2/3)

def _max_radius(discharge, AB3, velocity_0=None):
    if velocity_0 == None:
        velocity_0 = _circular_velocity_0(discharge, AB3)
    return (-4*velocity_0/AB3)**0.25

def _clamp(value, lower_bound, upper_bound):
    ''' Limits the of  value to one between lower and upper bounds.
    '''
    if upper_bound < lower_bound:
        print('_clamp function got an upper_bound that is lower than lower_bound. Reversing inputs...')
        upper_bound, lower_bound = lower_bound, upper_bound
    return max(lower_bound, min(upper_bound, value))

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
        nonlinear_mass_balance=False,
        melt_rate_scaling_factor=1,
        width_scaling_exp=0.3,
        width_scaling_const=1,
        thickness_to_width_ratio=0.25,
        density_ice=920,
        grav_accel = 9.8,
        glen_exp = 3, #Not used at the moment, hardcoded, equations need updating.
        erosion_exp = 2,
        erosion_const = 2.5*10**(-6),
        glen_const = 24*10**(-25),
        ):
        
        """Initialize the GlacialErosion model.

        Parameters
        ----------
        grid : ModelGrid
            Landlab ModelGrid object
        equilibrium_line_altitude : float or None
            Elevation of the equilibrium line, where ice accumulation == ablation [m]. If set to the standard value of None, assumes that all precipitation is converted to ice.
        full_ice_altitude : float or None
            Elevation of the line where all precipitation is converted to ice [m]. If set to the standard value of None, assumes that all precipitation is converted to ice.
        precipitation_rate : array or float
            Rate of precipitation [m/a].
        nonlinear_mass_balance : bool
            Whether to use the nonlinear mass balance model by Liebl et al. (2023). If set to False, uses the linear model by Hergarten (2021). For a comparison, see the Liebl et al. (2023) paper. 
        melt_rate_scaling_factor : float
            Scaling factor for adjusting the mass balance below the ELA. Set to values > 1 for faster melt and < 1 for slower melt.
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
        self._nonlinear_mass_balance = nonlinear_mass_balance
        self._melt_rate_scaling_factor = melt_rate_scaling_factor
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
    
    def _build_donor_dict(self):
        """Build a dictionary mapping each node to its donors. Called after flow routing."""
        self._donor_dict = {i: [] for i in range(self._grid.number_of_nodes)}
        
        for donor_node, receiver_node in enumerate(self._grid.at_node["flow__receiver_node"]):
            if donor_node != receiver_node:  # Don't add self-loops
                self._donor_dict[receiver_node].append(donor_node)

    def _cardinal_flowline(self, node):
        '''Find the cardinal flow line for the node (series of largest donors leading to the node)
        '''
        original_node = node
        cardinal_flowline = [original_node]
        end = False
        while not end:
            donors = self._donor_dict[node]
            if not donors:
                end = True
            elif len(donors) == 1:
                cardinal_flowline.append(donors[0])
                node = donors[0]
            else: 
                # Find the donor with largest flow:
                largest_disch = self._grid.at_node["ice__discharge"][donors[0]]
                largest_donor = donors[0]
                for donor in donors[1:]:
                    if self._grid.at_node["ice__discharge"][donor] > largest_disch:
                        largest_donor = donor
                        largest_disch = self._grid.at_node["ice__discharge"][donor]
                cardinal_flowline.append(largest_donor)
                node = largest_donor

            if self._dist_two_nodes(original_node, node) > 1/2*self._grid.at_node['glacier__width'][original_node]:
                end = True
                    
        return cardinal_flowline
    
    def _donors_in_swath(self, node, original_node, width_swath, use_flow_network=True, excluded_nodes=None):
        """Finds the donors of a node and the donors of those donors. 
        Stops at a donor that is in the excluded nodes list or is outside the glacier width.
        Uses BFS with a deque."""
        
        if excluded_nodes is None:
            excluded_nodes = []

        if use_flow_network == True:
            swath = []
            visited = set(excluded_nodes)  # Track visited nodes to avoid duplicates or nodes in the excluded list
            queue = deque([node])  # Use deque for O(1) popleft

            half_width = width_swath / 2
            
            while queue:
                current = queue.popleft()
                donors = self._donor_dict[current]
                
                for donor in donors:
                    if donor not in visited:
                        dist = self._dist_two_nodes(donor, original_node)
                        if dist < half_width:
                            swath.append(donor)
                            visited.add(donor)
                            queue.append(donor)  # Continue searching from this donor
        else:
            dx = self._grid.node_x - self._grid.node_x[node]
            dy = self._grid.node_y - self._grid.node_y[node]
            distances = np.sqrt(dx**2 + dy**2)
            swath = np.where((distances > 0) & (distances < width_swath/2))[0].tolist()
        
        return swath
    
    def _determine_flow(self):
        if self._equilibrium_line_altitude == None or self._full_ice_altitude == None:
            # All precipitation is converted to ice
            self._precipitation_rate_ice = self._precipitation_rate
        else:
            # Precipitation is converted to ice according to the mass balance model used
            elevation = self._grid.at_node["topographic__elevation"]
            ela = self._equilibrium_line_altitude
            fia = self._full_ice_altitude
            ice_multiplier = (elevation - ela) / (fia - ela)
            
            below_ela = elevation < ela
            # Apply nonlinear mass balance model, if defined by the parameter
            if self._nonlinear_mass_balance == True:
                fraction = (ela - elevation[below_ela])/(fia - ela)
                ice_multiplier[below_ela] = -fraction - 1/3*fraction**2
            
            # Adjust melt rate according to scaling factor
            ice_multiplier[below_ela] = ice_multiplier[below_ela]*self._melt_rate_scaling_factor

            self._precipitation_rate_ice = self._precipitation_rate * ice_multiplier.clip(max=1)

        # Run flow routing using ice precipitation
        fa = PriorityFloodFlowRouter(self._grid, runoff_rate=self._precipitation_rate_ice)
        fa.run_one_step()

        # Calculate ice discharge along cardinal flow line
        ice_discharge = np.maximum(return_array_at_node(self._grid, "surface_water__discharge"),0)
        _ = self._grid.add_field('ice__discharge', ice_discharge, at='node', clobber=True)

        #To do: Remove the temporary surface_water__discharge field to avoid confusion
        #To do: Run flowrouter again with precipitation of water? 

        # Calculate glacier width
        glacier_width = self._width_scaling_const*(ice_discharge*self._grid.dx)**self._width_scaling_exp
        _ = self._grid.add_field('glacier__width', glacier_width, clobber=True)

        self._build_donor_dict()  # Pre-compute donor relationships

    
    def calc_swaths(self, use_flow_network=False, exclude_cardinal_nodes=False):
        self.swaths = []
        swaths_number_of_nodes = np.zeros(self._grid.number_of_nodes)

        for center_node in range(self._grid.number_of_nodes):
            if self._grid.at_node['glacier__width'][center_node]/2 > self._grid.dx:

                if exclude_cardinal_nodes == True:
                    excluded_nodes = self._cardinal_flowline(center_node)
                else:
                    excluded_nodes = []

                donor_indices = self._donors_in_swath(center_node, center_node, self._grid.at_node['glacier__width'][center_node], use_flow_network=use_flow_network, excluded_nodes=excluded_nodes)
                swath = [center_node] + donor_indices

            else:
                swath = [center_node]

            self.swaths.append(swath)

            for node in swath:
                swaths_number_of_nodes[node] = len(swath)

        _ = self._grid.add_field('glacier__swath_number_of_nodes', swaths_number_of_nodes, clobber=True)

    def calc_ice_thicknesses(self):
        ice_elevation = np.full(self._grid.number_of_nodes, -np.inf)

        topography = self._grid.at_node["topographic__elevation"]
        glacier_width = self._grid.at_node['glacier__width']

        for swath in self.swaths:
            center_ice_level = topography[swath[0]] + glacier_width[swath[0]]*self._thickness_to_width_ratio
            ice_elevation[swath] = np.maximum(np.maximum(ice_elevation[swath], topography[swath]), center_ice_level)

        _ = self._grid.add_field('ice__elevation', ice_elevation, clobber=True)
        _ = self._grid.add_field('ice__thickness', ice_elevation - topography, clobber=True)

        #Take the average slope for the ice claculations, as using the maximum slope does not produce U-shaped valleys.
        ice_slope = self._grid.calc_slope_at_node(elevs='topographic__elevation')
        _ = self._grid.add_field('ice__slope', ice_slope, clobber=True)

    def cross_section_integration(self, center_node, constant_ice_level=False):
        slope = _clamp(self._grid.at_node['ice__slope'][center_node], 0.0001, 1)
        width = self._grid.at_node['glacier__width'][center_node]
        discharge = self._grid.at_node['ice__discharge'][center_node]/SECPERYEAR
        B = -0.5*self._density_ice*self._grav_accel*_sinarctan(slope)
        AB3 = self._glen_const*B**3
        cos_correction = _cosarctan(slope)

        swath = self.swaths[center_node]
        
        #Calculate distances accross the swath
        swath_array = np.array(swath)
        dx = self._grid.node_x[swath_array] - self._grid.node_x[center_node]
        dy = self._grid.node_y[swath_array] - self._grid.node_y[center_node]
        distances = np.sqrt(dx**2 + dy**2)

        # Sort swath by distance
        sort_indices = np.argsort(distances)
        distances = distances[sort_indices]
        swath = swath_array[sort_indices]

        #Obtain ice and topography elevations
        center_ice_surface = self._grid.at_node["ice__elevation"][center_node]
        topography_elevations = self._grid.at_node["topographic__elevation"][swath]
        ice_surface_elevations = self._grid.at_node["ice__elevation"][swath]

        if constant_ice_level == True:
            ice_surface_elevations = np.full_like(ice_surface_elevations, center_ice_surface)

        # Divide the distance between center and rim of glacier (halfwidth) into buckets, by calculating the distance between midpoints between nodes
        if len(swath) > 1:
            bucket_widths = np.zeros(len(distances))
            bucket_widths[0] = distances[1] / 2
            bucket_widths[1:-1] = (distances[2:] - distances[:-2]) / 2
            bucket_widths[-1] = (distances[-1] - distances[-2]) / 2 + width / 2 - distances[-1]
        else:
            bucket_widths = np.array([width / 2])

        
        #Where ice extends below the max range, set to the max range
        max_radius = _max_radius(discharge, AB3)
        max_thickness_below = max_radius**2 - np.square(distances) 

        thicknesses_above_center = np.clip((ice_surface_elevations - center_ice_surface)*cos_correction, a_min=0, a_max=None)
        ice_top_below_center = np.clip((center_ice_surface - ice_surface_elevations)*cos_correction, a_min=0, a_max=max_thickness_below) 
        ice_bottom_below_center = np.clip((center_ice_surface - topography_elevations)*cos_correction, a_min=0, a_max=max_thickness_below)

        #Where nodes are completely outside the max allowed radius (further horizontally), thicknesses should be set to 0.
        outside_range = distances > width/2
        thicknesses_above_center[outside_range] = 0
        ice_top_below_center[outside_range] = 0
        ice_bottom_below_center[outside_range] = 0

        area_cross_section = np.sum(bucket_widths*(ice_bottom_below_center - ice_top_below_center + thicknesses_above_center))
        
        if area_cross_section !=0:
            #Ice above the center line is assumed to move at the same rate as the surface (without deformation), integrated separately
            flow_below_center = np.sum(bucket_widths*(
                1/5 * (ice_bottom_below_center**5 - ice_top_below_center**5) + 
                2/3 * distances**2 * (ice_bottom_below_center**3 - ice_top_below_center**3) + 
                distances**4 * (ice_bottom_below_center - ice_top_below_center)
                ))
            flow_above_center = np.sum(bucket_widths*thicknesses_above_center*distances**4)
            flow = flow_below_center + flow_above_center

            velocity_0 = (discharge - AB3/4*flow*2)/(area_cross_section*2)
            basal_velocities = AB3/4*((ice_bottom_below_center**2 + distances**2)**2) + velocity_0
            basal_velocities[outside_range] = 0
        else:
            basal_velocities = np.zeros(len(swath))

        return swath, basal_velocities

    def calc_basal_velocities(self):
        sliding_velocities = np.zeros(self._grid.number_of_nodes)

        for swath in self.swaths:
            sorted_swath, basal_velocities = self.cross_section_integration(swath[0])
            sliding_velocities[sorted_swath] = np.maximum(sliding_velocities[sorted_swath], basal_velocities*SECPERYEAR)

        _ = self._grid.add_field('ice__sliding_velocity', sliding_velocities, clobber=True)
        _ = self._grid.add_field('ice__erosion_rate', self._erosion_const * sliding_velocities**self._erosion_exp, clobber=True)
       
    def run_one_step(self, dt, stabilise_erosion=False):
        self.calc_swaths(use_flow_network=False, exclude_cardinal_nodes=False)
        self.calc_ice_thicknesses()
        self.calc_basal_velocities()

        erosion_elevation = self._grid.at_node["topographic__elevation"] - self._grid.at_node['ice__erosion_rate']*dt

        if stabilise_erosion == True:
            receiver_elevation = self._grid.at_node["topographic__elevation"][self._grid.at_node['flow__receiver_node']]
            self._grid.at_node["topographic__elevation"] = np.maximum(erosion_elevation, receiver_elevation)
        else:
            self._grid.at_node["topographic__elevation"] = erosion_elevation

    def erode_topography(self, timestep_size=100, num_timesteps=5, flowroute_recalc_interval = 1):
        for t in range(num_timesteps):
            if t % flowroute_recalc_interval == 0:
                self._determine_flow()
            self.run_one_step(timestep_size)
