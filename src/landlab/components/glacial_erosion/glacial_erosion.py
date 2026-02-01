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

def _max_radius(discharge, AB3):
    ''' Calculates for a given discharge the radius at which a semicircular glacier would have 0 velocity.
    '''
    return (discharge*12/(PI*abs(AB3)))**(1/6)

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
        discharge_override=None,
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
        self._discharge_override = discharge_override
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
        """Build a dictionary mapping each node to its donors."""
        self._donor_dict = {i: [] for i in range(self._grid.number_of_nodes)}
        receivers = self._grid.at_node["flow__receiver_node"]
        
        for donor_node, receiver_node in enumerate(receivers):
            if donor_node != receiver_node:  # Don't add self-loops
                self._donor_dict[receiver_node].append(donor_node)

    def _obtain_largest_donors(self):
        '''Cache the largest donor for each node. If there are no donors, the node itself is recorded, like receivers in flowrouting. Requires _build_donor_dict to have been called.'''
        # Initialize largest_donor to node itself. If there are no donors, the donor is the node itself.
        self._largest_donor = np.arange(self._grid.number_of_nodes, dtype=int)
        discharge = self._grid.at_node["glacier__discharge"]
        
        for node, donors in self._donor_dict.items():
            if len(donors) > 0:
                # Find donor with max discharge
                self._largest_donor[node] = max(donors, key=lambda d: discharge[d])
            # else: stays as -1 (no donor)

    def _cardinal_flowline(self, node):
        '''Find the cardinal flow line for the node (series of largest donors leading to the node)
        '''
        original_node = node
        cardinal_flowline = [original_node]
        max_distance = 0.5 * self._grid.at_node['glacier__width'][original_node]
        
        while True:
            largest_donor = self._largest_donor[node]
            
            if largest_donor == node:  # No donors
                break
            
            cardinal_flowline.append(largest_donor)
            node = largest_donor
            
            if self._dist_two_nodes(original_node, node) > max_distance:
                break
        
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

        if self._discharge_override is not None:
            glacier_discharge = self._discharge_override
        else:
            # Calculate ice discharge along cardinal flow line
            glacier_discharge = np.maximum(return_array_at_node(self._grid, "surface_water__discharge"),0)

        _ = self._grid.add_field('glacier__discharge', glacier_discharge, at='node', clobber=True)

        #To do: Remove the temporary surface_water__discharge field to avoid confusion
        #To do: Run flowrouter again with precipitation of water? 

        # Calculate glacier width
        glacier_width = self._width_scaling_const*(glacier_discharge*self._grid.dx)**self._width_scaling_exp
        _ = self._grid.add_field('glacier__width', glacier_width, clobber=True)


        self._build_donor_dict()  # Pre-compute donor relationships
        self._obtain_largest_donors() #Pre-compute largest donors for every node
        self._calc_flow_directions() #Pre-compute flow directions and slope
        self.calc_swaths()
        self.calc_ice_thicknesses()
    
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
                swaths_number_of_nodes[node] = max(len(swath), swaths_number_of_nodes[node])

        _ = self._grid.add_field('glacier__swath_number_of_nodes', swaths_number_of_nodes, clobber=True)

    def _calc_flow_directions(self):
        """Calculate 2D flow direction components and slopes for all nodes.
        Computed from the distances between upstream and downstream edge nodes of the swath.
        Stores results as grid fields.
        """
        # Initialize arrays
        num_nodes = self._grid.number_of_nodes
        flow_x_normalized = np.zeros(num_nodes)
        flow_y_normalized = np.zeros(num_nodes)
        slope = np.zeros(num_nodes)
        elevations = self._grid.at_node["topographic__elevation"]
        receivers = self._grid.at_node["flow__receiver_node"]
        
        for center_node in range(num_nodes):
            width = self._grid.at_node['glacier__width'][center_node]
            
            # Find upstream edge: follow largest donors until outside swath width
            upstream_node = center_node
            max_distance = width / 2
            
            while True:
                largest_donor = self._largest_donor[upstream_node]
                
                if largest_donor == upstream_node:  # No more donors
                    break
                
                dist = self._dist_two_nodes(center_node, largest_donor)
                
                if dist > max_distance:  # Outside swath
                    break
                
                upstream_node = largest_donor
            
            # Find downstream edge: follow receivers until outside swath width
            downstream_node = center_node
            
            while True:
                receiver = receivers[downstream_node]
                
                if receiver == downstream_node:  # No more receivers
                    break
                
                dist = self._dist_two_nodes(center_node, receiver)
                
                if dist > max_distance:  # Outside swath
                    break
                
                downstream_node = receiver
            
            # Calculate flow vectors (downstream - upstream)
            flow_x = self._grid.node_x[downstream_node] - self._grid.node_x[upstream_node]
            flow_y = self._grid.node_y[downstream_node] - self._grid.node_y[upstream_node]
            
            # Calculate elevation change
            delta_z = elevations[downstream_node] - elevations[upstream_node]
            
            # Calculate horizontal distance
            delta_d = np.sqrt(flow_x**2 + flow_y**2)
            
            # Normalize and calculate slope
            if delta_d > 0:
                flow_x_normalized[center_node] = flow_x / delta_d
                flow_y_normalized[center_node] = flow_y / delta_d
                slope[center_node] = abs(delta_z / delta_d)
            else:
                # If upstream and downstream are the same node, use zero values
                flow_x_normalized[center_node] = 0.0
                flow_y_normalized[center_node] = 0.0
                slope[center_node] = 0.0
        
        # Store as grid fields
        self._grid.add_field('glacier__flow_direction_x', flow_x_normalized, clobber=True)
        self._grid.add_field('glacier__flow_direction_y', flow_y_normalized, clobber=True)
        self._grid.add_field('glacier__slope', slope, clobber=True)
    
    def calc_ice_thicknesses(self):
        topography = self._grid.at_node["topographic__elevation"]
        ice_elevation = topography.copy()

        for swath in self.swaths:
            center_node = swath[0]
            slope = self._grid.at_node['glacier__slope'][center_node]
            cos_correction = _cosarctan(slope)
            width = self._grid.at_node['glacier__width'][center_node]
            flow_x = self._grid.at_node['glacier__flow_direction_x'][center_node]
            flow_y = self._grid.at_node['glacier__flow_direction_y'][center_node]
            
            #Calculate distances along flow
            swath_array = np.array(swath)
            dx = self._grid.node_x[swath_array] - self._grid.node_x[center_node]
            dy = self._grid.node_y[swath_array] - self._grid.node_y[center_node]
            distances_along = flow_x*dx + flow_y*dy

            #Obtain ice elevations
            center_ice_thickness_perpendicular = width*self._thickness_to_width_ratio
            center_ice_thickness_vertical = center_ice_thickness_perpendicular/cos_correction
            center_ice_level = topography[center_node] + center_ice_thickness_vertical
            ice_surface_elevations = -slope * distances_along + center_ice_level

            #Update ice elevation if the newly calculated ones are higher
            ice_elevation[swath_array] = np.maximum(ice_surface_elevations, ice_elevation[swath_array])

        _ = self._grid.add_field('ice__elevation', ice_elevation, clobber=True)
        _ = self._grid.add_field('ice__thickness', ice_elevation - topography, clobber=True)


    def dir_cross_section_integration(self, center_node, thickness_agnostic=False):
        width = self._grid.at_node['glacier__width'][center_node]
        discharge = self._grid.at_node['glacier__discharge'][center_node]/SECPERYEAR
        swath = self.swaths[center_node]

        #Obtain flow direction and slope
        flow_x = self._grid.at_node['glacier__flow_direction_x'][center_node]
        flow_y = self._grid.at_node['glacier__flow_direction_y'][center_node]
        slope = self._grid.at_node['glacier__slope'][center_node]
        #slope = _clamp(abs(slope), 0.0001, 1)

        if flow_x == 0 and flow_y ==0: #No flow, therefore no velocities.
            return swath, np.zeros(len(swath)), np.zeros(len(swath)), 0
        
        B = -0.5*self._density_ice*self._grav_accel*_sinarctan(slope)
        AB3 = self._glen_const*B**3
        cos_correction = _cosarctan(slope)

        #Calculate distances along and perpendicular to flow
        swath_array = np.array(swath)
        dx = self._grid.node_x[swath_array] - self._grid.node_x[center_node]
        dy = self._grid.node_y[swath_array] - self._grid.node_y[center_node]
        distances = (dx**2 + dy**2)**0.5
        distances_along = flow_x*dx + flow_y*dy
        distances_perp = flow_y*dx - flow_x*dy
        
        #Obtain topography and ice elevations
        topography_elevations = self._grid.at_node["topographic__elevation"][swath_array]
        ice_surface_elevations = self._grid.at_node["ice__elevation"][swath_array]
        center_ice_elevation = self._grid.at_node["ice__elevation"][center_node]
        reference_ice_elevation = -slope * distances_along + center_ice_elevation

        if thickness_agnostic == True:
            # Calculate ice thicknesses purely according to the center node
            center_ice_thickness_perpendicular = width*self._thickness_to_width_ratio
            center_ice_thickness_vertical = center_ice_thickness_perpendicular/cos_correction
            center_ice_elevation = self._grid.at_node["topographic__elevation"][center_node] + center_ice_thickness_vertical
            reference_ice_elevation = -slope * distances_along + center_ice_elevation
            ice_surface_elevations = reference_ice_elevation

        # --- Binning by perpendicular distance ---
        # To determine glacier cross section, we use inverse along-distance weighting to calculate average topographic elevations in the swath.

        # First find unique perpendicular distances, used as natural bins (grid-aligned)
        unique_distances_perp = np.unique(distances_perp)
        unique_distances_perp = unique_distances_perp[np.abs(unique_distances_perp) <= width / 2] # Filter out distances beyond glacier width
        unique_distances_perp = np.sort(unique_distances_perp)

        # Initialize arrays for binned values
        bin_distances = []
        bin_elevations_topo = []
        bin_deviations_ice = []
        bin_widths = []
    
        for dist in unique_distances_perp:
            at_this_distance = (distances_perp == dist)
            distance_along_bin = distances_along[at_this_distance]
            topo_bin = topography_elevations[at_this_distance]
            ice_bin = ice_surface_elevations[at_this_distance] - reference_ice_elevation[at_this_distance]
            
            # Inverse-distance weighting based on along-flow distance
            epsilon = 1e-6
            weights = 1.0 / (np.abs(distance_along_bin) + epsilon)
            weights /= weights.sum()  # Normalize weights
            
            # Weighted average elevation
            avg_topo = np.sum(weights * topo_bin)
            avg_ice = np.sum(weights * ice_bin)
            
            # Store bin data
            bin_distances.append(dist)
            bin_elevations_topo.append(avg_topo)
            bin_deviations_ice.append(avg_ice)

        # Convert to numpy arrays (in order from -width/2 to +width/2)
        bin_distances = np.array(bin_distances)
        bin_elevations_topo = np.array(bin_elevations_topo)
        bin_deviations_ice = np.array(bin_deviations_ice)

        # Calculate bin widths based on midpoints between unique distances
        if len(bin_distances) > 1:
            bin_widths = np.zeros(len(bin_distances))
            # First bin: from -width/2 to midpoint with next bin
            bin_widths[0] = (bin_distances[1] + bin_distances[0]) / 2 - (-width / 2)
            # Middle bins: half-distance to neighbors on each side
            bin_widths[1:-1] = (bin_distances[2:] - bin_distances[:-2]) / 2
            # Last bin: from midpoint with previous bin to +width/2
            bin_widths[-1] = width / 2 - (bin_distances[-1] + bin_distances[-2]) / 2
        else:
            bin_widths = np.array([width])  # Full width if only center node
        
        # --- Cross sectional integration ---
        #Where ice extends below the max range, set to the max range
        max_radius = _max_radius(discharge, AB3)
        max_thickness_below = np.maximum(0, max_radius**2 - np.abs(bin_distances)**2)

        bin_ice_thicknesses_above_center = np.clip(
            bin_deviations_ice * cos_correction, 0, None
        )
        bin_ice_top_below_center = np.clip(
            - bin_deviations_ice * cos_correction, 0, max_thickness_below
        )
        bin_ice_bottom_below_center = np.clip(
            (center_ice_elevation - bin_elevations_topo) * cos_correction, 0, max_thickness_below
        )

        # Where nodes are completely outside the max allowed radius (further horizontally), thicknesses should be set to 0.
        outside_range_bins = np.abs(bin_distances) > max_radius
        bin_ice_thicknesses_above_center[outside_range_bins] = 0
        bin_ice_top_below_center[outside_range_bins] = 0
        bin_ice_bottom_below_center[outside_range_bins] = 0

        area_cross_section = np.sum(
            bin_widths * (bin_ice_bottom_below_center - bin_ice_top_below_center + bin_ice_thicknesses_above_center)
        )
        
        flow_below_center = np.sum(
            bin_widths * (
                1/5 * (bin_ice_bottom_below_center**5 - bin_ice_top_below_center**5) +
                2/3 * bin_distances**2 * (bin_ice_bottom_below_center**3 - bin_ice_top_below_center**3) +
                bin_distances**4 * (bin_ice_bottom_below_center - bin_ice_top_below_center)
            )
        )
        #Ice above the center line is assumed to move at the same rate as the surface (without deformation), integrated separately
        flow_above_center = np.sum(bin_widths * bin_ice_thicknesses_above_center * bin_distances**4)
        total_ice_flow = flow_below_center + flow_above_center

        if area_cross_section != 0:
            velocity_0 = (discharge - AB3/4 * total_ice_flow) / area_cross_section
            
            # Calculate basal velocities using original node topography (not binned)
            node_ice_bottom = np.clip(
                (reference_ice_elevation - topography_elevations) * cos_correction, 
                0, 
                np.maximum(0, max_radius**2 - np.abs(distances_perp)**2)
            )
            
            basal_velocities = AB3/4 * ((node_ice_bottom**2 + distances**2)**2) + velocity_0 #Previously used distance_perp, but using standard distance lets the influence of a node decay at distance.
            surface_velocities = AB3/4 * (distances**4) + velocity_0

            # Nodes outside glacier width get zero velocity
            outside_range_nodes = np.abs(distances_perp) > max_radius
            basal_velocities[outside_range_nodes] = 0
            surface_velocities[outside_range_nodes] = 0

        else:
            basal_velocities = np.zeros(len(swath_array))
            surface_velocities = np.zeros(len(swath_array))
        
        return swath_array, basal_velocities, surface_velocities, area_cross_section

    def calc_velocities(self):
        sliding_velocities = np.zeros(self._grid.number_of_nodes)
        surface_velocities = np.zeros(self._grid.number_of_nodes)
        glacier_cross_section = np.zeros(self._grid.number_of_nodes)
        glacier_center_nodes = np.zeros(self._grid.number_of_nodes)
        
        for swath in self.swaths:
            sorted_swath, sliding_velocities_swath, surface_velocities_swath, area_cross_section = self.dir_cross_section_integration(swath[0], thickness_agnostic=False)
            sorted_swath = np.array(sorted_swath)
            
            # Convert to annual velocities
            sliding_annual = sliding_velocities_swath * SECPERYEAR
            surface_annual = surface_velocities_swath * SECPERYEAR
            
            # Find which nodes get updated (where new basal velocity is larger)
            nodes_to_update = sliding_annual > sliding_velocities[sorted_swath]
            
            # Update both velocities only where basal velocity increases
            sliding_velocities[sorted_swath[nodes_to_update]] = sliding_annual[nodes_to_update]
            surface_velocities[sorted_swath[nodes_to_update]] = surface_annual[nodes_to_update]
            glacier_cross_section[sorted_swath[nodes_to_update]] = area_cross_section
            glacier_center_nodes[sorted_swath[nodes_to_update]] = swath[0]
        
        _ = self._grid.add_field('ice__sliding_velocity', sliding_velocities, clobber=True)
        _ = self._grid.add_field('ice__surface_velocity', surface_velocities, clobber=True)
        _ = self._grid.add_field('ice__erosion_rate', self._erosion_const * sliding_velocities**self._erosion_exp, clobber=True)
        _ = self._grid.add_field('glacier__cross_section', glacier_cross_section, clobber=True)
        _ = self._grid.add_field('glacier__center_node', glacier_center_nodes, clobber=True)
       
    def run_one_step(self, dt, stabilise_erosion=False):
        self.calc_velocities()

        erosion_elevation = self._grid.at_node["topographic__elevation"] - self._grid.at_node['ice__erosion_rate']*dt

        if stabilise_erosion == True:
            receiver_elevation = self._grid.at_node["topographic__elevation"][self._grid.at_node['flow__receiver_node']]
            self._grid.at_node["topographic__elevation"] = np.maximum(erosion_elevation, receiver_elevation)
        else:
            self._grid.at_node["topographic__elevation"] = erosion_elevation

    def erode_topography(self, number_of_years=100, max_erosion=2, flowroute_recalc_interval=10):
        always_recalc_flowroute = False
        flowroute_recalc_stops = []

        if flowroute_recalc_interval is None:
            pass
        elif flowroute_recalc_interval == 0:
            always_recalc_flowroute = True
        else:
            num_stops = int(number_of_years/flowroute_recalc_interval)
            flowroute_recalc_stops = [i*flowroute_recalc_interval for i in range(1,num_stops)]

        years_elapsed = 0

        while years_elapsed < number_of_years:
            self.calc_velocities()
            highest_erosion_rate = np.max(self._grid.at_node['ice__erosion_rate'])


            dt = max_erosion/highest_erosion_rate

            if years_elapsed + dt >= number_of_years:
                dt = number_of_years - years_elapsed
            elif flowroute_recalc_stops:
                if years_elapsed + dt >= flowroute_recalc_stops[0]:
                    dt = flowroute_recalc_stops[0] - years_elapsed
                    flowroute_recalc_stops.pop(0)
                    self._determine_flow()
            elif always_recalc_flowroute is True:
                self._determine_flow()
            
            self._grid.at_node["topographic__elevation"] -= self._grid.at_node['ice__erosion_rate']*dt
            years_elapsed += dt
            print('dt =', dt, 'years elapsed:', years_elapsed)