from landlab import Component
from landlab import RasterModelGrid
from landlab.components import FlowAccumulator
from landlab.components import PriorityFloodFlowRouter
from landlab.utils.return_array import return_array_at_node
import numpy as np
from pathlib import Path
from collections import deque
from scipy.interpolate import RegularGridInterpolator

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
        thickness_to_width_ratio_guess=0.25,
        density_ice=920,
        grav_accel = 9.8,
        glen_exp = 3, #Not used at the moment, hardcoded, equations need updating.
        sliding_const = 10**(-19),
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
        thickness_to_width_ratio_guess:
            Assumed thickness to width ratio for the glacier [-]. Should be <0.5 to prevent gaps in the glaciers. Defaults to 0.25 (Liebl et al 2023).
        density_ice : float
            Denisty of ice [kg/m^3]. Defaults to 920 kg/m^3.
        grav_accel : float
            Gravitation acceleration [m/s^2]. Defaults to 9.8 m/s^2.
        glen_exp : float
            Glen-Nye flow law exponent. Defaults to 3.
        sliding_const : float
            Sliding velocity proportionality constant [m^2 s^-1 Pa^-3]. Defaults to 10**(-19) (Prasicek et al., 2020).
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
        self._thickness_to_width_ratio_guess = thickness_to_width_ratio_guess
        self._density_ice = density_ice
        self._grav_accel = grav_accel
        self._glen_exp = glen_exp
        self._sliding_const = sliding_const
        self._erosion_exp = erosion_exp
        self._erosion_const = erosion_const
        self._glen_const = glen_const

        self._precipitation_rate = precipitation_rate
        self._discharge_override = discharge_override
        self._equilibrium_line_altitude = equilibrium_line_altitude
        self._full_ice_altitude = full_ice_altitude
        self._nonlinear_mass_balance = nonlinear_mass_balance
        self._melt_rate_scaling_factor = melt_rate_scaling_factor
        self.determine_flow()
        self._initialize_lookup_table()

    # --- Lookup table ---
    def _initialize_lookup_table(self):
        """Obtain a lookup table for nondimensional velocities based on the distance from the center line and the ice thickness, precalculated using Nye's numerical procedure.
        Adds a model parameter self.velocity_lookup, which is a dictionary with the following keys:

          W             - 1-D array of W parameter values, shape (NW)
          U             - 3-D nondimensional velocity array, shape (NW, NY, NZ)
          Y             - 2-D Y-coordinate grid, shape (NY, NZ)
          Z             - 2-D Z-coordinate grid, shape (NY, NZ)
          boundary_type - string describing the boundary (e.g. 'ellipse')
        """
        filepath = Path(__file__).parent / 'lookup_tables' / 'elliptical_velocity.npz'
        data = np.load(filepath, allow_pickle=True)
        W = data["W"]
        U = data["U"]
        Y = data["Y"]
        Z = data["Z"]

        # --- Compute actual cell edges and widths ---
        # Horizontal (Z) cell edges
        z_midpoints = (Z[:, :-1] + Z[:, 1:]) / 2
        z_left_edge = Z[:, :1] - (Z[:, 1:2] - Z[:, :1]) / 2
        z_right_edge = Z[:, -1:] + (Z[:, -1:] - Z[:, -2:-1]) / 2
        z_edges = np.concatenate([z_left_edge, z_midpoints, z_right_edge], axis=1)

        Z_left = z_edges[:, :-1]
        Z_right = z_edges[:, 1:]
        dZ = Z_right - Z_left

        # Vertical (Y) cell edges
        y_midpoints = (Y[:-1, :] + Y[1:, :]) / 2
        y_top_edge = Y[:1, :] - (Y[1:2, :] - Y[:1, :]) / 2
        y_bottom_edge = Y[-1:, :] + (Y[-1:, :] - Y[-2:-1, :]) / 2
        y_edges = np.concatenate([y_top_edge, y_midpoints, y_bottom_edge], axis=0)

        Y_top = y_edges[:-1, :]
        Y_bottom = y_edges[1:, :]
        dY = Y_bottom - Y_top

        # For each W, derive the corresponding 1/W result:
        #   - transpose the velocity field (swap Y and Z axes)
        #   - scale by (1/W)^4
        W_inv = 1.0 / W
        # U has shape (n_W, NY, NZ); transpose the last two axes
        # and scale each slice by (1/W_i)^4
        scale = (W_inv ** 4)[:, np.newaxis, np.newaxis]   # (n_W, 1, 1)
        U_inv = np.transpose(U, axes=(0, 2, 1)) * scale
    
        # Combine: original W values + reciprocal values, sorted and deduplicated (e.g. W=1 maps to itself)
        W_all = np.concatenate([W, W_inv])
        U_all = np.concatenate([U, U_inv], axis=0)
        _, unique_indices = np.unique(W_all, return_index=True)
        W_all = W_all[unique_indices]
        U_all = U_all[unique_indices]
        
        self.velocity_lookup = {
            "W": W_all,
            "U": U_all,
            "Y": Y,
            "Z": Z,
            "Y_top": Y_top,
            "Y_bottom": Y_bottom,
            "Z_left": Z_left,
            "Z_right": Z_right,
            "dY": dY,
            "dZ": dZ,
            "boundary_type": str(data["boundary_type"][0]),
        }

        # Set arctan parameters for central flow velocity extrapolation.
        self.arctan_A = 0.190908
        self.arctan_B = -0.049878
        self.arctan_k = 0.45253

    def _central_nd_velocity(self, W):
        '''Calculate the nondimensional velocity at the center line for a given W parameter, using an arctan function fitted to the numerical results. This is used for extrapolation of the central flow velocity when the ice thickness is below the range of the lookup table.'''
        if W <= 0:
            nd_velocity_0 = 0
            print("Warning: W parameter is non-positive, setting central nondimensional velocity to 0.")
        elif W < 1:
            nd_velocity_0 = W**4*self._central_nd_velocity(1/W)
        else:
            nd_velocity_0 = self.arctan_A * np.arctan(self.arctan_k * W) + self.arctan_B
        return nd_velocity_0

    def _lookup_velocity(self, W):
        # Obtain the corresponding nondimensional velocity from the lookup table. 
        if W <= self.velocity_lookup["W"].min():
            # Extrapolate for W values below the range of the lookup table using the central nondimensional velocity derived from symmetry of the fitted arctan function and the numerical results.
            velocity_0 = self._central_nd_velocity(W)
            velocity_profile = self.velocity_lookup["U"][0] / self.velocity_lookup["U"][0].max() * velocity_0
        
        elif W >= self.velocity_lookup["W"].max():
            # Extrapolate for W values above the range of the lookup table using the central nondimensional velocity derived from the fitted arctan function and the numerical results.
            velocity_0 = self._central_nd_velocity(W)
            velocity_profile = self.velocity_lookup["U"][-1] / self.velocity_lookup["U"][-1].max() * velocity_0

        else:
            # Linearly interpolate between the two closest W values in the lookup table to obtain the velocity profile for the given W.
            index_W_higher = np.searchsorted(self.velocity_lookup["W"], W, side='right')
            index_W_lower = index_W_higher - 1
            W_distance_lower = W - self.velocity_lookup["W"][index_W_lower]
            W_distance_higher = self.velocity_lookup["W"][index_W_higher] - W
            W_distance = self.velocity_lookup["W"][index_W_higher] - self.velocity_lookup["W"][index_W_lower]
            velocity_profile = self.velocity_lookup["U"][index_W_lower] * (W_distance_higher / W_distance) + self.velocity_lookup["U"][index_W_higher] * (W_distance_lower / W_distance)

        return velocity_profile
    
    def _calculate_lookup_overlap(self, bin_distances_normalized, bin_widths_normalized, ice_thicknesses_normalized):
        #Retrieve cell edges and widths from the velocity lookup table
        cell_right = self.velocity_lookup["Z_right"]
        cell_left = self.velocity_lookup["Z_left"]
        cell_top = self.velocity_lookup["Y_top"]
        cell_bottom = self.velocity_lookup["Y_bottom"]
        cell_dZ = self.velocity_lookup["dZ"]
        cell_dY = self.velocity_lookup["dY"]

        # Reshape bin distances, widths, and ice thicknesses for broadcasting against the 2D cell grids.
        distances = bin_distances_normalized[:, np.newaxis, np.newaxis]
        bin_widths = bin_widths_normalized[:, np.newaxis, np.newaxis]
        ice_thick = ice_thicknesses_normalized[:, np.newaxis, np.newaxis]

        # --- Horizontal overlap ---
        # Fix central bin: it only covers [0, bin_width/2], not [distance - bin_width/2, distance + bin_width/2]
        is_central = (bin_widths / 2 > distances)
        bin_left = np.where(is_central, 0.0, distances - bin_widths / 2)
        bin_right = np.where(is_central, bin_widths / 2, distances + bin_widths / 2)

        cr_bl = cell_right - bin_left
        br_cl = bin_right - cell_left
        br_cr = bin_right - cell_right
        cl_bl = cell_left - bin_left

        cr_bl_above0 = cr_bl >= 0
        br_cl_above0 = br_cl >= 0
        br_cr_above0 = br_cr >= 0
        cl_bl_above0 = cl_bl >= 0

        cells_fully_within_bin = br_cr_above0 & cl_bl_above0
        cells_partially_right = br_cl_above0 & cl_bl_above0 & ~cells_fully_within_bin
        cells_partially_left = cr_bl_above0 & br_cr_above0 & ~cells_fully_within_bin
        cells_encompassing_bin = ~br_cr_above0 & ~cl_bl_above0

        actual_bin_width = bin_right - bin_left  # bin_width/2 for central, bin_width for others

        left_overlap = cells_partially_left * cr_bl / cell_dZ
        right_overlap = cells_partially_right * br_cl / cell_dZ
        center_overlap = cells_fully_within_bin + cells_encompassing_bin * actual_bin_width / cell_dZ

        horizontal_overlap = left_overlap + right_overlap + center_overlap
        horizontal_overlap = np.where(is_central, horizontal_overlap * 2, horizontal_overlap) #Double central bin, to compensate for halving the bin earlier

        # --- Vertical overlap ---
        # Ice extends from Y=0 to Y=ice_thickness (positive downward)
        bin_ice_top = 0.0
        bin_ice_bottom = ice_thick

        has_vertical_overlap = cell_top < bin_ice_bottom
        full_vertical_overlap = (cell_bottom <= bin_ice_bottom) & (cell_top >= bin_ice_top)
        partial_top = (cell_top < bin_ice_top) & (cell_bottom > bin_ice_top)
        partial_bottom = (cell_bottom > bin_ice_bottom) & (cell_top < bin_ice_bottom)

        vertical_overlap = (
            full_vertical_overlap * 1.0
            + partial_top * (cell_bottom - bin_ice_top) / cell_dY
            + partial_bottom * (bin_ice_bottom - cell_top) / cell_dY
        )

        # --- Combine and sum over bins ---
        bin_overlap = horizontal_overlap * vertical_overlap
        overlap = bin_overlap.sum(axis=0) # Fraction of each cell that is covered by the bin, summed over all bins. 
        normalized_surface_area = overlap / (len(cell_dZ[0,:]) - 1) / (len(cell_dY[:,0]) - 1) #Fraction of domain surface area covered by the bins
        
        return normalized_surface_area
    
    # --- Flow routing and swath calculation ---
    def _delta_two_nodes(self, node1, node2, elevation_field="topographic__elevation"):
        '''Calculate distances along x,y and elevation between two nodes on the modelgrid.
        node1 : int
            index of the first node
        node2 : int
            index of the second node
        elevation_field : str
            grid field name to use for elevation (default: "topographic__elevation")
        '''
        if node1 == node2:
            return 0, 0, 0, 0
        delta_x = self._grid.node_x[node1] - self._grid.node_x[node2]
        delta_y = self._grid.node_y[node1] - self._grid.node_y[node2]
        delta_z = self._grid.at_node[elevation_field][node1] - self._grid.at_node[elevation_field][node2]
        horizontal_distance = (delta_x**2 + delta_y**2)**0.5
        return horizontal_distance, delta_x, delta_y, delta_z

    def _follow_downstream(self, starting_node, receiver_array, max_distance):
        '''
        Follows nodes downstream until the max_distance is reached. Can be used to follow upstream if instead a largest donor array is used as input.

        starting_node : int 
            Node ID of the starting node
        receiver_array : numpy array
            array indicating the receiver (or largest donor) node of every node's dicharge
        max_distance : float
            Maximum radius to follow nodes to.
        '''
        current_node = starting_node
        
        while True:
            next_node = receiver_array[current_node]

            if next_node == current_node:  # No more receivers/donors
                break
            
            distance, _, _, _ = self._delta_two_nodes(starting_node, next_node)
            
            if distance > max_distance:  # Outside range
                break
            
            current_node = next_node

        final_node = current_node

        return final_node
    
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

    def _calc_swaths(self):
        self.swaths = []
        swaths_number_of_nodes = np.zeros(self._grid.number_of_nodes)

        for center_node in range(self._grid.number_of_nodes):
            if self._grid.at_node['glacier__width'][center_node]/2 > self._grid.dx:
                width_swath = self._grid.at_node['glacier__width'][center_node]
                dx = self._grid.node_x - self._grid.node_x[center_node]
                dy = self._grid.node_y - self._grid.node_y[center_node]
                distances = np.sqrt(dx**2 + dy**2)
                swath = np.where((distances > 0) & (distances < width_swath/2))[0].tolist()
                swath = [center_node] + swath

            else:
                swath = [center_node]

            self.swaths.append(swath)

            for node in swath:
                swaths_number_of_nodes[node] = max(len(swath), swaths_number_of_nodes[node])

        _ = self._grid.add_field('glacier__swath_number_of_nodes', swaths_number_of_nodes, clobber=True)

    def _calc_flow_directions(self, elevation_field="topographic__elevation"):
        """Calculate 2D flow direction components and slopes for all nodes.
        Computed from the distances between upstream and downstream edge nodes of the swath.
        Stores results as grid fields.

        elevation_field : str
            Grid field to use for slope calculation (default: "topographic__elevation").
            Pass "ice__elevation" to compute ice surface slope instead.
        """
        # Initialize arrays
        num_nodes = self._grid.number_of_nodes
        flow_x_normalized = np.zeros(num_nodes)
        flow_y_normalized = np.zeros(num_nodes)
        slope = np.zeros(num_nodes)
        receivers = self._grid.at_node["flow__receiver_node"]
        donors = self._largest_donor

        for center_node in range(num_nodes):
            width = self._grid.at_node['glacier__width'][center_node]

            upstream_node = self._follow_downstream(center_node, donors, width/2)
            downstream_node = self._follow_downstream(center_node, receivers, width/2)
            horizontal_distance, delta_x, delta_y, delta_z = self._delta_two_nodes(downstream_node, upstream_node, elevation_field=elevation_field)

            if horizontal_distance != 0:
                flow_x_normalized[center_node] = delta_x / horizontal_distance
                flow_y_normalized[center_node] = delta_y / horizontal_distance
                slope[center_node] = delta_z / horizontal_distance
            else:
                flow_x_normalized[center_node] = 0.0
                flow_y_normalized[center_node] = 0.0
                slope[center_node] = 0.0

        # Store as grid fields
        self._grid.add_field('glacier__flow_direction_x', flow_x_normalized, clobber=True)
        self._grid.add_field('glacier__flow_direction_y', flow_y_normalized, clobber=True)
        self._grid.add_field('glacier__slope', slope, clobber=True)

    def determine_flow(self, use_ice_elevation=True):
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

        #TODO: Remove the temporary surface_water__discharge field to avoid confusion
        #TODO: Run flowrouter again with precipitation of water? 

        # Calculate glacier width
        glacier_width = self._width_scaling_const*(glacier_discharge*self._grid.dx)**self._width_scaling_exp
        _ = self._grid.add_field('glacier__width', glacier_width, clobber=True)


        self._build_donor_dict()  # Pre-compute donor relationships
        self._obtain_largest_donors() #Pre-compute largest donors for every node
        elevation_field = "topographic__elevation"
        if use_ice_elevation and 'ice__elevation' in self._grid.at_node:
            elevation_field = 'ice__elevation'
        self._calc_flow_directions(elevation_field=elevation_field) #Pre-compute flow directions and slope
        self._calc_swaths()
        self._upstream_node_order = self._grid.at_node['flow__upstream_node_order'].copy()
    
    def _swath_characteristics(self, center_node):
        
        #Calculate distances of the swath
        swath = self.swaths[center_node]
        swath_array = np.array(swath)
        dx = self._grid.node_x[swath_array] - self._grid.node_x[center_node]
        dy = self._grid.node_y[swath_array] - self._grid.node_y[center_node]
        swath_distances = (dx**2 + dy**2)**0.5

        #Calculate distances along and perpendicular to flow
        flow_x = self._grid.at_node['glacier__flow_direction_x'][center_node]
        flow_y = self._grid.at_node['glacier__flow_direction_y'][center_node]
        slope = self._grid.at_node['glacier__slope'][center_node]
        swath_distances_along = flow_x*dx + flow_y*dy
        swath_distances_perp = flow_y*dx - flow_x*dy
        
        #Obtain topography and correct for slope to get elevations relative to the center node
        swath_topography_elevations = self._grid.at_node["topographic__elevation"][swath_array]
        reference_surface = slope*swath_distances_along + self._grid.at_node["topographic__elevation"][center_node]
        swath_relative_elevations = swath_topography_elevations - reference_surface

        return swath_distances, swath_distances_along, swath_distances_perp, swath_relative_elevations

    # --- Ice calculation ---
    def _bin_cross_section(self, center_node, distances_along, distances_perp, topography_elevations):
        # --- Binning by perpendicular distance ---
        # To determine glacier cross section, we use inverse along-distance weighting to calculate average topographic elevations in the swath.
        width = self._grid.at_node['glacier__width'][center_node]

        # First find unique perpendicular distances, used as natural bins (grid-aligned)
        unique_distances_perp = np.unique(distances_perp)
        unique_distances_perp = unique_distances_perp[np.abs(unique_distances_perp) <= width / 2] # Filter out distances beyond glacier width
        unique_distances_perp = np.sort(unique_distances_perp)

        # Initialize arrays for binned values
        bin_distances = []
        bin_elevations_topo = []
        bin_widths = []
    
        for dist in unique_distances_perp:
            at_this_distance = (distances_perp == dist)
            distance_along_bin = distances_along[at_this_distance]
            topo_bin = topography_elevations[at_this_distance]
            
            # Inverse-distance weighting based on along-flow distance
            epsilon = 1e-6
            weights = 1.0 / (np.abs(distance_along_bin) + epsilon)
            weights /= weights.sum()  # Normalize weights
            
            # Weighted average elevation
            avg_topo = np.sum(weights * topo_bin)
            
            # Store bin data
            bin_distances.append(dist)
            bin_elevations_topo.append(avg_topo)

        # Convert to numpy arrays (in order from -width/2 to +width/2)
        bin_distances = np.array(bin_distances)
        bin_elevations_topo = np.array(bin_elevations_topo)

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
        
        return bin_distances, bin_widths, bin_elevations_topo
    
    def _discharge_from_thickness(self, center_thickness, bin_distances, bin_widths, bin_elevations_topo, slope):
        corrected_thickness = center_thickness * _cosarctan(slope)
        
        if center_thickness <= 0:
            return 0, self._lookup_velocity(1)*0, 0, 0

        # Deformation velocity
        # Calculate ice thicknesses for each bin
        ice_thicknesses = np.maximum(0, center_thickness - bin_elevations_topo)

        # Find largest distance with non-zero ice thickness
        nonzero_thickness = np.where(ice_thicknesses > 0)
        if nonzero_thickness[0].size == 0:
            return 0, self._lookup_velocity(1)*0, 0, 0
        largest_distance = bin_distances[nonzero_thickness].max() + bin_widths[nonzero_thickness][bin_distances[nonzero_thickness].argmax()]/2

        # Determine the W parameter (halfwidth to thickness ratio) and lookup velocity profile
        W = largest_distance / corrected_thickness
        nondimensional_velocity_profile = self._lookup_velocity(W)
        
        # Normalize bin distances, widths and ice thicknesses
        bin_distances_normalized = bin_distances / largest_distance
        bin_widths_normalized = bin_widths / largest_distance
        ice_thicknesses_normalized = ice_thicknesses / center_thickness

        normalized_crosssectional_area = self._calculate_lookup_overlap(bin_distances_normalized, bin_widths_normalized, ice_thicknesses_normalized)
        crosssectional_area = normalized_crosssectional_area * largest_distance * center_thickness
        fs = (self._density_ice*self._grav_accel)**self._glen_exp * self._sliding_const
        sliding_velocity = fs * center_thickness ** (self._glen_exp - 1) * abs(slope)**self._glen_exp
        
        k = self._density_ice * self._grav_accel * corrected_thickness * _sinarctan(abs(slope))
        velocity_profile = corrected_thickness * self._glen_const * k**self._glen_exp * nondimensional_velocity_profile

        deformation_discharge = velocity_profile * crosssectional_area
        sliding_discharge = sliding_velocity * crosssectional_area
        discharge = sliding_discharge + deformation_discharge
        discharge_sum = discharge.sum()

        return discharge_sum, velocity_profile, largest_distance, sliding_velocity

    def _discharge_iteration(self, center_node, bin_distances, bin_widths, bin_elevations_topo, initial_thickness_guess=None):
        '''Find thickness and velocity profile using regula falsi. This is done by iteratively adjusting the center thickness and calculating the resulting discharge until it matches the target discharge within a certain tolerance.'''
        target_discharge = self._grid.at_node['glacier__discharge'][center_node]/SECPERYEAR
        width = self._grid.at_node['glacier__width'][center_node]
        slope = self._grid.at_node['glacier__slope'][center_node]

        #Initialize values
        if initial_thickness_guess is not None:
            center_thickness = initial_thickness_guess
        else:
            center_thickness = width*self._thickness_to_width_ratio_guess
        thickness_min = 0
        discharge_min = 0
        thickness_max = np.inf
        discharge_max = np.inf
        predicted_discharge = 0
        velocity_profile = self._lookup_velocity(1)*0
        sliding_velocity = 0

        if target_discharge == 0:
            return 0, 0, velocity_profile, 0, sliding_velocity
        
        # Iteration tracking for debugging
        debug_mode = False
        MAX_NORMAL_ITERS = 10
        iteration = 0
        thickness_history = []
        discharge_history = []
        verbose_mode = False

        while abs((predicted_discharge - target_discharge)/target_discharge) > 1e-2:
            converged_thickness = center_thickness
            predicted_discharge, velocity_profile, largest_distance, sliding_velocity = self._discharge_from_thickness(center_thickness, bin_distances, bin_widths, bin_elevations_topo, slope)

            if verbose_mode:
                print(f"  iter {iteration}: thickness={converged_thickness:.8f}, discharge={predicted_discharge:.8f}")
            else:
                thickness_history.append(converged_thickness)
                discharge_history.append(predicted_discharge)

            if predicted_discharge < target_discharge:
                thickness_min = center_thickness
                discharge_min = predicted_discharge
            else:
                thickness_max = center_thickness
                discharge_max = predicted_discharge

            prev_thickness = center_thickness

            if iteration > MAX_NORMAL_ITERS and thickness_max - thickness_min < 1e-2:
                #Prevent infinite loop if thickness converges but discharge does not (due to numerical issues in the lookup table extrapolation). 
                #In that case, we take the converged thickness as the best estimate and move on, even if the discharge is not within the target tolerance.
                if verbose_mode:
                    print(f"Thickness converged, without discharge convergence after {iteration} iterations: thickness={center_thickness:.8f}, discharge={predicted_discharge:.8f} (target={target_discharge:.8f})")
                break

            #Regula falsi doesn't work if the lower bound is zero or the upper bound is infinite, so we use simple steps until we have a valid range, then switch to regula falsi.
            if np.isinf(thickness_max):
                center_thickness *= 1.5
            elif thickness_min == 0:
                center_thickness /= 1.5
            else:
                center_thickness = (thickness_min*(discharge_max - target_discharge) - thickness_max*(discharge_min - target_discharge)) / (discharge_max - discharge_min)

            if center_thickness == prev_thickness:
                print(f"Thickness stuck at {center_thickness:.8f} on iter {iteration} (node {center_node}, target={target_discharge:.8f} m^3/s, predicted={predicted_discharge:.8f})")
                #input("Press Enter to continue...")

            iteration += 1
            if not verbose_mode and iteration >= MAX_NORMAL_ITERS and debug_mode:
                verbose_mode = True
                print(f"Discharge iteration for node {center_node} taking long (target={target_discharge:.8f} m^3/s).")
                print(f"Thickness history so far:  {[f'{t:.8f}' for t in thickness_history]}")
                print(f"Discharge history so far:  {[f'{d:.8f}' for d in discharge_history]}")

        if verbose_mode:
            print(f"Converged after {iteration} iterations: thickness={converged_thickness:.8f}, discharge={predicted_discharge:.8f} (target={target_discharge:.8f})")
            #input("Press Enter to continue...")

        return converged_thickness, largest_distance, velocity_profile, predicted_discharge, sliding_velocity
    
    def _ice_in_swath(self, center_thickness, largest_distance, velocity_profile, sliding_velocity, swath_distances, swath_distances_along, swath_relative_elevations):
        # Calculate ice thicknesses using center thickness.
        # swath_relative_elevations is already defined relative to the tilted reference surface
        # (center_topo + slope*d_along), so the ice surface height above that reference is simply
        # center_thickness — constant in the along-flow direction and varying only across the cross-section.
        # Adding slope*d_along here would double-count the bed tilt, inflating thickness upstream.
        ice_elevations = np.full_like(swath_distances_along, center_thickness)
        ice_thicknesses = np.maximum(0, ice_elevations - swath_relative_elevations)

        # Interpolate velocity profile using scipy regularGridInterpolator
        normalized_thicknesses = ice_thicknesses/center_thickness 
        normalized_distances = swath_distances/largest_distance 

        y_coords = self.velocity_lookup["Y"][:, 0]  # 1D Y axis (normalized vertical, shape NY)
        z_coords = self.velocity_lookup["Z"][0, :]  # 1D Z axis (normalized horizontal, shape NZ)
        interpolator = RegularGridInterpolator(
            (y_coords, z_coords),
            velocity_profile,
            method='linear',
            bounds_error=False,
            fill_value=0.0,
        )
        has_ice = ice_thicknesses > 0
        # Sliding is only added where the node is within the reference cross-section (normalized distance <= 1).
        # Beyond that, the deformation velocity is zero by construction (fill_value=0.0), and applying
        # the full center sliding velocity there would give disproportionate velocities for thin rim ice.
        within_cross_section = normalized_distances <= 1.0

        query_points = np.column_stack([normalized_thicknesses, normalized_distances])
        swath_basal_velocities = interpolator(query_points)
        swath_basal_velocities[has_ice & within_cross_section] += sliding_velocity
        swath_basal_velocities[~has_ice] = 0.0  # No ice → no velocity

        surface_query_points = np.column_stack([np.zeros(len(normalized_distances)), normalized_distances])
        swath_surface_velocities = interpolator(surface_query_points)
        swath_surface_velocities[has_ice & within_cross_section] += sliding_velocity
        swath_surface_velocities[~has_ice] = 0.0  # No ice → no velocity

        return ice_thicknesses, swath_basal_velocities*SECPERYEAR, swath_surface_velocities*SECPERYEAR

    def calc_ice(self):
        topography = self._grid.at_node["topographic__elevation"]
        _ = self._grid.add_field('ice__thickness', np.zeros(self._grid.number_of_nodes), clobber=True)
        _ = self._grid.add_field('ice__basal_velocity', np.zeros(self._grid.number_of_nodes), clobber=True)
        _ = self._grid.add_field('ice__surface_velocity', np.zeros(self._grid.number_of_nodes), clobber=True)
        _ = self._grid.add_field('glacier__cross_section', np.zeros(self._grid.number_of_nodes), clobber=True)
        _ = self._grid.add_field('glacier__center_node', np.zeros(self._grid.number_of_nodes), clobber=True)
        #_ = self._grid.add_field('glacier__width', np.zeros(self._grid.number_of_nodes), clobber=True) # rename

        # Initialize thickness guesses for this timestep using the default ratio
        glacier_width = self._grid.at_node['glacier__width']
        thickness_guess = glacier_width * self._thickness_to_width_ratio_guess

        for center_node in reversed(self._upstream_node_order):
            if self._grid.at_node['glacier__discharge'][center_node] <= 0 or self._grid.at_node['glacier__slope'][center_node] >= -0.001:
                continue  # Skip nodes with no discharge or very low slope (or uphill), as they will have little to no ice flow and therefore won't contribute to erosion. This also prevents numerical issues in the discharge iteration for these nodes.

            # Calculate swath characteristics
            swath_distances, swath_distances_along, swath_distances_perp, swath_relative_elevations = self._swath_characteristics(center_node)

            # Bin the swath to get cross-sectional topography
            bin_distances, bin_widths, bin_elevations_topo = self._bin_cross_section(center_node, swath_distances_along, swath_distances_perp, swath_relative_elevations)

            # Use the binned cross section to iteratively solve for the center ice thickness that matches the target discharge, and obtain the velocity profile across the swath.
            center_thickness, largest_distance, velocity_profile, predicted_discharge, sliding_velocity = self._discharge_iteration(center_node, bin_distances, bin_widths, bin_elevations_topo, initial_thickness_guess=thickness_guess[center_node])

            # Pass converged thickness as initial guess to the receiver node
            receiver = self._grid.at_node['flow__receiver_node'][center_node]
            if receiver != center_node:
                thickness_guess[receiver] = center_thickness

            # Calculate ice thicknesses and velocities across the swath
            ice_thicknesses, swath_basal_velocities, swath_surface_velocities = self._ice_in_swath(center_thickness, largest_distance, velocity_profile, sliding_velocity, swath_distances, swath_distances_along, swath_relative_elevations)

            # Update grid fields for swath nodes where basal velocity exceeds current value
            swath_nodes = np.array(self.swaths[center_node])
            update_mask = swath_basal_velocities > self._grid.at_node['ice__basal_velocity'][swath_nodes]
            self._grid.at_node['ice__thickness'][swath_nodes[update_mask]] = ice_thicknesses[update_mask]
            self._grid.at_node['ice__basal_velocity'][swath_nodes[update_mask]] = swath_basal_velocities[update_mask]
            self._grid.at_node['ice__surface_velocity'][swath_nodes[update_mask]] = swath_surface_velocities[update_mask]


        erosion_rate = self._erosion_const * (self._grid.at_node['ice__basal_velocity'])**self._erosion_exp

        _ = self._grid.add_field('ice__elevation', topography + self._grid.at_node['ice__thickness'], clobber=True)
        _ = self._grid.add_field('ice__erosion_rate', erosion_rate, clobber=True)
    
    # --- Erosion ---
    def run_one_step(self, dt, stabilise_erosion=False):
        self.calc_ice()

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
            self.calc_ice()
            highest_erosion_rate = np.max(self._grid.at_node['ice__erosion_rate'])


            dt = max_erosion/highest_erosion_rate

            if years_elapsed + dt >= number_of_years:
                dt = number_of_years - years_elapsed
            elif flowroute_recalc_stops:
                if years_elapsed + dt >= flowroute_recalc_stops[0]:
                    dt = flowroute_recalc_stops[0] - years_elapsed
                    flowroute_recalc_stops.pop(0)
                    self.determine_flow()
            elif always_recalc_flowroute is True:
                self.determine_flow()
            
            self._grid.at_node["topographic__elevation"] -= self._grid.at_node['ice__erosion_rate']*dt
            years_elapsed += dt
            print('dt =', dt, 'years elapsed:', years_elapsed)