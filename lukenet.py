
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
from tqdm.auto import tqdm
import numba
import lukenet_input as inputs
import lukenet_helpers as helpers
from scipy.sparse import lil_matrix
import logging
import os


#################
# BASIC CLASSES #
#################

class Elements:
    def __init__(self):
        self.name = []

class Species:
    def __init__(self):
        self.name = []
        self.abundance = []
        self.number = []
        self.mass = []
        self.charge = []

class Gas:
    def __init__(self):
        self.n_gas = 1
        self.temperature = 1
        self.h2_col = 1

class Dust:
    def __init__(self):
        self.n_dust = 1           
        self.temperature = 1
        self.radius = 1e-5         
        self.rho = 2.5              
        self.mass = 1                
        self.binding_sites = 1e6

class Conditions:
    def __init__(self):
        self.gtd = 1
        self.Av = 1
        self.G_0 = 1
        self.G_0_unatt = 1
        self.Zeta_X = 1
        self.Zeta_CR = 1
        self.pah_ism = 1
        self.dg100 = 1
        
class Reactions:
    def __init__(self):
        self.educts = []
        self.products = []
        self.reaction_id = []
        self.itype = []
        self.a = []
        self.b = []
        self.c = []
        self.temp_min = []
        self.temp_max = []
        self.pd_data = []
        self.k = []
        self.labels = []

class Parameters:
    def __init__(self):
        self.n_elements = 0
        self.n_species = 0
        self.n_reactions = 0
        self.time_initial = 1.0
        self.time_final = 1.0
        self.delta_v = 0.2
        self.av_nH = (1/5.34e-22)
        self.column = True
        self.k_B = 1.3806504e-16
        self.yr_sec = 3.1556926e7
        self.m_p = 1.660538e-24
        self.G = 6.6743e-8



#############################
# FUNCTIONS FOR INTEGRATION #
#############################

@numba.jit(nopython=True, cache=True)
def calculate_derivatives(y, k, idx, ydot, nr):
    """
    Calculates the derivatives and reaction rates for a chemical network.

    This function computes the formation and destruction rates of chemical 
    species based on a set of reaction rate coefficients, reactant-product 
    relationships, and species concentrations. It uses the Numba JIT compiler 
    for performance optimization.

    Args:
        y (numpy.ndarray): Array of concentrations of chemical species.
        k (numpy.ndarray): Array of reaction rate coefficients.
        idx (numpy.ndarray): Array of indices defining reactants (educts) and 
            products for each reaction. The format is:
                idx[i,0:3] - indices of up to 3 reactants (educts).
                idx[i,3:8] - indices of up to 5 products.
        ydot (numpy.ndarray): Placeholder array for derivatives of concentrations.
        nr (int): Total number of reactions.

    Returns:
        tuple:
            numpy.ndarray: Array of net changes (formation - destruction) for 
                each species.
            numpy.ndarray: Array of reaction rates for each reaction.

    Notes:
        - The input `idx` is expected to have negative values for missing reactants 
          or products.
        - The function assumes that `y` and `k` are appropriately sized to 
          correspond to the number of reactions and species.
    """
    formation = np.zeros_like(ydot)
    destruction = np.zeros_like(ydot)
    rates = np.zeros(nr)
    
    # Loop over all reactions
    for i in range(nr):
        # Get indices for educts and products
        ir1 = idx[i,0]
        ir2 = idx[i,1]
        ir3 = idx[i,2]
        ip1 = idx[i,3]
        ip2 = idx[i,4]
        ip3 = idx[i,5]
        ip4 = idx[i,6]
        ip5 = idx[i,7]
        
        # Calculate reaction term
        term = k[i]
        
        # Multiply by concentrations of educts
        if ir1 >= 0:
            term *= y[ir1]
        if ir2 >= 0:
            term *= y[ir2]
        if ir3 >= 0:
            term *= y[ir3]
        
        # Save for the rate history 
        rates[i] = term
            
        # Add to formation rates for products
        if ip1 >= 0:
            formation[ip1] += term
        if ip2 >= 0:
            formation[ip2] += term
        if ip3 >= 0:
            formation[ip3] += term
        if ip4 >= 0:
            formation[ip4] += term
        if ip5 >= 0:
            formation[ip5] += term
            
        # Add to destruction rates for educts
        if ir1 >= 0:
            destruction[ir1] += term
        if ir2 >= 0:
            destruction[ir2] += term
        if ir3 >= 0:
            destruction[ir3] += term

    return (formation - destruction), rates


@numba.jit(nopython=True, cache=True)
def calculate_jacobian_dense(y, k, idx, nr):
    """
    Computes the dense Jacobian matrix for a chemical reaction network.

    This function calculates the Jacobian matrix, representing the partial derivatives 
    of reaction rates with respect to species concentrations. It accounts for 
    single-reactant and two-reactant reactions, optimizing the computation using 
    Numba's JIT compilation.

    Args:
        y (numpy.ndarray): Array of concentrations of chemical species.
        k (numpy.ndarray): Array of reaction rate coefficients.
        idx (numpy.ndarray): Array of indices defining reactants (educts) and 
            products for each reaction. The format is:
                idx[i, 0:3] - indices of up to 3 reactants (educts).
                idx[i, 3:8] - indices of up to 5 products.
        nr (int): Total number of reactions.

    Returns:
        numpy.ndarray: Dense Jacobian matrix of shape `(ns, ns)`, where `ns` 
            is the number of chemical species. Each entry `jac[j, i]` represents 
            the partial derivative of the rate of change of species `j` with 
            respect to species `i`.

    Notes:
        - Single-reactant reactions only affect the row and column of that reactant.
        - Two-reactant reactions account for the interaction between two species 
          and their contributions to products.
        - Unused indices in `idx` are expected to have a value of `-1`.
    """
    ns = len(y)
    jac = np.zeros((ns, ns))  # Use a dense NumPy array for calculation
    
    for i in range(nr):
        ir1 = idx[i, 0]
        ir2 = idx[i, 1]
        ir3 = idx[i, 2]
        ip1 = idx[i, 3]
        ip2 = idx[i, 4]
        ip3 = idx[i, 5]
        ip4 = idx[i, 6]
        ip5 = idx[i, 7]

        # Single reactant reactions
        if ir2 == -1 and ir3 == -1:
            if ir1 >= 0:
                # Effect on reactant
                jac[ir1, ir1] -= k[i]
                # Effect on products
                if ip1 >= 0:
                    jac[ip1, ir1] += k[i]
                if ip2 >= 0:
                    jac[ip2, ir1] += k[i]
                if ip3 >= 0:
                    jac[ip3, ir1] += k[i]
                if ip4 >= 0:
                    jac[ip4, ir1] += k[i]
                if ip5 >= 0:
                    jac[ip5, ir1] += k[i]

        # Two reactant reactions
        elif ir3 == -1:
            if ir1 >= 0 and ir2 >= 0:
                # Effect on first reactant
                term1 = k[i] * y[ir2]
                jac[ir1, ir1] -= term1
                jac[ir2, ir1] -= term1
                if ip1 >= 0:
                    jac[ip1, ir1] += term1
                if ip2 >= 0:
                    jac[ip2, ir1] += term1
                if ip3 >= 0:
                    jac[ip3, ir1] += term1
                if ip4 >= 0:
                    jac[ip4, ir1] += term1
                if ip5 >= 0:
                    jac[ip5, ir1] += term1

                # Effect on second reactant
                term2 = k[i] * y[ir1]
                jac[ir1, ir2] -= term2
                jac[ir2, ir2] -= term2
                if ip1 >= 0:
                    jac[ip1, ir2] += term2
                if ip2 >= 0:
                    jac[ip2, ir2] += term2
                if ip3 >= 0:
                    jac[ip3, ir2] += term2
                if ip4 >= 0:
                    jac[ip4, ir2] += term2
                if ip5 >= 0:
                    jac[ip5, ir2] += term2

    return jac

    ns = len(y)
    jac = np.zeros((ns, ns))  # Use a dense NumPy array for calculation
    
    for i in range(nr):
        ir1 = idx[i, 0]
        ir2 = idx[i, 1]
        ir3 = idx[i, 2]
        ip1 = idx[i, 3]
        ip2 = idx[i, 4]
        ip3 = idx[i, 5]
        ip4 = idx[i, 6]
        ip5 = idx[i, 7]

        # Single reactant reactions
        if ir2 == -1 and ir3 == -1:
            if ir1 >= 0:
                # Effect on reactant
                jac[ir1, ir1] -= k[i]
                # Effect on products
                if ip1 >= 0:
                    jac[ip1, ir1] += k[i]
                if ip2 >= 0:
                    jac[ip2, ir1] += k[i]
                if ip3 >= 0:
                    jac[ip3, ir1] += k[i]
                if ip4 >= 0:
                    jac[ip4, ir1] += k[i]
                if ip5 >= 0:
                    jac[ip5, ir1] += k[i]

        # Two reactant reactions
        elif ir3 == -1:
            if ir1 >= 0 and ir2 >= 0:
                # Effect on first reactant
                term1 = k[i] * y[ir2]
                jac[ir1, ir1] -= term1
                jac[ir2, ir1] -= term1
                if ip1 >= 0:
                    jac[ip1, ir1] += term1
                if ip2 >= 0:
                    jac[ip2, ir1] += term1
                if ip3 >= 0:
                    jac[ip3, ir1] += term1
                if ip4 >= 0:
                    jac[ip4, ir1] += term1
                if ip5 >= 0:
                    jac[ip5, ir1] += term1

                # Effect on second reactant
                term2 = k[i] * y[ir1]
                jac[ir1, ir2] -= term2
                jac[ir2, ir2] -= term2
                if ip1 >= 0:
                    jac[ip1, ir2] += term2
                if ip2 >= 0:
                    jac[ip2, ir2] += term2
                if ip3 >= 0:
                    jac[ip3, ir2] += term2
                if ip4 >= 0:
                    jac[ip4, ir2] += term2
                if ip5 >= 0:
                    jac[ip5, ir2] += term2

    return jac


def calculate_jacobian(y, k, idx, nr):
    """
    Computes the Jacobian matrix in sparse format.

    This function calculates the dense Jacobian matrix using `calculate_jacobian_dense` 
    and converts it to a sparse matrix in LIL (List of Lists) format for memory efficiency.

    Args:
        y (numpy.ndarray): Array of concentrations of chemical species.
        k (numpy.ndarray): Array of reaction rate coefficients.
        idx (numpy.ndarray): Array of indices defining reactants (educts) and products.
        nr (int): Total number of reactions.

    Returns:
        scipy.sparse.lil_matrix: Sparse Jacobian matrix in LIL format.

    See Also:
        `calculate_jacobian_dense`: Detailed computation of the dense Jacobian matrix.
    """
    jac_dense = calculate_jacobian_dense(y, k, idx, nr)
    jac_sparse = lil_matrix(jac_dense)  # Convert dense matrix to sparse (LIL format)
    return jac_sparse



###########
# LOGGING #
###########

# Ignore! Not currently properly implemented!
def setup_logging(log_dir='logs', log_level=logging.INFO):
    """
    Configure comprehensive logging for Lukenet
    
    Args:
        log_dir (str): Directory to store log files
        log_level (int): Logging level (default: INFO)
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure file logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'lukenet.log')),
            logging.StreamHandler()  # Also log to console
        ]
    )

    return logging.getLogger('Lukenet')


##############
# MAIN CLASS #
##############

class Lukenet:
    def __init__(self):
        self.elements = Elements()
        self.species = Species()
        self.gas = Gas()
        self.dust = Dust()
        self.conditions = Conditions()
        self.reactions = Reactions()
        self.parameters = Parameters()
        self.abundance_history = []  
        self._progress_bar = None
        self.logger = setup_logging()

    
    ##############
    # Initialise #
    ##############
    
    def init_lukenet(self):
    
        # Input parameters
        self.gas.n_gas = inputs.n_gas
        self.gas.t_gas = inputs.t_gas
        self.dust.n_dust = inputs.n_dust
        self.dust.t_dust = inputs.t_dust
        self.conditions.gtd = inputs.gtd
        self.conditions.Av = inputs.Av
        self.conditions.G_0 = inputs.G_0
        self.conditions.G_0_unatt = inputs.G_0_unatt
        self.conditions.Zeta_X = inputs.Zeta_X
        self.conditions.Zeta_CR = inputs.Zeta_CR
        self.conditions.pah_ism = inputs.pah_ism
        self.conditions.dg100 = ((1/self.conditions.gtd) * 100) / 1.4          # factor 1.4 accounts for He
        self.parameters.time_final = inputs.t_chem * self.parameters.yr_sec
        if self.parameters.column:
            self.gas.h2_col = inputs.h2_col
            
        # Chemical network
        network_data = helpers.read_chemnet(inputs.network)
        self.parameters.n_elements = network_data[0]
        self.elements.name = network_data[1]
        self.parameters.n_species = network_data[2]
        self.species.name = network_data[3]
        self.species.abundance = network_data[4]
        self.species.mass = network_data[5]
        self.species.charge = network_data[6]
        self.parameters.n_reactions = network_data[7]
        self.reactions.educts = network_data[8]
        self.reactions.products = network_data[9]
        self.reactions.reaction_id = network_data[10]
        self.reactions.itype = network_data[11]
        self.reactions.a = network_data[12]
        self.reactions.b = network_data[13]
        self.reactions.c = network_data[14]
        self.reactions.temp_min = network_data[15]
        self.reactions.temp_max = network_data[16]
        self.reactions.pd_data = network_data[17]
        self.reactions.k = np.zeros(self.parameters.n_reactions)
        self.reactions.labels = helpers.create_reaction_labels(self.parameters.n_reactions, self.reactions.educts, self.reactions.products)
        
        # PAH abundance
        i_PAH0 = self.species.name.index('PAH0')
        i_PAHp = self.species.name.index('PAH+')
        i_PAHm = self.species.name.index('PAH-')
        i_PAHH = self.species.name.index('PAH_H')
        if self.conditions.G_0 > 1e6:
            self.species.abundance[i_PAH0] = 1e-10
            self.species.abundance[i_PAHp] = 0.0
            self.species.abundance[i_PAHm] = 0.0
            self.species.abundance[i_PAHH] = 0.0
        else:
            self.species.abundance[i_PAH0] = 4.17e-07 * self.conditions.pah_ism
            self.species.abundance[i_PAHp] = 0.0
            self.species.abundance[i_PAHm] = 0.0
            self.species.abundance[i_PAHH] = 0.0
        
        # Abundances
        self.species.number = self.species.abundance * self.gas.n_gas
        
        # Self-shielding factors
        self.ss_co = helpers.read_selfshielding_co('data_coselfs_iso.dat')
        self.ss_n2 = helpers.read_selfshielding_n2('data_n2selfs_3d.dat')
        
        # Setup reaction indices
        self.setup_reaction_indices()
        

    ##########################
    # Setup reaction indices #
    ##########################
    
    def setup_reaction_indices(self):
        """Create index arrays for reactions to speed up calculations"""
        self.idx = np.zeros((self.parameters.n_reactions, 8), dtype=np.int32)
        
        for i in range(self.parameters.n_reactions):
            # Convert educts and products to indices in species array
            for j, educt in enumerate(self.reactions.educts[i]):
                if educt != '' and educt != 'M' and educt != 'PHOTON' and educt != 'CRP' and educt != 'CRPHOT' and educt != 'XELECTRON':
                    self.idx[i,j] = self.species.name.index(educt)
                else:
                    self.idx[i,j] = -1
                    
            for j, product in enumerate(self.reactions.products[i]):
                if product != '' and product != 'M' and product != 'PHOTON' and product != 'CRP' and product != 'CRPHOT' and product != 'XELECTRON':
                    self.idx[i,j+3] = self.species.name.index(product)
                else:
                    self.idx[i,j+3] = -1

    
    #############################
    # Compute rate coefficients #
    #############################
    
    def compute_rate_coefficients(self, y):
        
        self.logger.debug(f"Computing rate coefficients at temperature {self.gas.t_gas}")
                        
        # Index of some key species
        i_H   = self.species.name.index('H')
        i_H2  = self.species.name.index('H2')
        i_He  = self.species.name.index('He')
        i_CO  = self.species.name.index('CO')
        i_H2O = self.species.name.index('H2O')
        i_C   = self.species.name.index('C')
        i_N2  = self.species.name.index('N2')
        
        # Initialise H2 dissociation rate
        self.disso_H2 = 0.0
        
        # Dust properties
        ngrndust100 = 1e-12                                                    # gtd number density for gtd=100
        n_gr        = self.gas.n_gas * ngrndust100 * self.conditions.dg100     # n_grains used for chemistry (not self.dust.ndust!!)
        
        # n_ice and n_hydro
        n_hydro = 0.0
        n_ice   = 0.0
        for i in range(0, self.parameters.n_reactions):
            ir1, ir2, ir3, ip1, ip2, ip3, ip4, ip5 = self.idx[i,:]
            if self.reactions.itype[i] == 11:
                n_hydro += y[ir1]
            if self.reactions.itype[i] == 80:
                n_ice += y[ir1]

        # Loop through all reactions
        for i in range(0, self.parameters.n_reactions):
            ir1, ir2, ir3, ip1, ip2, ip3, ip4, ip5 = self.idx[i,:]             # indices for this reaction

            # 10: H2 formation on grains (Bosman)
            if self.reactions.itype[i] == 10:
                if self.dust.t_dust < 10.0:
                    eta   = 1.0
                    stick = 1.0/(1.0 + 0.04 * np.sqrt(self.gas.t_gas + self.dust.t_dust) + 2e-3 * self.gas.t_gas + 8e-6 * self.gas.t_gas**2)
                else:
                    # Calculate mean velocity for monolayers per second
                    v_mean_h2  = np.sqrt(8.0 * self.parameters.k_B * self.gas.t_gas / (np.pi * self.parameters.m_p))
                    stick      = 1.0/(1.0 + 0.04 * np.sqrt(self.gas.t_gas + self.dust.t_dust) + 2e-3 * self.gas.t_gas + 8e-6 * self.gas.t_gas**2)
                    f_mlps     = v_mean_h2 * y[i_H] * np.pi * self.dust.radius**2 / self.dust.binding_sites * stick                    
                    f_mlps     = np.maximum(f_mlps, 1e-30)
                    sqterm     = (1.0 + np.sqrt((30000.0-200.0)/(600.0-200.0)))**2
                    beta_H2    = 3e12 * np.exp(-320.0/self.dust.t_dust)
                    beta_alpha = 0.25 * sqterm * np.exp(-200.0/self.dust.t_dust)
                    xi         = 1.0/(1.0 + 1.3e13/(2.0 * f_mlps) * np.exp(-1.5 * 30000.0/self.dust.t_dust) * sqterm)
                    eta        = xi/(1.0 + 0.005*f_mlps/(2.0*beta_H2) + beta_alpha)
    
                s_eta = stick * eta
                k = s_eta * self.reactions.a[i] * (self.gas.t_gas**self.reactions.b[i]) / (1e-10 + y[i_H]) * self.gas.n_gas * self.conditions.dg100
            
            # 11: Hydrogenation
            elif self.reactions.itype[i] == 11:
                if self.gas.n_gas * self.conditions.dg100 < 1e3:
                    prehydro = 1e-99
                else:
                    prehydro = np.pi * self.dust.radius**2 * n_gr / np.maximum(n_hydro, self.dust.binding_sites*n_gr)
                k = prehydro * np.sqrt(8.0*self.parameters.k_B/(np.pi*self.parameters.m_p))*np.sqrt(self.gas.t_gas)
                  
            # 12: Photodesorption
            elif self.reactions.itype[i] == 12:
                if self.gas.n_gas * self.conditions.dg100 < 1e3:
                    pregrain = 1e-99
                else:
                    pregrain = np.pi * self.dust.radius**2 * n_gr / np.maximum(n_ice, self.dust.binding_sites*n_gr)
                k = pregrain * self.reactions.a[i] * 1e8 * (self.conditions.G_0+1e-4*((self.conditions.Zeta_CR+self.conditions.Zeta_X)/5e-17))

            # 20: Normal gas-phase reaction        
            elif self.reactions.itype[i] == 20:
                k = self.reactions.a[i] * (self.gas.t_gas/300.0)**self.reactions.b[i] * np.exp(-self.reactions.c[i]/self.gas.t_gas)
                # Not the highest temperature reaction available
                if self.gas.t_gas >= np.abs(self.reactions.temp_max[i]) and self.reactions.temp_max[i] < 0.0:
                   k = 0.0
                # Keep constant above maximum temperature
                if self.gas.t_gas > np.abs(self.reactions.temp_max[i]) and self.reactions.temp_max[i] > 0.0:
                    k = self.reactions.a[i] * (self.reactions.temp_max[i] / 300.0) ** self.reactions.b[i] * np.exp(-self.reactions.c[i] / self.reactions.temp_max[i])
                # Switch off below minimum temperature
                if self.gas.t_gas < self.reactions.temp_min[i]:
                    k = 0.0
                    
            # 21: Normal gas-phase reaction (do not extrapolate in temperature)
            elif self.reactions.itype[i] == 21:
                k = self.reactions.a[i] * (self.gas.t_gas/300.0)**self.reactions.b[i] * np.exp(-self.reactions.c[i]/self.gas.t_gas)
                if self.gas.t_gas > self.reactions.temp_max[i]:
                    k = self.reactions.a[i] * (self.reactions.temp_max[i]/300.0)**self.reactions.b[i] * np.exp(-self.reactions.c[i]/self.reactions.temp_max[i])
                if self.gas.t_gas < self.reactions.temp_min[i]:
                    k = self.reactions.a[i] * (self.reactions.temp_min[i]/300.0)**self.reactions.b[i] * np.exp(-self.reactions.c[i]/self.reactions.temp_min[i])
                    
            # 22: Normal gas-phase reaction (switch off outside temperature range)
            elif self.reactions.itype[i] == 22:
                k = self.reactions.a[i] * (self.gas.t_gas/300.0)**self.reactions.b[i] * np.exp(-self.reactions.c[i]/self.gas.t_gas)
                if self.gas.t_gas > self.reactions.temp_max[i] or self.gas.t_gas < self.reactions.temp_min[i]:
                    k = 0.0

            # 30: Photodissociation
            elif self.reactions.itype[i] in (30, 34, 35, 36, 37, 39):
                k = self.reactions.a[i] * self.reactions.b[i] * self.conditions.G_0_unatt * np.exp(-self.reactions.c[i]*self.conditions.Av)                

            # 31: H2 dissociation inc. self-shielding
            elif self.reactions.itype[i] == 31:
                if self.parameters.column:
                    col_h2 = self.gas.h2_col
                else:
                    col_h2 = self.conditions.Av * self.parameters.av_nH * (y[i_H2]/self.gas.n_gas)
                ssfact_H2  = helpers.calc_selfshielding_h2(col_h2, self.parameters.delta_v) 
                k = self.reactions.a[i] * self.reactions.b[i] * self.conditions.G_0_unatt * np.exp(-self.reactions.c[i]*self.conditions.Av) * ssfact_H2 
                self.disso_H2 = k   # needed for reaction types 90 & 91
    
            # 32: CO dissociation inc. self-shielding
            elif self.reactions.itype[i] == 32:
                if self.parameters.column:
                    col_h2 = self.gas.h2_col
                else:
                    col_h2 = self.conditions.Av * self.parameters.av_nH * (y[i_H2]/self.gas.n_gas)
                col_co = (col_h2/np.maximum(1e-20, (y[i_H2]/self.gas.n_gas))) * (y[i_CO]/self.gas.n_gas)
                ssfactor_co = helpers.calc_selfshielding_co(self.ss_co[0], self.ss_co[1], self.ss_co[2], col_h2, col_co)
                k  = self.reactions.a[i] * self.reactions.b[i] * self.conditions.G_0_unatt * np.exp(-self.reactions.c[i]*self.conditions.Av) * ssfactor_co 

            # 33: C ionization inc. self-shielding
            elif self.reactions.itype[i] == 33:
                if self.parameters.column:
                    col_h2 = self.gas.h2_col
                else:
                    col_h2 = self.conditions.Av * self.parameters.av_nH * (y[i_H2]/self.gas.n_gas)  
                col_c = (col_h2/np.maximum(1e-20, (y[i_H2]/self.gas.n_gas))) * (y[i_C]/self.gas.n_gas)
                ssfactor_c = helpers.calc_selfshielding_c(col_h2, col_c, self.gas.t_gas)
                k  = self.reactions.a[i] * self.reactions.b[i] * self.conditions.G_0_unatt * np.exp(-self.reactions.c[i]*self.conditions.Av) * ssfactor_c

            # 38: N2 photodissociation inc. self-shielding
            elif self.reactions.itype[i] == 38:
                if self.parameters.column:
                    col_h2 = self.gas.h2_col
                else:
                    col_h2 = self.conditions.Av * self.parameters.av_nH * (y[i_H2]/self.gas.n_gas)
                col_h  = (col_h2/np.maximum(1e-20, (y[i_H2]/self.gas.n_gas))) * (y[i_H]/self.gas.n_gas)
                col_n2 = (col_h2/np.maximum(1e-20, (y[i_H2]/self.gas.n_gas))) * (y[i_N2]/self.gas.n_gas)
                ssfactor_n2 = helpers.calc_selfshielding_n2(self.ss_n2[0], self.ss_n2[1], self.ss_n2[2], self.ss_n2[3], col_h2, col_h, col_n2)
                k = self.reactions.a[i] * self.reactions.b[i] * self.conditions.G_0_unatt * np.exp(-self.reactions.c[i]*self.conditions.Av) * ssfactor_n2
                
            # 40: Direct cosmic ray ionization
            elif self.reactions.itype[i] == 40:
                k = self.reactions.a[i] * (self.conditions.Zeta_CR/1.35e-17)     # scaled value (see eg. UMIST22 paper Millar+22 §2.2)
            
            # 41: Cosmic ray / X-ray induced FUV reaction
            elif self.reactions.itype[i] == 41:
                k = self.reactions.a[i] * ((self.conditions.Zeta_CR+self.conditions.Zeta_X)/1.35e-17) * self.reactions.c[i]/(1.0-0.5) * (self.gas.t_gas/300.0)**self.reactions.b[i]
            
            # 42: Cosmic ray induced FUV reaction CO dissociation with self-shielding   (Visser et al. 2018 & Heays et al. 2014)
            elif self.reactions.itype[i] == 42:
                co_abun = (y[i_CO] / self.gas.n_gas) + 1e-12
                pd_eff  = 56.14 / (5.11e4 * (co_abun**0.792) + 1.0) + 4.3        # Visser et al. (2018) Eq. 2
                k = pd_eff * (self.conditions.Zeta_CR+self.conditions.Zeta_X)
            
            # 43: Cosmic ray induced FUV reaction He decay from 2.1P
            elif self.reactions.itype[i] == 43:
                k = self.reactions.a[i] * ((self.conditions.Zeta_CR+self.conditions.Zeta_X)*0.0107) * y[i_He] / (self.reactions.b[i] * y[i_H2] + self.reactions.c[i] * y[i_H] + 1e-20)

            # 60: X-ray secondary ionization of H
            elif self.reactions.itype[i] == 60:
                k = self.reactions.a[i] * self.conditions.Zeta_X * 0.56          # 0.56 from Staeuber/Doty code Eixion(1)/Eixion(2)
                
            # 61: X-ray secondary ionization of H2
            elif self.reactions.itype[i] == 61:
                k = self.reactions.a[i] * self.conditions.Zeta_X
            
            # 62: X-ray secondary ionization of other molecules
            elif self.reactions.itype[i] == 62:
                k = self.reactions.a[i] * self.conditions.Zeta_X
                
            # 70: Photoelectron production from PAHs/grains
            elif self.reactions.itype[i] == 70:
                k = self.reactions.a[i] * (self.conditions.G_0+1e-4)             # +1e-4 simulates CR ionization
      
            # 71: Charge exchange/recombination with PAHs/grains
            elif self.reactions.itype[i] == 71:
                phi_pah = 0.5
                k = self.reactions.a[i] * phi_pah * ((self.gas.t_gas/100.0)**self.reactions.b[i])
            
            # 72: Charge exchange/recombination with PAHs/grains (species heavier than H)
            elif self.reactions.itype[i] == 72:
                phi_pah = 0.5
                k = self.reactions.a[i] * phi_pah * ((self.gas.t_gas/100.0)**self.reactions.b[i]) * 1/np.sqrt(self.species.mass[ir1])  
                
            # 80: Thermal desorption
            elif self.reactions.itype[i] == 80:
                if self.gas.n_gas * self.conditions.dg100 < 1e3:
                    pregrain = 1e-99
                else:
                    pregrain = (np.pi * self.dust.radius**2 * n_gr) / np.maximum(n_ice, self.dust.binding_sites*n_gr)
                k = 4.0 * pregrain * self.reactions.a[i] * np.exp(-self.reactions.b[i]/self.dust.t_dust)
                    
            # 81: Freezeout
            elif self.reactions.itype[i] == 81:
                if self.gas.n_gas * self.conditions.dg100 < 1e3:
                    prefreeze = 1e-99
                else:
                    prefreeze = np.pi * self.dust.radius**2 * n_gr * np.sqrt( 8.0 * self.parameters.k_B / (np.pi * self.parameters.m_p) )
                k = self.reactions.a[i] * prefreeze * np.sqrt(self.gas.t_gas/self.species.mass[ir1])
                
            # 90: Pumping of H2 to H2*
            elif self.reactions.itype[i] == 90:
                tinv    = (self.gas.t_gas / 1000.0) + 1.0  # downward collision rate from Le Bourlot et al. (1999)
                col_HH2 = 10**(-11.058+0.05554/tinv-2.3900/(tinv*tinv))*(y[i_H]) + 10**(-11.084-3.6706/tinv-2.0230 /(tinv*tinv))*y[i_H2]
                k = self.disso_H2 * 10.0 + col_HH2 * np.exp(-30163.0 / self.gas.t_gas)      

            # 91: Radiative and collisional de-excitation of H2* to H2
            elif self.reactions.itype[i] == 91:
                tinv    = (self.gas.t_gas / 1000.0) + 1.0  # downward collision rate from Le Bourlot et al. (1999)
                col_HH2 = 10**(-11.058+0.05554/tinv-2.3900/(tinv*tinv))*(y[i_H]) + 10**(-11.084-3.6706/tinv-2.0230 /(tinv*tinv))*y[i_H2]
                k = self.disso_H2 * 10.0 + 2e-7 + col_HH2
            
            # 92: Further reactions with H2* + XX
            elif self.reactions.itype[i] == 92:
                k = self.reactions.a[i] * (self.gas.t_gas/300.0)**self.reactions.b[i] * np.exp(-np.maximum(0.0, self.reactions.c[i]-30163.0)/self.gas.t_gas)
                
            # ERROR: Reaction ID not found
            else:
                raise ValueError(f'ABORTED. Reaction type {int(self.reactions.itype[i])} is unknown! (reaction nr {int(self.reactions.reaction_id[i])})')
                
            # ERROR: NaN or Inf rate coefficient
            if np.isnan(k) or np.isinf(k):
                raise ValueError(f'ABORTED. Invalid rate coefficient for reaction nr {int(self.reactions.reaction_id[i])} (k={k})')
            
            self.reactions.k[i] = k

    
    
    ##################
    # RUN THE SOLVER #
    ##################

    def solve_network(self):
        """
        Solves the chemical reaction network using a stiff ODE solver.

        This method integrates the system of ordinary differential equations (ODEs) 
        that govern the chemical reaction network using SciPy's `solve_ivp` function 
        with the backward differentiation formula (BDF) method.

        The function tracks species abundances and reaction rates over time, updates 
        species properties at the final time, and stores the integration results for analysis.

        Returns:
            dict: A dictionary containing:
                - `time` (numpy.ndarray): Time points of the solution.
                - `abundances` (numpy.ndarray): Species abundances over time.
                - `rates` (numpy.ndarray): Reaction rates over the evaluated time points.
                - `success` (bool): Whether the integration was successful.
                - `message` (str): Solver message (success or error).
                - `species` (list): Names of the chemical species.
                - `reaction_labels` (list): List of formatted strings describing each reaction.
        """
        
        # Initial conditions
        t0 = np.float64(self.parameters.time_initial)
        tf = np.float64(self.parameters.time_final)
        y0 = np.array(self.species.number, dtype=np.float64)
        
        # Add small buffer to prevent floating point issues
        eps = 1e-10 * tf
        
        # Create logarithmically spaced time points
        t_eval = np.unique(np.concatenate([
            np.logspace(np.log10(t0), np.log10(t0 + (tf-t0)*0.1), 50),
            np.logspace(np.log10(t0 + (tf-t0)*0.1), np.log10(tf-eps), 50)
        ]))

        # Initialise storage for rates
        self.rate_history = np.zeros((len(t_eval), self.parameters.n_reactions))
        
        # Initialize lists to store ALL rates and times during integration
        self._all_rates = []
        self._all_times = []
        
        # Initialize progress bar
        print("Starting chemical network integration...")
        start_time = time.time()
        self._progress_bar = tqdm(total=100, desc="Progress", bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

        # Wrapper for derivatives()
        def wrapper_derivatives(t, y):
            # Update progress bar
            progress = min(100, int((t - t0) / (tf - t0) * 100))
            if self._progress_bar is not None:
                self._progress_bar.n = progress
                self._progress_bar.refresh()
            
            # Compute rate coefficients
            self.compute_rate_coefficients(y)

            # Calculate derivatives
            ydot = np.zeros(self.parameters.n_species, dtype=np.float64)
            ydot, rates = calculate_derivatives(y, self.reactions.k, self.idx, ydot, self.parameters.n_reactions)

            # Store time and rates
            self._all_times.append(t)
            self._all_rates.append(rates)
            return ydot

        # Wrapper for jacobian()
        def wrapper_jacobian(t, y):
            return calculate_jacobian(y, self.reactions.k, self.idx, self.parameters.n_reactions)

        # Solve the ODE system
        try:
            solution = solve_ivp(
                fun=wrapper_derivatives,
                t_span=(t0, tf),
                y0=y0,
                method='BDF',
                t_eval=t_eval,
                jac=wrapper_jacobian,
                rtol=1e-3,
                atol=1e-8,
                max_step=tf/10,
                first_step=t0/1e4               
                )
            
            # Ensure progress bar reaches 100%
            if self._progress_bar is not None:
                self._progress_bar.n = 100
                self._progress_bar.refresh()
                self._progress_bar.close()
            
            end_time = time.time()
            print(f"Integration completed in {end_time - start_time:.2f} seconds")
            
            if solution.success:
                print("Integration successful!")
                
                # Convert lists to arrays
                all_times = np.array(self._all_times)
                all_rates = np.array(self._all_rates)
                
                # For each evaluation time, find the closest actual computed time
                for i, eval_time in enumerate(t_eval):
                    idx = np.argmin(np.abs(all_times - eval_time))
                    self.rate_history[i] = all_rates[idx]
                
                # Store results
                self.abundance_history = solution.y.T
                self.time_points = solution.t
                self.species.number = solution.y[:,-1]
                
                return {
                    'time': solution.t,
                    'abundances': solution.y,
                    'rates': self.rate_history,
                    'success': True,
                    'message': solution.message,
                    'species': self.species.name,
                    'reaction_labels': self.reactions.labels
                }
            else:
                print(f"Integration failed: {solution.message}")
                return {
                    'success': False,
                    'message': solution.message
                }
                
        except Exception as e:
            if self._progress_bar is not None:
                self._progress_bar.close()
            print(e)
            return {
                'success': False,
                'message': str(e)
            }
        finally:
            if self._progress_bar is not None:
                self._progress_bar.close()
