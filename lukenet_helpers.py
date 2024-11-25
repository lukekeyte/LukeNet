import numpy as np
from scipy.interpolate import interp1d

##############################
# READ CHEMICAL NETWORK FILE #
##############################

def read_chemnet(network_file):

    with open(network_file, 'r') as file:
        for _ in range(4):
            file.readline()
        
        n_elements = int(file.readline().strip())

        file.readline() 
        
        elements_name = []
        for _ in range(n_elements):
            elem = file.readline().strip()
            elements_name.append(elem)
        
        for _ in range(4):
            file.readline()
                    
        n_species = int(file.readline().strip())

        file.readline()
        file.readline()
        
        # Initialize species arrays
        species_name   = []
        species_abu    = np.zeros(n_species)
        species_mass   = np.zeros(n_species)
        species_charge = np.zeros(n_species)
        
        # Extract species
        for i in range(n_species):
            line = file.readline().split()
            species_name.append(line[0])
            species_abu[i] = float(line[1])
            species_mass[i] = float(line[2])
            species_charge[i] = float(line[3])
            
        for _ in range(4):
            file.readline()
        
        n_reactions = int(file.readline().strip())
        
        file.readline()
        file.readline()
        
        # Initialize reactions arrays
        reactions_educts      = []
        reactions_products    = []
        reactions_reaction_id = np.zeros(n_reactions)
        reactions_itype       = np.zeros(n_reactions)
        reactions_a           = np.zeros(n_reactions)
        reactions_b           = np.zeros(n_reactions)
        reactions_c           = np.zeros(n_reactions)
        reactions_temp_min    = np.zeros(n_reactions)
        reactions_temp_max    = np.zeros(n_reactions)
        reactions_pd_data     = []
        
        # Extract reactions
        for i in range(0, n_reactions):
            line     = file.readline()
            educts   = [line[i:i+14].strip() for i in range(0, 3*14, 14)]
            products = [line[i:i+14].strip() for i in range(3*14, 8*14, 14)]
            
            reactions_educts.append(educts)
            reactions_products.append(products)

            remainder = line[8*14:].split()
            
            reactions_reaction_id[i] = remainder[0]
            reactions_itype[i] = remainder[1]
            reactions_a[i] = remainder[2]
            reactions_b[i] = remainder[3]
            reactions_c[i] = remainder[4]
            reactions_temp_min[i] = remainder[5]
            reactions_temp_max[i] = remainder[6]
            reactions_pd_data.append(remainder[7])
    
    return (n_elements, elements_name, n_species, species_name, species_abu, species_mass, species_charge, n_reactions, 
            reactions_educts, reactions_products, reactions_reaction_id, reactions_itype, reactions_a, reactions_b, reactions_c, 
            reactions_temp_min, reactions_temp_max, reactions_pd_data)



##########################
# CREATE REACTION LABELS #
##########################

def create_reaction_labels(n_reactions, educts, products):
    labels = []
    for i in range(0, n_reactions):
        educt_str   = ' + '.join(j for j in educts[i] if j != '')
        product_str = ' + '.join(j for j in products[i] if j != '')
        labels.append(f"{educt_str} -> {product_str}")
    return labels


###################
# PRINT DALI CELL #
###################

def print_dali_cell(dali_model_outdat_path, r, z):
    '''
    Prints key parameters from a specified cell in a DALI model
    
    Parameters:
    dali_model_outdat_path (string): Path to the DALI out.dat file
    r (int) : r-index of cell
    z (int) : z-index of cell
    '''
    
    dali = read_outdat(dali_model_outdat_path)

    rr = np.array(dali['ra'])[r, z]
    zz = np.array(dali['za'])[r, z]
    ngas = np.array(dali['n_gas'])[r, z]
    ndust = np.array(dali['n_dust'])[r, z]
    tgas = np.array(dali['t_gas'])[r, z]
    tdust = np.array(dali['t_dust'])[r, z]
    g0 = np.array(dali['G_0'])[r, z]
    g0_unatt = np.array(dali['G_0_unatt'])[r, z]
    av = np.array(dali['A_V'])[r, z]
    zetax = np.array(dali['Zeta_X'])[r, z]
    dg100 = np.array(dali['dg100'])[r, z]
    gtd = ngas/ndust
    pah_ism = np.array(dali['pah_abu'])[r, z]
    h2_col = np.array(dali['H2_COL'])[r, z]
    
    
    print('DALI MODEL')
    print('----------')
    print(f'n_gas     = {ngas:.10e}')
    print(f'n_dust    = {ndust:.10e}')
    print(f't_gas     = {tgas:.1f}')
    print(f't_dust    = {tdust:.1f}')
    print(f'gtd       = {gtd:.10e}')
    print(f'Av        = {av:.10e}')
    print(f'G_0       = {g0:.10e}')
    print(f'G_0_unatt = {g0_unatt:.10e}')
    print(f'Zeta_X    = {zetax:.10e}')
    print(f'h2_col    = {h2_col:.10e}')
    
    

############################
# SELF SHIELDING FUNCTIONS #
############################

def locate(x, arr):
    """
    Find where value x fits in sorted array arr.
    Returns (index, alpha) where:
    - index is the lower bound index
    - alpha is the interpolation factor (0-1) between index and index+1
    """
    arr = np.asarray(arr)
    
    # Handle out-of-bounds cases
    if arr[0] < arr[-1]:  # Increasing array
        if x <= arr[0]: return 0, 0.0
        if x >= arr[-1]: return len(arr)-2, 1.0
    else:  # Decreasing array
        if x >= arr[0]: return 0, 0.0
        if x <= arr[-1]: return len(arr)-2, 1.0
    
    # Find index using numpy's searchsorted
    idx = np.searchsorted(arr, x)
    if idx > 0:
        idx -= 1
    
    # Calculate interpolation factor
    alpha = (x - arr[idx]) / (arr[idx + 1] - arr[idx])
    
    return idx, alpha


def read_selfshielding_co(file):
    with open(file, "r") as fp:
        # Skip first 5 lines
        for _ in range(5):
            fp.readline()
        
        chem_coss_nNCO = int(fp.readline().split()[-1])
        chem_coss_nNH2 = int(fp.readline().split()[-1])
        
        if chem_coss_nNCO != 47 or chem_coss_nNH2 != 42:
            print("STOP: Format problem in data_coselfs_iso.dat")

        chem_coss_NCO = np.zeros(chem_coss_nNCO)
        chem_coss_NH2 = np.zeros(chem_coss_nNH2)
        chem_coss     = np.zeros((chem_coss_nNCO, chem_coss_nNH2))
        
        fp.readline()
        
        for i in range(chem_coss_nNCO):
            chem_coss_NCO[i] = float(fp.readline().strip())
        chem_coss_NCO = np.log10(chem_coss_NCO)

        fp.readline()

        for i in range(chem_coss_nNH2):
            chem_coss_NH2[i] = float(fp.readline().strip())
        chem_coss_NH2 = np.log10(chem_coss_NH2)

        fp.readline()

        for i in range(0, chem_coss_nNH2):
            chem_12coss_temp = np.array([]) 
            for _ in range(5):
                line = fp.readline().split()  
                sub_arr = np.array([np.log10(float(i)) for i in line])
                chem_12coss_temp = np.concatenate((chem_12coss_temp, sub_arr)) 
            chem_coss[:,i] = chem_12coss_temp
    
    return chem_coss_NCO, chem_coss_NH2, chem_coss


def read_selfshielding_n2(file):
    with open(file, 'r') as fp:
        # Skip first three lines
        for _ in range(3):
            fp.readline()
        
        # Read dimensions
        chem_n2ss_nNN2 = int(fp.readline().split()[-1])
        chem_n2ss_nNH2 = int(fp.readline().split()[-1])
        chem_n2ss_nNH  = int(fp.readline().split()[-1])
        
        fp.readline()
        
        if chem_n2ss_nNN2 != 46 or chem_n2ss_nNH2 != 46 or chem_n2ss_nNH != 10:
            print("STOP: Format problem in data_n2selfs_3d.dat")

        # Initialize arrays
        chem_n2ss_NN2 = np.zeros(chem_n2ss_nNN2)
        chem_n2ss_NH2 = np.zeros(chem_n2ss_nNH2)
        chem_n2ss_NH  = np.zeros(chem_n2ss_nNH)
        chem_n2ss     = np.zeros((chem_n2ss_nNN2, chem_n2ss_nNH2, chem_n2ss_nNH))
        
        for i in range(chem_n2ss_nNN2):
            chem_n2ss_NN2[i] = float(fp.readline().strip())
        chem_n2ss_NN2 = np.log10(chem_n2ss_NN2)
        
        fp.readline()

        for i in range(chem_n2ss_nNH2):
            chem_n2ss_NH2[i] = float(fp.readline().strip())
        chem_n2ss_NH2 = np.log10(chem_n2ss_NH2)
        
        fp.readline()

        for i in range(chem_n2ss_nNH):
            chem_n2ss_NH[i] = float(fp.readline().strip())
        chem_n2ss_NH = np.log10(chem_n2ss_NH)

        fp.readline()

        # Read N2 self-shielding factors
        for k in range(chem_n2ss_nNH):  # 10 blocks
            for j in range(chem_n2ss_nNH2):  # 46 lines per block
                # Read one line and split it into values
                line = fp.readline().strip()
                values = [float(x) for x in line.split()]
                for i in range(chem_n2ss_nNN2):  # 46 values per line
                    chem_n2ss[i][j][k] = np.log10(values[i])        

    return chem_n2ss_NN2, chem_n2ss_NH2, chem_n2ss_NH, chem_n2ss


def calc_selfshielding_co(chem_coss_NCO, chem_coss_NH2, chem_coss, col_h2, col_co):
    
    idx_h2, alpha_h2 = locate(np.log10(max(1.0, col_h2)), chem_coss_NH2)
    idx_co, alpha_co = locate(np.log10(max(1.0, col_co)), chem_coss_NCO)

    ssfact_CO = 10.0 ** (chem_coss[idx_co][idx_h2] * (1.0 - alpha_h2) * (1.0 - alpha_co) +
                        chem_coss[idx_co + 1][idx_h2] * alpha_co * (1.0 - alpha_h2) +
                        chem_coss[idx_co][idx_h2 + 1] * (1.0 - alpha_co) * alpha_h2 +
                        chem_coss[idx_co + 1][idx_h2 + 1] * alpha_h2 * alpha_co)
    
    return ssfact_CO


def calc_selfshielding_n2(chem_n2ss_NN2, chem_n2ss_NH2, chem_n2ss_NH, chem_n2ss, col_h2, col_h, col_n2):

    h2_idx, alpha_h2 = locate(np.log10(max(1.0, col_h2)), chem_n2ss_NH2)
    h_idx, alpha_h   = locate(np.log10(max(1.0, col_h)), chem_n2ss_NH)
    n2_idx, alpha_n2 = locate(np.log10(max(1.0, col_n2)), chem_n2ss_NN2)

    # Calculate first interpolation term
    dum1_ss = (chem_n2ss[n2_idx, h2_idx, h_idx] * (1.0 - alpha_h2) * (1.0 - alpha_n2) * (1.0 - alpha_h) +
               chem_n2ss[n2_idx + 1, h2_idx, h_idx] * alpha_n2 * (1.0 - alpha_h2) * (1.0 - alpha_h) +
               chem_n2ss[n2_idx, h2_idx + 1, h_idx] * (1.0 - alpha_n2) * alpha_h2 * (1.0 - alpha_h) +
               chem_n2ss[n2_idx + 1, h2_idx + 1, h_idx] * alpha_h2 * alpha_n2 * (1.0 - alpha_h))

    # Calculate second interpolation term 
    dum2_ss = (chem_n2ss[n2_idx, h2_idx, h_idx + 1] * (1.0 - alpha_h2) * (1.0 - alpha_n2) * alpha_h +
               chem_n2ss[n2_idx + 1, h2_idx, h_idx + 1] * alpha_n2 * (1.0 - alpha_h2) * alpha_h +
               chem_n2ss[n2_idx, h2_idx + 1, h_idx + 1] * (1.0 - alpha_n2) * alpha_h2 * alpha_h +
               chem_n2ss[n2_idx + 1, h2_idx + 1, h_idx + 1] * alpha_h2 * alpha_n2 * alpha_h)

    # Calculate self-shielding factor
    ssfact_N2 = 10.0 ** (dum1_ss + dum2_ss)

    return ssfact_N2


def calc_selfshielding_h2(col_h2, delta_v):
    delta_v    = 0.2
    nh2_5e14   = col_h2 / 5e14
    ssfact_H2  = 0.965 / (1.0+(nh2_5e14/delta_v))**2 + 0.035 / np.sqrt(1.0 + nh2_5e14) * np.exp(-8.5e-4 * np.sqrt(1.0+nh2_5e14)) 
    return ssfact_H2

def calc_selfshielding_c(col_h2, col_c, t_gas):
    ssfactor_c = np.exp(-col_c*1.1e-17) * np.maximum(0.5, np.exp(-0.9 * (t_gas**0.27) * ((col_h2/1e22)**0.45)))
    return ssfactor_c


##########################
# READ DALI OUT.DAT FILE #
##########################
def read_outdat(fname):

	'''
	Read DALI out.dat file
	Argument:
	- fname (full path to out.dat file)
	'''

	data={}

	# read file
	lines=open(fname,"r").readlines()
	
	# read header
	data['n_r'], data['n_z']=map(int,lines[1].split())	
	head=lines[3].split()	
	data['n_col']=len(head)

	# set up array   
	for h in head:
		data[h]=[[0.0 for i_z in range(data['n_z'])] for i_r in range(data['n_r'])]
	
	# read data
	lin=5
	for i_r in range(data['n_r']):
		for i_z in range(data['n_z']):
			splt = list(map(float, lines[lin].split()))		
			for i_c in range(data['n_col']):
				data[head[i_c]][i_r][i_z]=splt[i_c]
			lin+=1
			
	# find wavelength
	wave=[]
	for d in data.keys():
		if d[0:2]=="J=":			
			wave.append(float(d[2:]))
	wave.sort()
	data['wave_grid']=wave
	
	
	# define midpoint of the cell (used for interpolation)
	data['r_mid']=[[0.0 for i_z in range(data['n_z'])] for i_r in range(data['n_r'])]
	data['z_mid']=[[0.0 for i_z in range(data['n_z'])] for i_r in range(data['n_r'])]
	
	rmax=0.0
	zmax=0.0	
	for i_r in range(data['n_r']):
		for i_z in range(data['n_z']):
			data['r_mid'][i_r][i_z]=0.5*(data['ra'][i_r][i_z]+data['rb'][i_r][i_z])
			data['z_mid'][i_r][i_z]=0.5*(data['za'][i_r][i_z]+data['zb'][i_r][i_z])	
			zmax=max(zmax,data['zb'][i_r][i_z])
			rmax=max(rmax,data['rb'][i_r][i_z])				
	data['r_max']=rmax
	data['z_max']=zmax					
	return data