import os
import numpy as np
import itertools
import subprocess

def grouper(iterable, n, fillvalue = None):
    """
    Reads chunks of n lines from iterable. Fills missing lines at the end with fillvalue.
    Taken from the recipes in the itertools docs.
    """
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

def get_shuffled_indices(num_examples):
    indices = np.arange(num_examples)
    np.random.shuffle(indices)
    return indices

def load_data(paths, tgt, natom):
    """ Load data from Gromacs output format. """
    data = [parse_force_data(path, natom) for path in paths]
    print(data[0][0].shape, len(data[0]))

    data_all = []
    for d in range(len(data[0])):
        data_all.append(np.concatenate(([j[d] for j in data])))
        
    #decide if site or couplings
    if tgt == "cpl":
        return data_all[0], data_all[1]
    elif tgt == "site":
        return data_all[2], data_all[3]

def shuffle_and_split(xyz, targets, ntrain):
    shuffled_idx = get_shuffled_indices(targets.shape[0])

    xyz = xyz[shuffled_idx[:ntrain]]
    y = targets[shuffled_idx[:ntrain]]

    #separate coordinates from gradients
    x = xyz[:,:,1:4]
    grads = xyz[:,:,4:]
    return x, grads, y, shuffled_idx[ntrain:]

def generate_invd_list(tgt, rep, natom):
    # decide if reduced (intermolecular) or full representation (couplings only):
    if tgt == "cpl" and rep == "inter":
        invd_list =  [[i, j] for i in range(natom) for j in range(natom, 2*natom)]
    elif tgt == "site":
            invd_list =  [[i, j] for i in range(natom) for j in range(natom)]
    else:
        invd_list =  [[i, j] for i in range(2*natom) for j in range(2*natom)]
    return invd_list

def parse_single_file(file, n_atoms, cutoff=None, every_nth=1):
    lines_per_geom = n_atoms+2

    geom_data = []
    tgt_data = []

    with open(file) as f:

        for idx, lines in enumerate(grouper(f, lines_per_geom, '')):
            #Alternative End Conditions
            if not all((l.strip for l in lines)):
                print(f"end of file reached at chunk {idx}")
                break # Whitespace at EOF
            if cutoff is not None and idx ==cutoff:
                print("cutoff reached")
                break # Cutoff
            if idx % every_nth != 0:
                print(f"skipping step {idx}")
                continue # read out only every n-th geometry

            mol_geoms = np.empty((n_atoms, 7))
            #print(mol_geoms.shape)
            tgt = [float(l) for l in lines[0].rstrip().split()]
            if len(tgt) == 1:
                tgt_data.append(tgt[0])
            else: 
                tgt_data.append(tgt)
            
            for at_idx, l in enumerate(lines[1:-1]):                    
                vals = [float(i) for i in l.split()]
                mol_geoms[at_idx] = np.asarray(vals+[0.0]*(7-len(vals)))
            geom_data.append(mol_geoms)
    return np.asarray(geom_data, dtype=np.float32), np.asarray(tgt_data, dtype=np.float32)

def parse_force_data(path, n_atoms_per_mol,
                     cutoff = None, every_nth = 1):

    pair_geoms, couplings, site_geoms, sites = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
    for file in os.listdir(path):
        print(file)
        if "_OFFDIAG" in file:
            n_atoms = 2*n_atoms_per_mol
        elif "_DIAG" in file:
            n_atoms = n_atoms_per_mol
        else:
            continue

        geom_data, tgt_data = parse_single_file(os.path.join(path, file), 
                                                n_atoms, 
                                                cutoff=cutoff, every_nth=every_nth)

        print(geom_data.shape, tgt_data.shape)
        if "_OFFDIAG" in file:
            pair_geoms = geom_data
            couplings = tgt_data
        elif "_DIAG" in file:
            site_geoms = geom_data
            sites = tgt_data
    return pair_geoms, couplings, site_geoms, sites

def parse_multiple(path, **kwargs):
    data = [[] for _ in range(4)]
    for d in os.listdir(path):
        if "c-dir" in d:
            continue
        p = os.path.join(path, d)
        file_data = parse_force_data(p, **kwargs)
        for i in range(len(file_data)):
            data[i].append(file_data[i])
    out = [np.vstack(d) for d in data]
    return out

def normalize_invdist(coords):
    a = np.expand_dims(coords, axis = 1)
    b = np.expand_dims(coords, axis = 2)
    c = b-a
    d = np.sum(np.square(c), axis = -1)
    d = d[~np.isclose(d,0)]
    d = 1/np.sqrt(d)
    return d.mean(), d.std()

def normalize_angles(coords, anglist):
    return 1,1

def normalize_dihedrals(coords, dihedlist):
    return 1,1

def get_file_length(file_path):
    result = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE, text=True)
    # The output will be in the form "number file_path", so we split it and take the first part
    line_count = int(result.stdout.split()[0])
    return line_count

def extract_number_of_atoms(file_path, lines_to_skip=0):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    num_atoms = 0
    for line in lines:
        # Count the number of lines until the first empty line
        line_tmp = line.strip()
        if line_tmp == "":
            break
        num_atoms += 1
    
    num_atoms -= lines_to_skip # Subtract one because the first line is the energy
    return num_atoms

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

unit_conversions = {
    'A2Bohr': 1.8897259886,
    'EhtoeV': 27.2114,
    'ehtonm': 299792458 * 6.62607015e-7 / 4.3597482,
    'HaB_to_eVA': 27.2114 / 0.52918,
}

def load_data_excited_states_forces(inputfile, lines_to_skip):

    AtoBohr = unit_conversions['A2Bohr']

    n_atoms = extract_number_of_atoms(inputfile, lines_to_skip)
    linestotal = get_file_length(inputfile)
    ntotal = linestotal / (n_atoms + lines_to_skip + 1) # + 1 because of the empty line at the end of each geometry
    if not ntotal.is_integer():
        raise ValueError("Number of Lines incorrect.")
    data_size = int(ntotal)

    with open(inputfile, "r") as data:
        x = []
        y = []
        for i in range(data_size):
            energy = data.readline()
            tmpy = []
            tmpy.append(np.sum([float(energy) for energy in energy.split()]))
            comp_tmp = []
            for j in range(n_atoms):
                line = data.readline()
                line_split = [float(n) for n in line.split()[1:]]
                coords = [AtoBohr * n for n in line_split[:3]]
                coords.append(line_split[3])
                comp_tmp.append(coords)
                tmpy.extend(line_split[4:])
            x.append(comp_tmp)
            y.append(tmpy)
            data.readline()
    
    return x, y