#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:18:28 2020

Additional Tools for working with pymatgen structure object

@author: daniel
"""
import numpy as np 
import pickle
import pymatgen as mg 
from pymatgen.io.ase import AseAtomsAdaptor as AAA

from ase.io import read 
from ase.visualize import view 

def structure2poscar(structure,filename):
    mg.io.vasp.inputs.Poscar(structure).write_file(filename)

def xyz_to_structure(filename):
    return AAA().get_structure(read(filename)) 

def structure2atoms(structure):
    return AAA().get_atoms(structure)

def structure2pickle(structure,filename):
    pickle.dump(structure, open(f'{filename}.pkl','wb'))

def pickle2structure(filename):
    return pickle.load(open(filename,'rb'))

def center_of_mass(structure):
    """
    calculates the center of mass of a pymatgen structure obj\n
    Args:\n 
    structure: a pymatgen structure object
    """
    atoms = AAA().get_atoms(structure)
    return atoms.get_center_of_mass()

def translate_cell_to_center_mass(structure):
    """
    translates a structure to the center of mass\n
    Args: 
    structure: a pymatgen structure object
    """
    cell = structure
    cm  = center_of_mass(structure) # struc. center mass 
    cm0 =np.array([1/2,1/2,1/2])*cell.lattice.abc # cell center mass 
    v = cm0 - cm # translation vector
    cell.translate_sites(range(len(cell)), v, frac_coords=False)
    print(f'New center mass = {cm0}')

def quickview(cell ):
    """
    quick view of structure when working in Spyder IDE. Uses ASE view package.\n  
    Args:     
        structure: a pymatgen structure object
    """
    view(AAA().get_atoms(cell))

def quickview_from_file(filename ):
    """
    quick view of structure when working in Spyder IDE. Uses ASE view package.\n  
    Args:     
        structure: a pymatgen structure object
    """
    atoms = read(filename)
    view(atoms)

def calculate_vacuum_size(structure):
    """
    Calculate the approixmate vac. size.\n  
    Args:     
        structure: a pymatgen structure object
    """
    cell = structure
    a_min = cell.cart_coords[:,2].min()
    a_max = cell.cart_coords[:,2].max()
    vac_dist1 = cell.lattice.c - a_max
    vac_dist2 = a_min
    return vac_dist1 + vac_dist2

def add_vacuum_old(structure,addvac=[0,0,5]):
    """
    add vaccumm to a given structure\n
    Args: 
        structure: a pymatgen structure object\n
        addvac: 1x3 input corresponding to size of vaccuum in the x,y or z direction- [x,y,z]         
    """
    cell = structure
    abc = np.array(cell.lattice.abc)
    frac = cell.frac_coords
    unit_matrix = cell.lattice.matrix/abc
    new_abc = abc + addvac
    N = new_abc/abc 
    cell.modify_lattice(mg.Lattice(unit_matrix*new_abc))
    cent = np.array([0.5,0.5,0.5])
    shift = cent - cent/N 
    new_frac = frac/N + shift 
    for i in range(cell.num_sites):
        cell[i] = str(cell.species[i]),new_frac[i].tolist()
    return cell


def add_vacuum(structure,addvac=[0,0,5]):
    # get old cell params
    abc0 = np.array(structure.lattice.abc)
    ang0 = np.array(structure.lattice.angles)
    r0   = structure.frac_coords
    spc0 = structure.species
    # new lattice params: 
    abc1 = abc0 + addvac
    M = mg.Lattice.from_lengths_and_angles(abc1,ang0)
    f = abc0/abc1 # scaling factor
    r1 = r0*f     # adjust/scaling posiitons
    cell = mg.Structure(lattice=M,
                        species=spc0,
                        coords=r1,
                        coords_are_cartesian=False,
                        site_properties=structure.site_properties)
    
    atoms0 = structure2atoms(structure)
    atoms1 = structure2atoms(cell)
    
    d0 = atoms0.get_all_distances()
    d1 = atoms1.get_all_distances()
    print('delta atom distances, i.e. error round 1e-9 ')
    print(np.round(d1-d0,9))
    return cell
    
def compare_structure(cell1,cell2):
    atoms0 = structure2atoms(cell1)
    atoms1 = structure2atoms(cell2)
    d0 = atoms0.get_all_distances()
    d1 = atoms1.get_all_distances()
    return np.round(d1-d0,9)
    
def calc_surface_area(structure,latdir=[0,1]): 
    """
    Calculate the surface area of a unit cell\n
    Args: 
        structure: a pymatgen structure object\n
        latdir: 1x2 list where: 
            0 = x-direction
            1 = y-direction
            2 = z-direction
    """    
    u = structure.lattice.matrix[latdir[0]]
    v = structure.lattice.matrix[latdir[1]]
    area = np.linalg.norm(np.cross(u,v))
    return area
    
    
    
    
    
    
    
    