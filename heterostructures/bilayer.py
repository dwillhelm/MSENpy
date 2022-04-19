#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 18:46:03 2021

@author: daniel
"""

import numpy as np
import pymatgen as mg
from pymatgen.core.structure import Structure
from pymatgen.core.structure import Lattice
from pymatgen.core.structure import Species
import cell_tools as ct
# from pyvasp.toolbox import cell_tools as ct
from pymatgen.io.ase import AseAtomsAdaptor as AAA
from mendeleev import element

#%%
class BilayerBuilder (): 
    def __init__(self,max_latmm=5.0,vdw_space=3.4,vacuum=38.0,buffer=0): 
        self.max_latmm = max_latmm
        self.vacuum = vacuum
        self.vdw_space = vdw_space
        self.buffer = buffer

    def monolayer_input(self,layer1,layer2): 
        self.ml1 = layer1
        self.ml2 = layer2

    def lattice_mismatch(self):
        a1 = self.ml1.lattice.a     
        a2 = self.ml2.lattice.a
        a = np.array([a1,a2])
        a.sort()
        a1,a2 = a
        return 100 * abs( (a1 - a2) / a1)    
    
    def get_bilayer_lattice(self): 
        # get new lattice parameters
        a1,b1,_ = self.ml1.lattice.abc
        a2,b2,_ = self.ml2.lattice.abc
        new_a = (a1 + a2)/2
        new_b = (b1 + b2)/2
        abc = (new_a,new_b,self.vacuum)
        lattice = np.array([[1,    0,            0],
                            [-1/2, np.sqrt(3)/2, 0],
                            [0,    0,            1]])  
        bilayer_lattice = Lattice(abc*lattice)
        return bilayer_lattice
    
    def structure2atoms(structure):
        return AAA().get_atoms(structure)
    
    def add_vacuum(self,structure,new_c):
        C = new_c - structure.lattice.c 
        addvac=[0,0,C]
        # get old cell params
        abc0 = np.array(structure.lattice.abc)
        ang0 = np.array(structure.lattice.angles)
        r0   = structure.frac_coords
        spc0 = structure.species
        # new lattice params: 
        abc1 = abc0 + addvac
        M = Lattice.from_lengths_and_angles(abc1,ang0)
        f = abc0/abc1 # scaling factor
        r1 = r0*f     # adjust/scaling posiitons
        cell = Structure(lattice=M,
                            species=spc0,
                            coords=r1,
                            coords_are_cartesian=False,
                            site_properties=structure.site_properties)
        
        atoms0 = ct.structure2atoms(structure)
        atoms1 = ct.structure2atoms(cell)
        
        d0 = atoms0.get_all_distances()
        d1 = atoms1.get_all_distances()
        # print('delta atom distances, i.e. error round 1e-9 ')
        # print(np.round(d1-d0,9))
        return cell
    
    def get_covalent_radius(self,specie): 
        ele = element(specie.name)
        return ele.covalent_radius/100 # convert to Angstrom
    
    def make_composite_layer(self,structure,which:str): 
        # build new cell from monolayer coords/species with bilayer lattice
        coords  = structure.frac_coords
        species = structure.species
        cell    = Structure(self.bilayer_lattice,species,coords)
        
        # build bottom layer
        if which == 'bottom': 
            keyatom = cell[cell.cart_coords[:,2].argmax()]
            atom_buffer = self.get_covalent_radius(keyatom.specie) * self.buffer
            shift = (cell.lattice.c/2 - keyatom.z - (3.4/2) - atom_buffer)
        
        # build top layer
        elif which == 'top': 
            keyatom = cell[cell.cart_coords[:,2].argmin()]
            atom_buffer = self.get_covalent_radius(keyatom.specie) * self.buffer
            shift = (cell.lattice.c/2 - keyatom.z + (3.4/2) + atom_buffer)
        
        shift_vector = np.array([0,0,shift])
        idxs = [i for i in range(len(cell))]
        cell.translate_sites(idxs,shift_vector,to_unit_cell=False,frac_coords=False)
        return cell
        
        
    def build_bilayer(self):
        # check lattice mismatch
        latmm = self.lattice_mismatch()
        if latmm > self.max_latmm: 
            print(f'lattice mismatch of {latmm} exceeds max lattice mismatch paramerter = {self.max_latmm}')
            return np.nan
        
        # get bilayer lattice
        self.bilayer_lattice = self.get_bilayer_lattice()
        
        # add vacuum and scale monolayers
        new_c = self.bilayer_lattice.c
        cell1 = self.add_vacuum(self.ml1,new_c)
        cell2 = self.add_vacuum(self.ml2,new_c)
        cell1 = self.make_composite_layer(cell1,'bottom')
        cell2 = self.make_composite_layer(cell2,'top')
        
        # build bilayer: 
        coords = np.vstack((cell1.frac_coords,cell2.frac_coords))
        species = np.hstack((cell1.species,cell2.species))
        bilayer = Structure(self.bilayer_lattice, species, coords)        
        
        
        
        return bilayer




class Bilayer_Structure_Analysis():
    """
    Class for quick/simple analysis/parsing of a bilayer heterostructure
    
    """
    def __init__(self,structure_object,layer0_idx,layer1_idx):
        self.cell = structure_object
        self.idxbot = layer0_idx
        self.idxtop = layer1_idx
        
        def pull_layer(cell,indx):
            coords,species = [],[]
            for i in indx:
                atom = cell[i]
                coords.append([atom.a,atom.b,atom.c])
                species.append(atom.specie)
            coords = np.array(coords)
            return [coords,species]
        l0 = pull_layer(self.cell,layer0_idx)
        l1 = pull_layer(self.cell,layer1_idx)
        self.layer0 = Structure(lattice=self.cell.lattice,
                             species=l0[1],
                             coords=l0[0],
                             coords_are_cartesian=False)
        self.layer1 = Structure(lattice=self.cell.lattice,
                             species=l1[1],
                             coords=l1[0],
                             coords_are_cartesian=False)
        self.blta = self.idxbot[self.cell.cart_coords[self.idxbot][:,2].argmax()]
        self.tlba = self.idxtop[self.cell.cart_coords[self.idxtop][:,2].argmin()]
    
    def __repr__(self):
        r = repr(self.cell)
        return r

    @classmethod
    def from_split(cls,cell,split=0.5): 
        bot_idx = np.where(cell.frac_coords[:,2] < split)[0]
        top_idx = np.where(cell.frac_coords[:,2] > split)[0]
        return cls(cell,bot_idx,top_idx)
        
    @property
    def interlayer_distance(self):
        if isinstance(self.cell,str):
            return 'Error, check if heterstucture is possible'         
        else:
            cell = self.cell 
        
        d = cell.cart_coords[self.tlba][2] - cell.cart_coords[self.blta][2]       
        return d 
        
    @property
    def get_face_atom_idxs(self):
        if isinstance(self.cell,str):
            return 'Error, check if heterstucture is possible' 
        else:
            # BLTA = Bottom Layer Top-Facing Atom
            # TLBA = Top Layer Bottom-Faceing Atom
            cell1 = self.layer0
            cell2 = self.layer1
        
            BLTA = cell1.cart_coords[:][:,2].argmax()
            TLBA = cell2.cart_coords[:][:,2].argmin()
            tlba2 = TLBA+(cell1.num_sites)
            het_idx = [BLTA,tlba2]
            return het_idx,BLTA,TLBA
    
    def get_MX_(self,order='row'):
        if order not in ['row','oxi']:
            print('error: order can only be "row" or "oxi"')
        def MX(cell,order='row'):
            cell.add_oxidation_state_by_guess()
            sp = cell.composition.elements
            if len(sp) < 2:
                M = sp[0].name
                X = np.nan
            else: 
                row_order = np.argsort([mg.Element(i.name).group for i in sp])
                oxi_order = np.argsort([i.oxi_state for i in sp])[::-1]
                cell.remove_oxidation_states() 
                if order == 'oxi': 
                    sp =[sp[oxi_order[0]],sp[oxi_order[1]]]
                if order == 'row': 
                    sp =[sp[row_order[0]],sp[oxi_order[1]]]
                M = sp[0].name
                X = sp[1].name
            return M,X
        self.M0,self.X0 = MX(self.layer0)
        self.M1,self.X1 = MX(self.layer1)
        
    @property
    def get_inner_atoms_idx(self):
        out0 = self.get_face_atom_idxs[1]
        out1 = self.get_face_atom_idxs[2]
        
        try:
            l0 = list(self.layer0.symbol_set)
            if len(l0)<2:
                l0 = l0[0]
            else:
                l0.remove(self.layer0.species[out0].symbol)
                l0 = l0[0]
        except:
            l0 = np.NaN
        try:
            l1 = list(self.layer1.symbol_set)
            if len(l1)<2:
                l1 = l1[0]
            else: 
                l1.remove(self.layer1.species[out1].symbol)
                l1 = l1[0]
        except:
            l1 = np.NaN
        
        return l0,l1
    
    def gen_name(self):
        name1 = f'{self.layer0.composition.get_integer_formula_and_factor()[0]}'
        name2 = f'{self.layer1.composition.get_integer_formula_and_factor()[0]}'
        return f'{name1}-{name2}'
    
    def _get_cell_area_OLD(self):
        return self.cell.lattice.a * self.cell.lattice.b
        
    def get_cell_area(self): 
        a,b,_ = self.cell.lattice.abc
        v1, v2 = self.cell.lattice.matrix[:2] # pull x and y lattice vector
        v1 = v1/np.linalg.norm(v1)  # get norm vec
        v2 = v2/np.linalg.norm(v2)  # get norm vec 
        theta = np.arccos(v1@v2)    # angle b/w v1 and v2
        return a*b*np.sin(theta)    # area = a*b*sin(theta)
        
        

#%%