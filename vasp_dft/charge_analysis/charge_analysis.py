#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@File    :   charge_dipole.py
@Time    :   2022/04/18 12:41:30
@Author  :   Daniel W 
@Version :   1.0
@Contact :   willhelmd@tamu.edu

Description: 
    A charge analysis class. Can calculate the elec dipole, ionic dipole, total dipole of a 2D material or 2D vdW bilayer (i.e. dipole along z-axis).
    *Note that the cell should not be periodic in the z-direction, i.e. with a large z-direction vacuum (to avoid cell image interactions). This is the 
    typical protocol for DFT calculations on 2D materials.  
    
    In addition, can determine the charge center of cell for use in dipole correction in VASP calculations. 
    Pass a pymatgen Vasprun, Chgcar, and Potcar class object as inputs (see other @classmethods for alternate inputs) 
    
'''
import os
import argparse
import json 
import numpy as np
from pathlib import Path
from pymatgen.io.vasp.outputs import Chgcar, Vasprun
from pymatgen.io.vasp.inputs import UnknownPotcarWarning, Potcar
import warnings

# get basepath
basedir = Path(os.path.dirname(os.path.abspath(__file__)))

# setup warnings
warnings.simplefilter('ignore',UnknownPotcarWarning )

class ChargeAnalysis (): 
    def __init__(self,vasprun:Vasprun, chgcar:Chgcar, potcar:Potcar ) -> None:
        self.vr     = vasprun 
        self.chgcar = chgcar 
        self.potcar = potcar 
    
    @classmethod
    def from_files(cls,vasprun_file, chgcar_file, potcar_file):
        vr     = Vasprun(vasprun_file)
        chgcar = Chgcar.from_file(chgcar_file)
        potcar = Potcar.from_file(potcar_file)
        return cls(vr, chgcar, potcar)

    @classmethod
    def from_dir(cls, path): 
        path = Path(path)
        in_files = [path.joinpath(file) for file in ['vasprun.xml','CHGCAR', 'POTCAR']] 
        all_exist = sum([i.exists() for i in in_files])
        if all_exist != 3:
            raise ValueError('Files could not be found in target directory') 
        else: 
            in_obj   = [Vasprun, Chgcar.from_file, Potcar.from_file]
            out = [iobj(ipath) for iobj,ipath in zip(in_obj, in_files)]
            return cls(*out)
    
    @property
    def structure(self): 
        return self.vr.final_structure

    @property
    def lattice_matrix(self): 
        return self.structure.lattice.matrix

    @property 
    def charge_density(self): 
        volume = self.vr.final_structure.volume 
        return self.chgcar.data['total'] / volume     # charge / volume

    @property
    def valence_from_potcar(self): 
        structure = self.structure
        total_electrons = sum([n_elec * structure.composition.as_dict()[element] for (element, n_elec) in [[p.element, p.nelectrons] for p in self.potcar] ])
        return total_electrons

    @property
    def valence_from_chgcar(self): 
        charge_density  = self.charge_density
        structure       = self.structure
        Nxyz            = np.array(charge_density.shape)
        dV              = np.linalg.det(structure.lattice.matrix) / Nxyz.prod()
        return charge_density.flatten().sum() * dV * -1

    def ionic_dipole_z(self): 
        """Get the inonic dipole alonge z-direction"""
        valence = {element: n_elec for (element, n_elec) in [[p.element, p.nelectrons] for p in self.potcar]  } 
        structure = self.vr.final_structure
        idipole_z = sum([valence[i.specie.symbol] * i.coords[2] for i in structure ])
        return idipole_z

    def electronic_dipole_z(self): 
        """Get the electronic dipole alonge z-direction"""
        # get charge density/vol
        lattice_matrix = self.vr.final_structure.lattice.matrix
        charge_density = self.charge_density
        # Get all grids and grids in z-direction
        Nxyz = np.array(charge_density.shape)
        Nz   = Nxyz[2]
        # get dV
        dV = np.linalg.det(lattice_matrix) / Nxyz.prod()
        # get charge total along z-direction 
        Zavg_z = np.array( [charge_density[:,:,ipt].sum() for ipt in range(Nz)] ) * dV 
        # determine the dipole moment
        c = self.vr.final_structure.lattice.c 
        edipole_z = -1*sum([ (z/Nz)*c * Zavg_z[i] for i,z in enumerate(range(Nz)) ])
        return edipole_z

    def total_dipole(self): 
        """Get the total dipole alonge z-direction"""
        total_dipole = self.electronic_dipole_z() + self.ionic_dipole_z() 
        return total_dipole

    def get_charge_center(self): 
        """Get the charge center"""
        lattice_abc   = self.structure.lattice.abc
        lattice_matrix = self.structure.lattice.matrix

        # charge properties
        n_gridpts = np.array(self.charge_density.shape)
        n_gridpts_yz = n_gridpts[1] * n_gridpts[2]
        n_gridpts_xz = n_gridpts[0] * n_gridpts[2]
        n_gridpts_xy = n_gridpts[0] * n_gridpts[1]

         # get dV
        dV = np.linalg.det(lattice_matrix) / n_gridpts.prod()
        area_yz, area_xz, area_xy = [
            np.linalg.det( np.delete(np.delete(lattice_matrix,i,0), i, 1) ) for i in range(3)] 
        dA_yz, dA_xz, dA_xy = area_yz/n_gridpts_yz, area_xz/n_gridpts_xz, area_xy/n_gridpts_xy # elementary area

        ##  get avg along a given direction by area
        Zavg_x = np.array([self.charge_density[ipt,:,:].sum() for ipt in range(n_gridpts[0])]) * dA_yz
        Zavg_y = np.array([self.charge_density[:,ipt,:].sum() for ipt in range(n_gridpts[1])]) * dA_xz
        Zavg_z = np.array([self.charge_density[:,:,ipt].sum() for ipt in range(n_gridpts[2])]) * dA_xy
        # Zavg_xyz = e/A

        avg_x=avg_y=avg_z=0.0
        edipole_z = 0
        for i in range(n_gridpts[0]):
            x = float(i)/n_gridpts[0]
            avg_x += x*Zavg_x[i]/Zavg_x.sum()

        for i in range(n_gridpts[1]):
            y = float(i)/n_gridpts[1]
            avg_y += y*Zavg_y[i]/Zavg_y.sum()

        for i in range(n_gridpts[2]):
            z = float(i)/n_gridpts[2]
            edipole_z += -1 * ( (z*lattice_abc[2]) * (Zavg_z[i]/dA_xy) * dV ) 
            avg_z += z*Zavg_z[i]/Zavg_z.sum()

        chg_center_frac = (avg_x, avg_y, avg_z)
        return chg_center_frac

    def get_charge_center_cart(self): 
        chg_center_frac = self.get_charge_center()
        chg_center_cart = tuple([sum(chg_center_frac[i] * self.lattice_matrix[:,i]) for i in range(3)])
        return chg_center_cart


    def as_dict(self): 
        data = dict() 
        data['total_dipole'] = self.total_dipole()
        data['elec_dipole'] = self.electronic_dipole_z()
        data['ionic_dipole'] = self.ionic_dipole_z() 
        data['charge_center_frac'] = self.get_charge_center()
        data['charge_center_cart'] = self.get_charge_center_cart()
        data['valence_elec'] = {'from_potcar':self.valence_from_potcar, 'from_chgcar':self.valence_from_chgcar}
        return data 
    
    def save_data(self, name='charge_analysis.json'):
        with open(name, 'w') as fh: 
            json.dump(self.as_dict(), fh) 


def printout(chg:ChargeAnalysis): 
    data = chg.as_dict()
    print('Charge Analysis')
    for k,v in data.items(): 
        print(f'\t{k}:\t{v}')
    print('\n')

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_dir',type=str,default='.',help='path/to/target/dir')
    parser.add_argument('--save', type=bool, default=False, help='0:dont save file 1:save file')
    args = parser.parse_args()
    charge_analy = ChargeAnalysis.from_dir(args.from_dir)
    if args.save == True: 
        charge_analy.save_data()
        print('\nSaving Charge Analysis to JSON')
        printout(charge_analy)
    else: 
        printout(charge_analy)

if __name__ == '__main__':
    main()


