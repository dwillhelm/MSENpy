#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@File    :   general_analysis.py
@Time    :   2020/04/19 13:01:39
@Author  :   Daniel W 
@Version :   1.0
@Contact :   willhelmd@tamu.edu
'''
import numpy as np
import pandas as pd
import os
import glob
import pymatgen as mg 
from pymatgen.io.vasp.outputs import Poscar 
from pymatgen.io.vasp.outputs import Outcar 
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp.outputs import Locpot
from pymatgen.io.vasp.outputs import Procar
from pymatgen.io.vasp.outputs import VolumetricData
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp.inputs import Incar
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.inputs import Structure 
from pathlib import Path

# get basepath
basedir = Path(os.path.dirname(os.path.abspath(__file__)))

class PullVaspFiles ():
    """Helper class to quickly pull/parse some important VASP output/input files."""
    def __init__(self,directory_path,cell_is_final_struc=True):
        self.dir = directory_path
        self.cell_is_final_struc = cell_is_final_struc
        self.file_list = ['POSCAR',
                          'CONTCAR',
                          'vasprun.xml',
                          'LOCPOT',
                          'PROCAR',
                          'KPOINTS',
                          'INCAR',
                          'OUTCAR']
        # Check if file exist in target dir: 
        d = {} 
        for ifile in self.file_list:
            if os.path.isfile(os.path.join(self.dir,ifile)):
                d.update( {ifile:os.path.join(self.dir,ifile)} ) 
            else:
                d.update( {ifile:None} ) 
        self.path_dict = d 
        
        # Pull files and mk pymatgen objs: 
        if self.cell_is_final_struc:
            self.cell = Structure.from_file(self.path_dict['CONTCAR'])
        if d['POSCAR'] is not None: 
            self.poscar = Poscar.from_file(self.path_dict['POSCAR'])
        if d['CONTCAR'] is not None: 
            self.contcar = Poscar.from_file(self.path_dict['CONTCAR'])
        if d['vasprun.xml'] is not None: 
            self.vr = Vasprun(self.path_dict['vasprun.xml'])
        if d['LOCPOT'] is not None: 
            self.locpot = Locpot.from_file(self.path_dict['LOCPOT'])
            self.voldata = VolumetricData(self.cell,self.locpot.data)
        if d['PROCAR'] is not None: 
            self.procar = Procar(self.path_dict['PROCAR'])
        if d['KPOINTS'] is not None: 
            self.kpoints = Kpoints.from_file(self.path_dict['KPOINTS'])
        if d['INCAR'] is not None: 
            self.incar = Incar.from_file(self.path_dict['INCAR'])
        if d['OUTCAR'] is not None: 
            self.outcar = Outcar(self.path_dict['OUTCAR'])
    
    def get_runtime(self,time_unit='s'):
        """
        get the run time in seconds (s), minutes (m), or hours (h)
        """
        script_params = ['s','m','h']
        if time_unit not in script_params:
            print('ERROR: incorrect usage')
            print('---- please set time_unit as "s", "m", or "h"' )
            return np.nan
        runtime = self.outcar.run_stats['Total CPU time used (sec)']
        
        if time_unit == 's': 
            return runtime
        if time_unit == 'm': 
            return runtime * (1/60)
        if time_unit == 'h': 
            return runtime * (1/3600)
    
    def as_dict(self):
        return self.path_dict
        """
        return {'POSCAR':  self.poscar,
                'CONTCAR': self.contcar,
                'vasprun.xml': self.vr,
                'LOCPOT':self.locpot,
                'PROCAR':self.procar,
                'KPOINTS':self.kpoints,
                'INCAR':self.incar,
                'VolumetricData':self.voldata,
                'cell':self.cell}        
        """


class PyVaspAnalysis (): 
    def __init__(self,target_dir='./'):
        self.vaspfiles = PullVaspFiles(target_dir,True)
        self.vr = self.vaspfiles.vr
        # check if run succesfuly completed: 
        if self.vr.converged: 
            self.converged = True
        else:
            self.converged = False
        
    def get_runtime(self,time_unit='s'):
        """
        get the run time in seconds (s), minutes (m), or hours (h)
        """
        script_params = ['s','m','h']
        if time_unit not in script_params:
            print('ERROR: incorrect usage')
            print('---- please set time_unit as "s", "m", or "h"' )
            return np.nan
        runtime = self.vaspfiles.outcar.run_stats['Total CPU time used (sec)']
        
        if time_unit == 's': 
            return runtime
        if time_unit == 'm': 
            return runtime * (1/60)
        if time_unit == 'h': 
            return runtime * (1/3600)

    def get_ionic_steps(self):
        pass
        
def check_convergence (vasprun_path):
    path = Path(vasprun_path)
    if path.is_file() is False:
        return np.nan 
    try:
        vr = Vasprun(path)
        if vr.converged: 
            return True
        else: 
            return False
    except: 
        return np.nan 