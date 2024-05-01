#!/usr/bin/env python

#################################################################################################
# Import module
#################################################################################################

import sys
import os, shutil
sys.path.append('..')  # to import from GP.kernels and property_predition.data_utils
import random

import numpy as np
import pandas as pd
import crossover as co
import mutate as mu
import math

import sascorer as sascorer
from rdkit import Chem
from rdkit.Chem import AllChem

from property_prediction.data_utils import TaskDataLoader, featurise_mols

#import matplotlib.pyplot as plt
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

#################################################################################################
# Main function
#################################################################################################
    
def main():
    # Load learning sets
    data_loader = TaskDataLoader('QM_E1_CAM', 'qm8_1000.csv')
    next_generation, dummy = data_loader.load_property_data()
   
    #parameters for GA
    cycle = 20
    target_pool = 100
    target_SAS = 2.2
    
    co.average_size = 14
    co.size_stdev = 5
    co.string_type = 'SMILES'
    
    mu.mutation_rate = 0.1 #50% probability to mutate
    
    # GA cycle 
    for steps in range(cycle):
        
        # put selected smiles in 'selection'
        selection = selection_mol(next_generation, target_pool, target_SAS)
        
        next_generation = []
        ncross = 0
        while (ncross < target_pool):
            
            # select two random molecules to crossover
            X_tmp1 = random.choice(selection)
            X_tmp2 = random.choice(selection)
        
            # convert selected molecules from smiles to fingerprint
            mol1 = Chem.MolFromSmiles(X_tmp1)
            mol2 = Chem.MolFromSmiles(X_tmp2)
            
            # crossover two molecules
            child = co.crossover(mol1,mol2)
          
            # exception handling 
            if (child == None):
                continue
        
            # mutation
            if (random.random() < mu.mutation_rate):
                mutated_child = mu.mutate(child,1)
                child = mutated_child
                
                # exception handling 
                if (child == None):
                    continue
        
            # add offspring to pools for next generation
            next_generation.append(Chem.MolToSmiles(child))

            ncross += 1

        print(f"----------step {steps + 1}----------")
        for nmol, childs in enumerate(next_generation):
            print(f"#{nmol + 1:4.0f} Molecule: {childs}. It's SAS is equal to {sascorer.calculateScore(Chem.MolFromSmiles(childs))}")
    for val in selection:
        print(f"Molecule {val} has a SAS value equals to - {sascorer.calculateScore(Chem.MolFromSmiles(val))}")

#################################################################################################
# Calculation functions
#################################################################################################

def selection_mol(input_smiles_list, target_pool, target_SAS):
    # SAS score of molecules for evaluation
    sas_list = []
    temp_list = []
    n_cycle = 0 
    for smi in input_smiles_list:
        mol = Chem.MolFromSmiles(smi)
        #sas_list.append(sascorer.calculateScore(mol)) # SAscore in RDKit
        temp_list.append([sascorer.calculateScore(mol), n_cycle])
        n_cycle += 1
    temp_list.sort()

    ideal_list = []
    ind = 0 # closest to 2,2 value
    min_dif = 100000

    for i in range(len(temp_list)):
        if abs(temp_list[i][0] - 2.2) <= min_dif:
            min_dif = abs(temp_list[i][0] - 2.2)
            ind = i
    #now we have closest index
    print(ind)
    x = ind
    y = ind

    while len(sas_list) < target_pool:
        if abs(temp_list[x][0] - 2.2) <= abs(temp_list[y][0] - 2.2):
            sas_list.append(input_smiles_list[temp_list[x][1]])
            x -= 1
        else:
            sas_list.append(input_smiles_list[temp_list[y][1]])
            y += 1

    output_smiles_list = []

    ###################################
    ######## Change this part #########
    ###################################
   
    # write a code of selection step
    # In the selection step, molecules with good score(SAS is close to 'target_SAS' variable) have to 
    # have more probability to survive than bad score(SAS is far from 'target_SAS' variable) molecules.

    # Evaluate molecule based on SAS_list, append survive molecules into 'output_smiles_list'
    for i in range(target_pool):
        #output_smiles_list.append(random.choice(input_smiles_list))
        output_smiles_list.append(sas_list[i])

    ###################################

    return output_smiles_list


#################################################################################################
# Why python using this? :P
#################################################################################################

if __name__ == "__main__":
    main()
 
