#!/usr/bin/env bash

# import pandas as pd
import os
from shutil import copyfile
from glob import glob
from global_variables import create_folder

# Where the folders containing the strain are
sj_data = r'/Users/alestawsky/Downloads/complete data Curr Biol 20, 1099-1103 (2010)/'

# Path of where we want to copy TO
datasets_path = r'/Users/alestawsky/PycharmProjects/Thesis/Datasets/'

# Name of all the different strains
strains = glob(sj_data+'*/')

wang_datasets = []

for s in strains:
    print('strain:', s.split('/')[-2])
    
    experiments = glob(s+'*/')
    for exp in experiments:
        print('exp:', exp.split('/')[-2])
        
        new_path = datasets_path + exp.split('/')[-2] + '_' + s.split('/')[-2].replace('. ', '_').replace(' ', '_').replace(':', '') + '_Wang2010'
        
        # Create the places where we will leave the data
        create_folder(new_path)
        create_folder(new_path + '/RawData')
        create_folder(new_path + '/ProcessedData')
        create_folder(new_path + '/Figures')
        print(new_path)
        
        # the list of it all
        wang_datasets.append(new_path.split('/')[-1])
        
        # There's an extra level of folders that I do not understand YFP000(1/2)
        if (exp.split('/')[-2] == '20090529') or (exp.split('/')[-2] == '20090525'):
            proteins = ['YFP0001', 'YFP0002']
            for prot in proteins:
                positions = glob(exp+'{}/'.format(prot)+'*/')
                for pos in positions:
                    print('pos:', pos.split('/')[-2])
                    channels = glob(pos + '*.dat')
                    for chan in channels:
                        if 'cell0' not in chan:
                            continue
                        print('channel:', chan.split('/')[-1])
                        new_file = new_path + '/RawData/' + pos.split('/')[-2] + '_' + chan.split('/')[-1].split('.')[0] + '_' + prot + '.' + chan.split('/')[-1].split('.')[1]
                        
                        copyfile(chan, new_file)
        else:
            positions = glob(exp+'*/')
            for pos in positions:
                print('pos:', pos.split('/')[-2])
                channels = glob(pos+'*.dat')
                for chan in channels:
                    if 'cell0' not in chan:
                        continue
                    print('channel:', chan.split('/')[-1])
                    new_file = new_path + '/RawData/' + pos.split('/')[-2] + '_' + chan.split('/')[-1].split('.')[0] + '.' + chan.split('/')[-1].split('.')[1]

                    copyfile(chan, new_file)
                    
print(wang_datasets)
    
