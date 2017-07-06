#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute tsne for information bottleneck
(for python 3)

- USER: specify folder path, perplexity and outputfile
- identifies list of files in the specified folder
(here: CrossingSegments)
- loads local linear models: number of models x 4 x 4 (coupling of 4 modes)
(note: [:,0,:]-offset, [:,5:9,:]-variation)
- stitches all the data points together but keeps track of how many models
belong to each sequence
- performs tsne with specified complexity value (cannot be run in ipython)
- h5file as output with
entrynumber = number of linear models in each file
allthetas = all linear models stitched together sum(entynumber)x16
space = linear models in tsne space

based on: ClusteringForInfoBottleneck.py
Created on Wed Jul 05 2017
@author: SWerner
"""

import os
import numpy as np
import h5py
import sys
sys.path.append('./bhtsne-master')
import bhtsne


# -- USER INPUT: data folder, perplexity of tsne, output file
inputfolder='CrossingSegments'
perplexity=50 #typical values 45, 50, 55
out_file='TsneResults/Perp'+str(perplexity)+'.h5'
# -- END


#identify list of h5 files
h5filelist=os.listdir(inputfolder)
print('Number of files: '+str(len(h5filelist)))

user_input='n'
if os.path.isfile(out_file):
    print('Output file already exists.')
    user_input = input('Do you want to replace file (y/n): ')
else:
    user_input='y'

if user_input=='y':
    #load h5 file
    entrynumber=np.empty(len(h5filelist)) * np.nan
    for i in range(len(h5filelist)):
        f=h5py.File(inputfolder + '/' + h5filelist[i],'r') #r - read only
        loadedthetas=np.array(f['thetas'])
        #matrix with models x 16 entries
        #reshape attaches [0,0,:],[0,1,:],...
        usethetas=np.reshape(loadedthetas[:,1:5,:],
                             (loadedthetas.shape[0],16,1))[:,:,0]
        entrynumber[i]=usethetas.shape[0]; #number of linear models in file
        if i==0:
            allthetas=usethetas
        else: #stitches all together
            allthetas=np.vstack((allthetas, usethetas))
    print('Size of data: '+str(allthetas.shape))
    
    #perform tsne algorithm
    allthetas=allthetas.astype('float64')
    pca_dim=14
    space=bhtsne.run_bh_tsne(allthetas,verbose=True,perplexity=perplexity,
                             initial_dims=pca_dim,max_iter=5000)

    #Save the result
    f=h5py.File(out_file,'w')
    entrynumbers=f.create_dataset('entrynumber',(entrynumber.shape))
    entrynumbers[...]=entrynumber
    alltheta=f.create_dataset('allthetas',(allthetas.shape))
    alltheta[...]=allthetas
    tsne_s=f.create_dataset('space',(len(allthetas),2),maxshape=(None,None))
    tsne_s.resize(space.shape)
    tsne_s[...]=space
    f.close()