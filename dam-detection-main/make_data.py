import numpy as np
import tifffile as tiff
import os
import matplotlib.pyplot as plt
import pickle

#Takes folders with .tifs and creates test and train folders of .npys

#to store npy images:
im_direct = './images/'
testim_direct = './images/test_images/'
trainim_direct = './images/train_images/'

#to load tifs:
base_path = './NAIP/'
#folders representing different image locations within the base_path:
folders = ['CHIPS_01_Wyoming_BeaverDam_Lengths','CHIPS_02_WyomingDams','CHIPS_03_OregonDams_shp','CHIPS_04_IdahoDams_shp','CHIPS_05_ColoradoDams_shp','CHIPS_06_CaliforniaDams_shp']

train_dict = {}
test_dict = {}

labels = []
train_ims = []
test_ims = []

ind = -1
for f in folders:
    folder_path = os.path.join(base_path,f)
    files = os.listdir(folder_path)
    for im_file in files:
         ind+=1
         imnum = "%06d" % ind
         im = tiff.imread(folder_path+'/'+im_file)
         if np.sum(np.isnan(im.flatten())) > 0:
             raise Exception('NaN found in: '+f+'/'+im_file)
         
         if 'no' in im_file:
             labels.append(0); 
         elif 'yes' in im_file:
             labels.append(1); 
         else:
             raise Exception('Incorrect file name')
         
         if np.random.rand() <= .1:
             test_dict[ind] = f+'/'+im_file 
             test_ims.append(ind)
             np.save(testim_direct+'im'+imnum+'_l'+str(labels[-1])+'.npy',im)

         else:
             train_dict[ind] = f+'/'+im_file 
             train_ims.append(ind)
             np.save(trainim_direct+'im'+imnum+'_l'+str(labels[-1])+'.npy',im)
             


print(len(test_dict),len(train_dict))              

with open(im_direct+'train_dictionary.pkl', 'wb') as f:
    pickle.dump(train_dict, f)
with open(im_direct+'test_dictionary.pkl', 'wb') as f:
    pickle.dump(test_dict, f)
    
np.save(im_direct+'testims_inds.npy',np.array(test_ims))
np.save(im_direct+'trainims_inds.npy',np.array(train_ims))
np.save(im_direct+'labels.npy',np.array(labels))
    

        


