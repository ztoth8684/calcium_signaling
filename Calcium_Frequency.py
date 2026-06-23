import numpy as np
import tifffile
from matplotlib import pyplot as plt
from cellpose import models
import torch
from scipy.signal import find_peaks
from skimage.measure import label, regionprops, regionprops_table
import math
from joblib import Parallel, delayed
from Calcium_Frequency_functions import *

#%%

#image file name and the timepoints to take here.
imgset = tifffile.imread('50x_2.tif')[0:1142,:,:]

#%%

data, mask, morph, size= extract_data_wloc(imgset, imgset[0,:,:])
data = poly_corr(data, 6)
peak_data, peak_locations, peak_binary, pk_height = select_spiking(data, 80)
pk_flt = []
for i in range(len(peak_locations)):
    if peak_binary[i]==1:
        pk_flt.append(peak_locations[i])

print("Fraction of cells deemed spiking")
print(len(pk_flt)/len(peak_locations))
x = get_cell_freq(pk_flt, 40)


#%%


plt.figure(figsize = (12,5));
plt.hist(1/(x*5)*1000, bins=np.arange(1,25,2));
plt.xticks(fontsize=20);
plt.yticks(fontsize=20);
print("The average frequency is: "+np.nanmean(1/(x*5)*1000))


#%%


#output the peak info as a csv file
np.savetxt('n4.csv', np.append(np.asarray([peak_binary]).T, data, axis=1), delimiter=',')
file = open('n4_peaks.txt', 'w')

for peak in peak_locations:
    file.write(' '.join([str(i) for i in peak]) + '\n')
file.close()


#%%


#snippet to remove nan data (nonspiking cells)
peak_data = peak_data[~np.isnan(peak_data).any(axis=1)]
y=[i for i in peak_locations if i[0]!='nan']

#get the peak timepoint locations for each cell
pk_flt = []
for i in range(len(peak_locations)):
    if peak_binary[i]==1:
        pk_flt.append(peak_locations[i])