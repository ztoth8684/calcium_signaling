import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, io
from cellpose.io import imread
from cellpose import plot
from natsort import natsorted
from glob import glob
import skimage.io
from cellpose import denoise, io
import glob
import os
import imageio

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from cellpose import models, io
# from cellpose.io import imread
# from cellpose import plot
# from natsort import natsorted
# from glob import glob
# import skimage.io
# from cellpose import denoise, io
# import glob
# import matplotlib.pyplot as plt
# import imageio
# import os

import scipy.ndimage as ndimage
# import cv2
from IPython.display import Image, display

#SET THE PATH TO THE MAX_PROJECTION_INTENSITY IMAGS, MAKING SURE THEIR NAMES ARE MASK_EVL_{NUMBER}, ALSO MAKE SURE "Layer_mask" dir is inside

path_to_files = "/content/drive/MyDrive/Test_Images/images"
# path_to_files = "AnimalView_Phalloidin488"

# #SET PATH TO OUTPUT FOLDER
output_path = os.path.join(path_to_files,'cellpose');
os.makedirs(output_path, exist_ok=True)


io.logger_setup()
files = natsorted(glob.glob(os.path.join(path_to_files, '*.tif')))
print(files)
imgs = [skimage.io.imread(f) for f in files]

# model = denoise.CellposeDenoiseModel(gpu=True, model_type="cyto3", restore_type="denoise_cyto3")
model = models.CellposeModel(gpu=True)

flow_threshold = 0.4
cellprob_threshold = 0.0
tile_norm_blocksize = 0

masks, flows, styles = model.eval(imgs, batch_size=32, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
                                  normalize={"tile_norm_blocksize": tile_norm_blocksize})

# fig = plt.figure(figsize=(12,5))
# plot.show_segmentation(fig, imgs, masks, flows[0])
# plt.tight_layout()
# plt.show()

cellpose_visuals_dir = os.path.join(output_path, "cellpose_visuals")
os.makedirs(cellpose_visuals_dir, exist_ok=True)
cellpose_gif = os.path.join(cellpose_visuals_dir, "cellpose_plot_gif.gif")

masks_dir = os.path.join(output_path, "masks")
os.makedirs(masks_dir, exist_ok=True)
masks_file = os.path.join(masks_dir, "masks.tif")
io.imsave(masks_file, masks)

filenames = []
nimg = len(imgs)
for i in range(nimg):
    fig = plt.figure(figsize=(12, 5))
    plot.show_segmentation(fig, imgs[i], masks[i], flows[i][0])
    plt.tight_layout()

    filename = f'{cellpose_visuals_dir}/plot_{i}.png'
    plt.savefig(filename)
    plt.close()
    filenames.append(filename)
    display(Image(filename))

with imageio.get_writer(cellpose_gif, mode='I', duration=0.5) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
display(Image(filename=cellpose_gif))
