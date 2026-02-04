from __future__ import print_function, unicode_literals, absolute_import, division
import json
import os
import pickle
import sys
import numpy as np

from os.path import exists
from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible
from stardist.models import StarDist3D

# Batch processing: Predict segmentation for each 3D TIFF file found.
# Outputs label images and star-convex shapes as Python pickle.
# Uses StarDist3d. Details: https://stardist.net/docs/faq.html
# inDir points to dir of TIFF files to be segmented
# outDir where label maps and geometry is stored, same base file names as input files.
# tileJson contains the list of files to be segmented, and a flag of whether segmentation should be attempted. Empty images are excluded from processing.
#
# For label files already existing in outDir, no segmentation is computed.

# Call, e.g.,:
# python predBatch.py /media/rr/Seagate\ Expansion\ Drive/_out/Sec10/TIFFtiles /media/rr/Seagate\ Expansion\ Drive/_out/Sec10/segmentation /media/rr/Seagate\ Expansion\ Drive/_out/Sec10/tileArrangement.json

# NOTE: Anisotropy and scale of image data must fit model. Model needs to be registered with StarDist3D and then can be selected with CLI parameter 4.

### Not supporting piping of input file names as follows:
### echo $(ls tile__H{005..006}_V{023..024}.tif)
### ls /media/rr/Seagate\ Expansion\ Drive/_out/Sec10/TIFFtiles/tile__H01{0,1}_V015.tif  | python predBatch.py 

if len(sys.argv) < 4:
    print ('Only', len(sys.argv), 'arguments provided!')
    print ('Usage: python' , sys.argv[0], ' inDir outDir tileJsonFilename segmentationModelName(optional)')
    sys.exit(1)

inDir = sys.argv[1]
outDir = sys.argv[2]
tileJsonFilename = sys.argv[3]

if not os.path.isdir(outDir):
    os.mkdir(outDir)

tileData = open(tileJsonFilename)
   
tileDataDict = json.load(tileData)

if len(sys.argv) < 5:
    model = StarDist3D.from_pretrained('3D_demo')
else:
    model = StarDist3D(None, name=sys.argv[4], basedir='models')
    # For further parameters, see: https://github.com/stardist/stardist/blob/810dec4727e8e8bf05bd9620f91a3a0dd70de289/stardist/models/model3d.py#L267
print ('StarDist3D segmentation, using model', model)    
   
# Iterating through non-empty tiles in files specified in tileArrangement.json
for imageFileName, isNotEmpty in zip(tileDataDict['tiff3DFilesForSegmentation'], tileDataDict['tileNotEmpty']):
    print('Processing image: ', imageFileName, end = ' ')
   
    if not isNotEmpty:
        print('  --> Image is empty - skipping ...')
        continue
   
    inFileImagePath = os.path.join(inDir, imageFileName)
    outFileLabelImagePath = os.path.join(outDir,imageFileName + '_LABELS.tif') 
    if exists(outFileLabelImagePath):
        print('  --> Segmentation exists. Skipping ...')
        continue           

    image = imread(inFileImagePath)
    
    maxGV = np.max(image)  
    minGV = np.min(image)    
    print ("Min/ Max gv: " , minGV, '/',maxGV)
    if maxGV - minGV < 1000:
        print("Image likely empty --> skipping")
        continue

    n_channel = 1 if image[0].ndim == 3 else image[0].shape[-1]
    axis_norm = (0,1,2)   # normalize channels independently
    # axis_norm = (0,1,2,3) # normalize channels jointly
    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
        

    print ("Image SHAPE: ", image.shape)
    imgTile = normalize(image, 1,99.8, axis=axis_norm)

    try:
        labels, details = model.predict_instances(imgTile, n_tiles=[1,1,1], show_tile_progress=True)
        # For very large images, it may be necessary to predict segmentation on sub-tiles and merge results.
        # Avoid if possible, as this takes significantly more time. 
        # Parameter min_overlap must be chosen at least as large as the larges object (nucleus) to segment.
        #labels, details = model.predict_instances_big(img, axes='ZYX', block_size=4096, min_overlap=128, context=128, n_tiles=(4,4,1))
    
    except: #Should catch std::bad_alloc from C++ env   
        print('Encountered memory allocation problems. Image might be mal-formed or empty. --> Skipping')
        continue
        # Retry on same image with finer partitioning (n_tiles parameter)?
           
    outFileGeometryDataPath = os.path.join(outDir,imageFileName + '_.pickle')
    with open(outFileGeometryDataPath, 'wb') as f:
        pickle.dump(details, f)
                
    ###save_tiff_imagej_compatible(fileName+'_SRC.tif', imgTile, axes='ZYX')
    save_tiff_imagej_compatible(outFileLabelImagePath, labels, axes='ZYX')
    #np.savetxt('stardist3Ddists.out', details['dist'], delimiter=',')

tileData.close()
