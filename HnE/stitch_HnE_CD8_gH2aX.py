
# Scanning CD8 and gammaH2aX blocks (3D) to get z-layer with maximum surface across both. Used for visualisation in AIDA-3D. Corresponding H&E layers will be extracted at this z-layer.

from pathlib import Path
from glob import glob
import tifffile
import numpy as np

inputPath = Path('..')/'..'/'extracted'

EXTRACTED_Z_LAYER=11
AIDA_TILE_SIZE=302
TILE_SIZE=682
AIDA_GAP_SIZE=25  # This may not be required
AIDA_GAP_SIZE_PRE_SCALE =int( 1.0 * TILE_SIZE / AIDA_TILE_SIZE * AIDA_GAP_SIZE )

GAP = AIDA_GAP_SIZE_PRE_SCALE # *0 # may use negative numbers to reduce overlap for stitching (one wins)

MAX_COL=3
MAX_ROW=3

IMG_WIDTH_STITCHED = MAX_COL * (TILE_SIZE + GAP) - GAP
IMG_HEIGHT_STITCHED = MAX_ROW * (TILE_SIZE + GAP) - GAP



# File name format: 
# tile__H000_V000.tif_CELLPOSE-LABELS_cp_masks_cd8.tif     
# tile__H002_V000.tif_CELLPOSE-LABELS_cp_masks_gh2ax.tif

stitched_hne = 255 * np.ones((IMG_HEIGHT_STITCHED,IMG_HEIGHT_STITCHED,4),dtype=np.uint8) 
stitched_cd8 = np.zeros((IMG_HEIGHT_STITCHED,IMG_HEIGHT_STITCHED,4),dtype=np.uint8) 
stitched_gH2aX = np.zeros((IMG_HEIGHT_STITCHED,IMG_HEIGHT_STITCHED,4),dtype=np.uint8)

for row in range(0,MAX_ROW):
    for col in range(0,MAX_COL):
        hne_slice = tifffile.imread(f'pseudoHnE_COLD_{col}_ROW_{row}.tif')
        print(row, col, hne_slice.shape)

        c_start = col*(TILE_SIZE+GAP)
        c_end = (col+1)*(TILE_SIZE+GAP)-GAP
        r_start = row*(TILE_SIZE+GAP)
        r_end = (row+1)*(TILE_SIZE+GAP)-GAP

        stitched_hne[c_start:c_end,r_start:r_end,0:3] = hne_slice
        stitched_hne[:,:,3] =255 #Alpha channel

        cd8_slice = tifffile.imread(f'{inputPath}/tile__H00{row}_V00{col}.tif_CELLPOSE-LABELS_cp_masks_cd8.tif')[EXTRACTED_Z_LAYER]
        gH2aX_slice = tifffile.imread(f'{inputPath}/tile__H00{row}_V00{col}.tif_CELLPOSE-LABELS_cp_masks_gh2ax.tif')[EXTRACTED_Z_LAYER]
        
        stitched_cd8[c_start:c_end,r_start:r_end,0] = 255*(cd8_slice > 0) #red
        stitched_gH2aX[c_start:c_end,r_start:r_end,1] = 255*(gH2aX_slice > 0) # green
        stitched_gH2aX[c_start:c_end,r_start:r_end,0] = 255*(gH2aX_slice > 0) # red
        stitched_cd8[c_start:c_end,r_start:r_end,3]=(cd8_slice>0)*255 # alpha
        stitched_gH2aX[c_start:c_end,r_start:r_end,3]=(gH2aX_slice>0)*255 # alpha

tifffile.imwrite('stitched_gH2aX.tif', stitched_gH2aX)
tifffile.imwrite('stitched_cd8.tif', stitched_cd8)

tifffile.imwrite('stitched_hne.tif', stitched_hne)

