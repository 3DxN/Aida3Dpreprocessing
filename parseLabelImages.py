# SPDX-License-Identifier: MIT

# Scanning CD8 and gammaH2aX blocks (3D) to get z-layer with maximum surface across both. Used for visualisation in AIDA-3D. Corresponding H&E layers will be extracted at this z-layer.

from pathlib import Path
from glob import glob
import tifffile
import numpy as np

inputPath = Path('..')/'extracted'

cd8Files = glob(f'{inputPath}/*cd8.tif')
gammaH2aXFiles = glob(f'{inputPath}/*gh2ax.tif')

print("Input path: ", inputPath)

print("Number of CD8 files: ", len(cd8Files))
print("Number of gammaH2aX files: ", len(gammaH2aXFiles))

cd8_zSum = []
for cd8File in cd8Files:
    cd8array = tifffile.imread(cd8File)
    cd8_zSum.append( np.sum((cd8array>0),axis=(1,2)))
z_cd8 = np.sum(np.asarray(cd8_zSum),axis=0)
z_cd8_total = np.sum(z_cd8)
print(z_cd8_total, z_cd8)
print()

gH2aX_zSum = []
for gH2aXFile in gammaH2aXFiles:
    gH2aXarray = tifffile.imread(gH2aXFile)
    gH2aX_zSum.append( np.sum((gH2aXarray>0),axis=(1,2)))
z_gH2aX = np.sum(np.asarray(gH2aX_zSum),axis=0)
z_gH2aX_total = np.sum(z_gH2aX)
print(z_gH2aX_total, z_gH2aX)

weightedAvg = (z_gH2aX_total * z_gH2aX + z_cd8_total* z_cd8)/(z_gH2aX_total + z_cd8_total)

print("Average max of cd8 and gh2ax: ", weightedAvg)
print("Max z-index: ", weightedAvg.argmax())
