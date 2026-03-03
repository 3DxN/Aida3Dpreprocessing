'''Reads a 3d label image, outputs a Startdist3D pickle containing star-convex geometry'''
import numpy as np
import json
import pickle
import sys

from glob import glob
from skimage import measure
from termcolor import colored
from time import time

from stardistGeo import Rays_GoldenSpiral
from stardistGeo import _check_label_array #matching 
from stardistGeo import star_dist3D #geom3d

print('Converting LABEL image to starconvex geometry...')

if len(sys.argv) < 3:
    print ('Only', len(sys.argv), 'arguments provided!')
    print ('Usage: python' , sys.argv[0], 'dataDirectory labelMapAnisotropy')
    print ('Or: python' , sys.argv[0], 'dataDirectory 0 [path to tileArangements.json file]')    
    sys.exit(1)

directory = sys.argv[1]
voxelAnisotropy = float(sys.argv[2])


if not voxelAnisotropy > 0.0:
    if len(sys.argv) < 4:
        print('No valid anisotropy value in CLI params and no tileArangements.json file provided.')
        sys.exit(1)
    tileArangmentsfile = sys.argv[3]
    
    with open(tileArangmentsfile, 'r') as tF:
        tileFile=tF.read()
    try:
        tileDataDict = json.loads(tileFile)
    except ValueError: 
        print(colored('Value error in tileFile ' + tileArangmentsfile,'red'))
        sys.exit(1)            
    except:
        print(colored('Error: Cannot decode tileFile ' + tileArangmentsfile,'red'))
        sys.exit(1)    
    
    if not 'voxelDimensionsTIFFFiles' in tileDataDict:
        print('Key voxelDimensionsTIFFFiles not available!')
        sys.exit(1)
    segmentationDataVoxelDimensions = tileDataDict['voxelDimensionsTIFFFiles']
    svx,svy,svz = segmentationDataVoxelDimensions[0], segmentationDataVoxelDimensions[1], segmentationDataVoxelDimensions[2]
    voxelAnisotropy = (svx/2.0 + svy/2.0) / svz
    
print('Using voxel anisotropy', voxelAnisotropy)

numRays = 96
anisotropy = (1.0/voxelAnisotropy,1.0,1.0) # # check anisotropy matches correct image axes
rays = Rays_GoldenSpiral(numRays, anisotropy=anisotropy)

#imFiles = glob(directory+'/*.tif')
imFiles = glob(directory+'/*.npy')

for imFile in imFiles:
    print(f'Processing image tile: {imFile}', end= ' ', flush=True)

    #lbl = imread(imFile)
    imageData = np.load(imFile, allow_pickle=True).item()
    flows = imageData['flows']
    masks = imageData['masks'] 
    lbl = masks
    probabilitesImage = flows[1].astype(np.uint8)

    print('.',end= '', flush=True)
    _check_label_array(lbl, "lbl")
    if not lbl.ndim==3:
        raise ValueError("Label image should be 3 dimensional.")

    time1=time()
    dist_all = star_dist3D(lbl, rays)
    print(f'.(stardist3dTime={time()-time1})',end= '', flush=True)
    regs = measure.regionprops(lbl)

    points = np.array(tuple(np.array(r.centroid).astype(int) for r in regs))
    labels = np.array(tuple(r.label for r in regs))
    dist = np.array(tuple(dist_all[p[0], p[1], p[2]] for p in points))
    dist = np.maximum(dist, 1e-3)
    print('.',end= '', flush=True)

    # Picking single point from probability image. 
    # Can be improved by sampling more points and averaging, or averaging over the entire label image.
    nucleusProbabilites = []
    for p in points:
        xx,yy,zz = p[0],p[1],p[2]
        prob = probabilitesImage[xx,yy,zz] /255.0 # 255 - max value by cellpose definition
        nucleusProbabilites.append(prob)

    nucleusGeometry = {}
    nucleusGeometry['points'] = points
    nucleusGeometry['rays_vertices'] = rays.vertices
    nucleusGeometry['dist'] = dist
    nucleusGeometry['rays_faces'] = rays.faces
    nucleusGeometry['prob'] = np.array([p for p in nucleusProbabilites])
    print('.',end= '', flush=True)
    pickle.dump( nucleusGeometry, open( imFile+'.pickle', "wb" ) )  #TODO: replace with JSON; fix file name (currently `tile__H000_V001_CELLPOSE-LABELS_seg.npy.pickle`)!
    print(' ', flush=True)
