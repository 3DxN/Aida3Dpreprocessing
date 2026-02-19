'''Reads a 3d label image, outputs a Startdist3D pickle containing star-convex geometry'''
import numpy as np
import json
import pickle
import sys

from glob import glob
from tqdm import tqdm
from tifffile import imread
from termcolor import colored

from stardist import relabel_image_stardist3D, Rays_GoldenSpiral, calculate_extents
from stardist import fill_label_holes, random_label_cmap
from stardist import matching 
from stardist.geometry import geom3d

from skimage import measure

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
    print('Processing image tile:', imFile)

    #lbl = imread(imFile)
    imageData = np.load(imFile, allow_pickle=True).item()
    flows = imageData['flows']
    masks = imageData['masks'] 
    lbl = masks
    probabilitesImage = flows[1].astype(np.uint8)

    matching._check_label_array(lbl, "lbl")
    if not lbl.ndim==3:
        raise ValueError("Label image should be 3 dimensional.")

    dist_all = geom3d.star_dist3D(lbl, rays)
    regs = measure.regionprops(lbl)

    points = np.array(tuple(np.array(r.centroid).astype(int) for r in regs))
    labels = np.array(tuple(r.label for r in regs))
    dist = np.array(tuple(dist_all[p[0], p[1], p[2]] for p in points))
    dist = np.maximum(dist, 1e-3)

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

    pickle.dump( nucleusGeometry, open( imFile+'.pickle', "wb" ) )
