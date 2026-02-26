import mahotas
import math
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance_matrix, ConvexHull
from skimage.feature import peak_local_max 
from skimage.measure import mesh_surface_area, marching_cubes_lewiner
from scipy.ndimage import distance_transform_edt

#from mayavi.mlab import triangular_mesh,mesh,outline,show
#import mayavi.mlab
"""
2. Texture features:

2a. Punctuateness/punctation: Two scalars: Number of local intensity peaks close to nucleus envelope and number of local intensity peaks in the remainder of the nucleus. Two parameters influence the feature values: 
    1) Minimal distance betweeen local intensity peaks. Note: For small values of minimal distances, multiple peaks may be detected for the same object of interest. These may need to be filtered by a suitable non-maximum suppression method.  
    2) Spatial binning, i.e., nucleus envelope support area, defined by thresholding the distance transformed image of the label map with a scalar, also discarding all non-labelled voxels (i.e., voxels not segmented as belonging to the nucleus). 
2b. 3D Haralick features: 169D-feature vector: 13 features based on co-occurency matrices for 13 different directions in 3D, derived from image signals used for segmentation. Isotropic sampling may not be required.

Omitting features discussed previously: Chromatin content (sum of DAPI channel voxel values in segmented volume), Gabor filter based features.
"""

def cellBoundingBoxes(pickledData, zAnisotropy):

    zAnisotropyReverseTransform = np.array([[1/zAnisotropy,0,0],[0,1,0],[0,0,1]])    

    starCenters_origAnisotropy = pickledData['points']
    starCenters = np.dot(pickledData['points'], zAnisotropyReverseTransform)
    print ('starCenters.shape', starCenters.shape)
    dists = pickledData['dist'] #96 entries
    print ('dists.shape', dists.shape)
    ray_verts_origAnisotropy = pickledData['rays_vertices']
    ray_verts = np.dot(pickledData['rays_vertices'], zAnisotropyReverseTransform)
    print ('ray_verts' , ray_verts.shape)
    ray_faces = pickledData['rays_faces']
    print ('ray_faces', ray_faces.shape)

    cellPosteriors = pickledData['prob']
    print ('cellPosteriors', cellPosteriors.shape, cellPosteriors)

    cellBoundingBoxesAtSrcImageScale = []

    for i in range(len(starCenters)): # for every cell
               
        cellHullPoints = starCenters[i] + np.swapaxes(dists[i] * ray_verts.T, 0, 1)
        cellHullPoints_origAnisotropy = starCenters_origAnisotropy[i] +\
            np.swapaxes(dists[i] * ray_verts_origAnisotropy.T, 0, 1)

        #Bounding boxes for feature computations
        minX = max(0, min(cellHullPoints_origAnisotropy[:,1])) #max -> suppress out of range indices
        maxX = max(cellHullPoints_origAnisotropy[:,1]) 
        minY = max(0, min(cellHullPoints_origAnisotropy[:,2]))
        maxY = max(cellHullPoints_origAnisotropy[:,2]) 
        minZ = max(0, min(cellHullPoints_origAnisotropy[:,0]))
        maxZ = max(cellHullPoints_origAnisotropy[:,0]) 
       
        cellBoundingBoxesAtSrcImageScale.append([int(minX),int(minY),int(minZ),\
            int(maxX),int(maxY),int(maxZ)])
    
    return cellBoundingBoxesAtSrcImageScale


#Compute spatial histrogram of punctatness (local intensity peaks) of single nucleus, guided by mask image
def _punctationSpatialHistogram(peaks, labelMask):
    numBins = 4
    dTIm = distance_transform_edt(labelMask)
    maxDist = np.amax(dTIm)
    ###print('MAXDIST', maxDist)
    spatialHist = np.zeros(numBins, dtype=np.uint16)

    for peak in peaks:    
        distanceFromNuclEnv = dTIm[peak[0],peak[1],peak[2]]
        if distanceFromNuclEnv > 0:
            spatialBin =  min(numBins-1, int(distanceFromNuclEnv / maxDist * numBins)) 
            ##print('sbin', spatialBin)
            spatialHist[spatialBin] = spatialHist[spatialBin] + 1
    
    ###print ("sHist", maxDist, spatialHist)
    return spatialHist


def computeTextureFeatures(cellBoundingBoxesAtSrcImageScale, imTexture, imLabels, includeHaralickFeatures=True, verboseLog = True):
    punctuatenessFeatures = []
    haralickFeatures = []
    overview = np.zeros(imTexture.shape, dtype=np.uint16)
    if verboseLog:
        print("shape of label patches:",imLabels.shape)
    imMaxZ, imMaxY, imMaxX = imLabels.shape[0], imLabels.shape[1], imLabels.shape[2] # # Indices: ZYX or ZXY ?
        
    progressBar = tqdm(total=len(cellBoundingBoxesAtSrcImageScale), desc="Texture features computed", \
                            bar_format="{l_bar}{bar} [ time left: {remaining} ]")
       
    for i,cellBox in enumerate(cellBoundingBoxesAtSrcImageScale):
        progressBar.update(1)
        minX,minY,minZ = cellBox[0], cellBox[1], cellBox[2]
        maxX,maxY,maxZ = cellBox[3], cellBox[4], cellBox[5] 
        
        # # Add border (of size 1) to all dimensions for correctly extracting all local intensity maxima.
        overview[minZ+1:maxZ+1,minX:maxX,minY:maxY] = 55000 +i       # # +1 only in z-coords?
        ####imsave('oTst/cellPatchesTest'+ str(i).zfill(4) +'.tif', imTexture[minZ+1:maxZ+1,minX:maxX,minY:maxY],\
        ##### photometric='minisblack')        
        
        # Produce masks for cell image patches, which wipe out margins not belonging to nucleus 
        # or belong to a different cell than the one delineated by bounding box.
        # Implementation: Get pixel value of center of label map patch. For all pixels in label patch
        # not having same value, set cooresponding pixel (at same coords) in texture patch to zero.
        # Zero values suppress co-occurencey based feature computation in mahotas lib routines. 

        # Use slightly larger patch than bbox to facilitate correct extraction of local intensity maxima 
        # - does not affect Haralick features
        labelPatchBorderWidth = 1
        minX, minY, minZ = max(0,minX-labelPatchBorderWidth), \
                                 max(0,minY-labelPatchBorderWidth), max(0,minZ-labelPatchBorderWidth)
        maxX, maxY, maxZ = min(imMaxX,maxX+labelPatchBorderWidth), \
                                 min(imMaxY,maxY+labelPatchBorderWidth), min(imMaxZ,maxZ+labelPatchBorderWidth)
        
        stardistLabelValue = imLabels[(maxZ+minZ)//2,(maxX+minX)//2,(maxY+minY)//2]
        labelBlock = imLabels[minZ+1:maxZ+1,minX:maxX,minY:maxY]
        labelMask = (labelBlock == stardistLabelValue)
        texBlock = imTexture[minZ+1:maxZ+1,minX:maxX,minY:maxY]
        
        #croppedTextureMap = texBlock if labelMask else 0
        croppedTextureMap = np.where(labelMask>0,texBlock * 4096, labelMask).astype(np.uint16) 
        # # Conversion to float32 by default, requiring the 2**12 factor ? 
        # # Does this normalize the signal in any way that could impact validity of feature values?
        
        ##imsave('oTst/cropCellTest'+ str(i).zfill(4) +'.tif', croppedTextureMap, photometric='minisblack')  
        #imsave('oTst/cropCellTest'+ str(i).zfill(4) +'.tif', imTexture[minZ+1:maxZ+1,minX:maxX,minY:maxY], \
        #            photometric='minisblack')  

        if includeHaralickFeatures:                
            # Haralick: 28s for 4300 patches on 12 Xeon cores
            hf = mahotas.features.haralick((croppedTextureMap / 256).astype(np.uint8) ) 
            # #       Leads to value error for empty input: , ignore_zeros=True)            
            haralickFeatures.append(np.ndarray.tolist(hf))
            # https://mahotas.readthedocs.io/en/latest/api.html#mahotas.features.haralick        
                
        # # Should peaks be extracted on uncropped Texture map, and then be filtered again? 
        # # This makes sense as part of constructing spatial histogram.
        peaks = peak_local_max(texBlock, min_distance=1, threshold_abs=None, threshold_rel=None, \
                                    exclude_border=True, indices=True) 
                             #, num_peaks=inf, footprint=None, labels=None, num_peaks_per_label=inf, p_norm=inf)
                
        punctuatenessFeatures.append(np.ndarray.tolist(_punctationSpatialHistogram(peaks, labelMask)))
        
    ####imsave('oTst/_cellPatchesTestOverview.tif', overview, photometric='minisblack')                    
        
    return haralickFeatures, punctuatenessFeatures


def getLabelMapVolumeAndSurface(voxelMap,zAnisotropy):
    vms = voxelMap.shape
    stardistLabelValue = voxelMap[vms[0]//2,vms[1]//2,vms[2]//2]
    labelMask = (voxelMap == stardistLabelValue)

    verts, faces, _ , _ = marching_cubes_lewiner(labelMask, level=None, spacing=(1/zAnisotropy, 1.0, 1.0))#,\
         #gradient_direction='descent', step_size=1, allow_degenerate=True, use_classic=False, mask=None)   
    surfaceArea = mesh_surface_area(verts, faces)
    """
    # # Debug visualization:
    if surfaceArea > 2000:                
        mayavi.mlab.triangular_mesh(verts[:,0],verts[:,1],verts[:,2],faces)        
        #xx, yy, zz = np.where(labelMask > 0)
        #mayavi.mlab.points3d(xx, yy, zz, mode="cube", color=(0, 1, 0), scale_factor=1)
        mayavi.mlab.outline()
        mayavi.mlab.show()
    """
    voxelVolume = np.count_nonzero(labelMask) / zAnisotropy
    return voxelVolume, surfaceArea


# IrregalurityScores derived from label maps (and thus in textureFeatures module):
# Compute a score indicating morphological irregulariy of nucleus, similar to form factor, 
# but adjusted to elongation:
#                       labelMapNucleusSurface ^^ (1/2) / labelMapNucleusVolume ^^ (1/3) 
# irregularityScore =   -------------------------------------------------------------------
#                       fittedEllipsoidSurface ^^ (1/2) / fittedEllipsoidVolume ^^ (1/3) 
# Larger values indicate higher degree of irragularity. Values close to 1 indicate regular shape.
def computeNucleusIrregalurityScores(cellBoundingBoxesAtSrcImageScale, imLabels, nucleusOrientations,\
                                        zAnisotropy, verboseLog = True):
    def euclLength(v):
        return math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
    irregularityScores = []    
    p, oneOverP = 1.6075, 1/1.6075 # For Knud Thomsen's formula for ellipsoid surface: relative error of at most 1.061%    
    progressBar = tqdm(total=len(cellBoundingBoxesAtSrcImageScale), desc="Irregularity scores computed", \
                            bar_format="{l_bar}{bar} [ time left: {remaining} ]")
    for i,cellBox in enumerate(cellBoundingBoxesAtSrcImageScale):
        progressBar.update(1)    
        orientationTriad = nucleusOrientations[i]
        a,b,c = euclLength(orientationTriad[0]),euclLength(orientationTriad[1]),euclLength(orientationTriad[2])
        ellipsoidVolume = (4/3)*np.pi*a*b*c*8 # a,b,c - ellipsoid SEMI-axes
        aP,bP,cP = a**p,b**p,c**p
        ellipsoidSurface = 4*np.pi* ((aP*bP+aP*cP+bP*cP)/3)**oneOverP # Knud Thomsen's formula

        minX,minY,minZ = cellBox[0], cellBox[1], cellBox[2]
        maxX,maxY,maxZ = cellBox[3], cellBox[4], cellBox[5] 
        cellboxLabels = imLabels[minZ:maxZ+2,minX:maxX+2,minY:maxY+2] # Indices confirmed by mayavi inspection
        
        labelMapVolume,labelMapSurface = getLabelMapVolumeAndSurface(cellboxLabels,zAnisotropy)        
        irregularityScore = (labelMapSurface/ellipsoidSurface)**0.5 * (ellipsoidVolume/labelMapVolume)**0.333
        irregularityScores.append(irregularityScore)
    
    return irregularityScores

