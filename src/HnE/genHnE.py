# SPDX-License-Identifier: MIT
import sys
import os
import math
import json
import argparse
import numpy as np
import time
from pathlib import Path
from termcolor import colored
from tifffile import imwrite as imsave # fix legacy code
from skimage import exposure
from skimage.transform import rescale
from skimage.morphology import remove_small_holes
from skimage.morphology import remove_small_objects
from scipy import ndimage
from tqdm import tqdm
import xml.etree.ElementTree as ET


# NOTE: Bioformats Java classnotfound_exceptions (which clutter std::out) do not affect Imaris data reading
# See: https://www.openmicroscopy.org/community/viewtopic.php?p=18686
# Potential fix to remove std::out clutter: 
# Set WARN->INFO in https://github.com/ome/bioformats/blob/v5.7.1/tools/logback.xml

"""
genHnE.py
Tool to process Imaris image stacks. Generates:
    - Pseudo H&E images from substacks of nuclear and cyto channels (automatically selected confocal layers), stored as 3D TIFF where each tile corresponds to an Imaris 3D tile.
    - Images of Nuclear and cyto-channels used for generating the H&E above.
    - Masks (2D) of non-empty image regions. May be used for excluding empty regions when computing embeddings for tissue patch clustering.
    - 3D TIFF files at specified resolution and anisotropy, used for segmentation and texture feature computation.
"""


# Color definition for pseudo HnE stain
HnE_COLORS_4DAPI = [0.17, 0.27, 0.105]
HnE_COLORS_4AUTOFLUORESCENCE = [0.05, 1.0, 0.54]


def confocalStackToMask(imarisFileReader, resolutionLevel, maskThresh = 10, cytoChannelIdx = 0, nuclearChannelIdx = 1):

    imarisImageXdim, imarisImageYdim, imarisImageZdim = imarisFileReader.get_3Ddimensions(resolutionLevel)
    imageTileIn = np.empty((imarisImageZdim, 2, imarisImageYdim, imarisImageXdim), dtype = np.uint16)
    maskTile = np.empty((imarisImageYdim, imarisImageXdim)) 
    
    for confocalSliceIdx in range(imarisImageZdim):
        tile_size = (imarisImageXdim, imarisImageYdim)
        confocalSliceChannelCYTO = imarisFileReader.read_region( 
            (0,0), confocalSliceIdx, cytoChannelIdx, resolutionLevel, tile_size)            
        confocalSliceChannelNUCL = imarisFileReader.read_region( 
            (0,0), confocalSliceIdx, nuclearChannelIdx, resolutionLevel, tile_size)
        # # Param 4096: Is this conversion beneficial?
        imageTileIn[confocalSliceIdx,0,:,:] = np.uint16(4096 * confocalSliceChannelCYTO) 
        imageTileIn[confocalSliceIdx,1,:,:] = np.uint16(4096 * confocalSliceChannelNUCL) 
    
    imageIn = np.array([imageTileIn])  # Wrap for TIFF <-> IMS compatibility

    maxIntensityProjAUTOFL = np.max(imageTileIn[:,0,:,:], axis=0) # Axis 0 for IMS tile, axis 1 for 5D TIFF
    maxIntensityProjDAPI = np.max(imageTileIn[:,1,:,:], axis=0) 

    #Binarization convention:  0 means no signal is present, and > 0 means there is
    maskTile =  (maxIntensityProjDAPI > maskThresh)
    
    # # Consider making min_size/area_threshold a function of tile size. For tiles only 32x32, 100 may be too much.
    # Prevent overextension of tissue areas
    # remove_small_holes(maskTile, area_threshold=10, connectivity=1, in_place=False) #maskTile
    return 255 * remove_small_holes(remove_small_objects(maskTile, min_size= 100), area_threshold=100)

       
def getMaxZIntensity(imarisImageReader, zSelectionResLevel, nuclearChannelIdx = 1):

    imarisImageXdim, imarisImageYdim, imarisImageZdim = imarisImageReader.get_3Ddimensions(zSelectionResLevel)
    imageTileIn = np.empty((imarisImageZdim, imarisImageYdim, imarisImageXdim), dtype = np.uint16)    
    for confocalSliceIdx in range(imarisImageZdim):
        tile_size = (imarisImageXdim, imarisImageYdim)
        confocalSliceChannelNUCL = imarisImageReader.read_region( 
            (0,0), confocalSliceIdx, nuclearChannelIdx, zSelectionResLevel, tile_size)
        imageTileIn[confocalSliceIdx,:,:] = np.uint16(4096 * confocalSliceChannelNUCL) 

    # Sum of pixel grayvalues for each z-layer, to obtain maximum position over the z channel.
    xySums = np.sum(np.sum(imageTileIn, axis=2), axis=1)
    zcoordOfMaxIntensity = np.argmax(xySums, axis=0)  
    print ( colored('ZMAXcoord: ' + str(zcoordOfMaxIntensity), 'green')  )
    return zcoordOfMaxIntensity


def confocalStackToHNE(imarisFileReader, resolutionLevel, zMin, zMax, TIFFwriteoutArgs, tileCol, tileRow, colorWeight_HEMAT, colorWeight_EOSIN, cytoChannelIdx = 0, nuclearChannelIdx = 1):
   
    imarisImageXdim, imarisImageYdim, imarisImageZdim = imarisFileReader.get_3Ddimensions(resolutionLevel)
    imageTileIn = np.empty((imarisImageZdim, 2, imarisImageYdim, imarisImageXdim), dtype = np.uint16)
    RGB_image = np.empty((3, imarisImageYdim, imarisImageXdim)) 
    
    tic1 = time.perf_counter()

    # For iterating through the confocal slices, the aggreation for HnE rendering could 
    # be done with range(zMin, zMax). However, this does not cover data required for 3D
    # segmentation, and also does not result in a major speedup, as operations are I/O
    # limited, and Imaris image chunks have a 3D layout.
    for confocalSliceIdx in range(0, imarisImageZdim):
        tile_size = (imarisImageXdim, imarisImageYdim)
        confocalSliceChannel0 = imarisFileReader.read_region( \
            (0,0), confocalSliceIdx, cytoChannelIdx, resolutionLevel, tile_size)            
        confocalSliceChannel1 = imarisFileReader.read_region( \
            (0,0), confocalSliceIdx, nuclearChannelIdx, resolutionLevel, tile_size)
        # # Param 4096: Is this conversion beneficial?
        imageTileIn[confocalSliceIdx,0,:,:] = np.uint16(4096* confocalSliceChannel0) 
        imageTileIn[confocalSliceIdx,1,:,:] = np.uint16(4096 * confocalSliceChannel1) 
        
    imageIn = np.array([imageTileIn])  # For IMS compatibility
    tic2 = time.perf_counter()
    
    subSlice = imageTileIn[zMin:zMax,:,:,:] if (zMax>zMin) else imageTileIn[zMin:zMin+1,:,:,:] # else: case thin slices

    # Maximum intensity projection, contrast normalised, for DAPI and autofluoresence channel.
    # For more contrast adjustment options,
    # see https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html
    maxIntensityProjAUTOFL = np.max(subSlice[:,0,:,:], axis=0) # Axis 0 for IMS tile, axis 1 for 5D TIFF
    maxIntensityProjDAPI = np.max(subSlice[:,1,:,:], axis=0) 
    p2AUTOFL, p98AUTOFL = np.percentile(maxIntensityProjAUTOFL, (2, 98))   		   
    p2DAPI, p98DAPI = np.percentile(maxIntensityProjDAPI, (2, 98))
     
    contrastAdjustedAUTOFLUORESCENCE = exposure.rescale_intensity(maxIntensityProjAUTOFL, in_range=(p2AUTOFL, p98AUTOFL))    
    contrastAdjustedDAPI = exposure.rescale_intensity(maxIntensityProjDAPI, in_range=(p2DAPI, p98DAPI))
 	    
    # Compose color image tile
    maxGray = 16384 # if (slices5um.dtype == np.uint16) else 256  # # May be derived from image signal, but better know before scanning entire dataset. Metadata available?

    for rgbChannel in range(RGB_image.shape[0]):
        tmp_AUTOFL = HnE_COLORS_4AUTOFLUORESCENCE[rgbChannel] * colorWeight_EOSIN \
            * contrastAdjustedAUTOFLUORESCENCE / maxGray 
        tmp_DAPI = HnE_COLORS_4DAPI[rgbChannel] * colorWeight_HEMAT \
            * contrastAdjustedDAPI / maxGray 
        RGB_image[rgbChannel] = 255 * np.multiply(np.exp(-tmp_AUTOFL), np.exp(-tmp_DAPI))
               
    RGB_singleTile = np.moveaxis(RGB_image, 0, -1).astype(np.uint8)

    toc = time.perf_counter()

    print(colored(f"Read image data in         {tic2 - tic1:0.4f} seconds", 'green'))
    print(colored(f"Rendered HnE image data in {toc - tic2:0.4f} seconds", 'green'))
    
    # # Optional?
    # # Scale down z-axis with factor TIFFwriteoutArgs[0] ?
    nuclImage = imageTileIn[:,1,:,:]
    # TIFFwriteoutArgs[1] -> scale factor Z
    rescaledNuclImage = rescale(nuclImage, (1.0 / float(TIFFwriteoutArgs[1]), 1.0,1.0) , anti_aliasing=True, preserve_range=True) 
    Tiff3DforSegm_filename = TIFFwriteoutArgs[0] + '/tile__H' + str(tileCol).zfill(3) + '_V' + str(tileRow).zfill(3) + '.tif'  # # Could add to tileArrangments json file.
    imsave(Tiff3DforSegm_filename, 
        rescaledNuclImage.astype(np.uint16), photometric='minisblack') # Data conversion to avoid greyvalue leveling.        
        
    return RGB_singleTile, contrastAdjustedDAPI, contrastAdjustedAUTOFLUORESCENCE, os.path.basename(Tiff3DforSegm_filename)


def padXYtoPowerOf2(imageStack, rgb = False):
    def NextPowerOfTwo(number):
        return int(math.pow(2,(np.ceil(np.log2(number)))))
    
    oldXdim = imageStack.shape[1]
    oldYdim = imageStack.shape[2]    
    newXdim = NextPowerOfTwo(oldXdim)
    newYdim = NextPowerOfTwo(oldYdim)            
    
    print(colored(('Resizing/padding image: old size (x/y):', oldXdim, oldYdim , '. New size (x/y): ', newXdim, newYdim, '. Image stack shape:', imageStack.shape), 'green'))
        
    if rgb:
        return np.pad(imageStack, ((0,0),(0,newYdim-oldYdim),(0,newXdim-oldXdim),(0,0)), 'reflect')
    else:
        return np.pad(imageStack, ((0,0),(0,newYdim-oldYdim),(0,newXdim-oldXdim)), 'reflect')    
    

def main():

    parser = argparse.ArgumentParser(description='Generate pseudo HnE image \
        from Imaris image collection.')
    
    parser.add_argument('imarisXMLinfile', metavar='ImarisXMLinput', type=argparse.FileType('r'), 
            help = 'XML image metadata file.'    )

    parser.add_argument('--outfileHNEmaxIntensity', metavar='HnE output', type=argparse.FileType('w'),
            help = '8bit RGB 3D TIFF; each Imaris tile corresponds to one HnE TIFF page.',
            default='HNEmaxIntensity.tif')            
    parser.add_argument('--maskFile', metavar='mask output', type=argparse.FileType('w'),
            help = 'Low resolution masks corresponding to rendered HnE used for clustering: \
                0 pixel value if no nuclear signal is present.',
            default='clusteringMask.tif')                        
    parser.add_argument('--outfileNUCLsubstackMaxIntensity', metavar='nuclear channel projection output', 
            type=argparse.FileType('w'),
            help = '16bit 3D TIFF; each Imaris tile corresponds to one nuclear TIFF page.',
            default='NUCLmaxIntensity.tif')
    parser.add_argument('--outfileCYTOsubstackMaxIntensity', metavar='cyto channel projection output',
            type=argparse.FileType('w'),
            help = '16bit 3D TIFF; each Imaris tile corresponds to one cyto TIFF page.',
            default='CYTOmaxIntensity.tif')            
    
    parser.add_argument('--tileArrangementFile', metavar='tile arrangement output (json)',
            type=argparse.FileType('w'),
            help = '16bit 3D TIFF; each Imaris tile corresponds to one cyto TIFF page.',
            default='data/features/tileArrangement.json')            
    
    parser.add_argument('--substackThickness',  metavar='sT', type=float,
            help = 'Thickness of stack for HnE generation in microns. Default 5 micron.',
            default=5.0)
    parser.add_argument('--useZStackFraction',  metavar='zF', type=float,
            help = 'Use specific fraction of z Stack around nuclear intensity maximum for axial projection.\
                    Set to 1.0 if using thin (5 micron) tissues, to avoid sampling empty 3D regions.\
                    Non-positive values trigger automatic computation of substack thickness based on target physical dimensions.',
            default=-1.0) # # May wish to adjust substack thickness for case this parameter is set to a positive value. Skip computation entirely if param >= 1.0

    parser.add_argument('--cytoChannel', metavar='cC', type=int,
            help = 'Image channel containing cyto signal (e.g., autofluorescence).',
            default=0)
    parser.add_argument('--nuclearChannel', metavar='nC', type=int,
            help = 'Image channel containing nuclear signal (e.g., DAPI).',
            default=1)

    parser.add_argument('--maskThresh', metavar='mT', type= int,
            help = 'Threshold for binarizing in mask generation (default=10).',
            default=10)    

    parser.add_argument('--colorWeightNUCL',  metavar='cWN', type=float,
            help = 'Color weight for generated H stain.',
            default=2.56)
    parser.add_argument('--colorWeightCYTO',  metavar='cWC', type=float,
            help = 'Color weight for generated E stain.',
            default=0.1)

    parser.add_argument('--HnEresLevel',  metavar='resLvl', type=int,
            help = 'Imaris resolution level to use for HnE rendering (default 0 is highest).',
            default=0)

    parser.add_argument('--TIFFwriteout',  metavar=('target dir/filename prefix', 'zAnisotropyTarget'), 
            type=str, nargs=2,  # # Could switch to optional arg by nargs='?' or nargs='*'
            help = 'Use writeoutChannel to convert Imaris data to TIFF at selected HnEresLevel.\
                Write with filenamePrefix. Scale down z-Axis with scale facotr zAnisotropy',
            default = ['data/TIFFtiles/', 2.0]) # # Add sanity check for dir and zAnisotropy?

    parser.add_argument('--padImgDimsToPowerOf2', 
            help = 'If set, pads image dimensions (X/Y) to next power of two.',
            action='store_true')    

    parser.add_argument('--computeMaskOnly', 
            help = 'If set, computes only tissue masks at lowest resolution level. Useful for testing integrity of a dataset.',
            action='store_true')    

    parser.add_argument('--fixedZposition',  metavar='zCoord', type=int,
            help = 'If non-negative, render HnE image from user specified confocal layer with zCoord. \
                    Use if automatic detection of best z coordinte fails (e.g., if there are only very few tiles in the dataset).',
            default=-1)  

    parser.add_argument('--imsFilenameFilter',  metavar='substring', type=str,
            help = 'Process only those Imaris tiles containing [substring] in filename',
            default='')  
                    
    args = parser.parse_args()

    tiffWriteoutDir = args.TIFFwriteout[0]
    if not os.path.exists(tiffWriteoutDir):
        os.makedirs(tiffWriteoutDir) 
    
    try:
        zScalingFactor = float(args.TIFFwriteout[1])
        print (colored('Downscaling 3D image with an additional factor of', zScalingFactor, 'along optical axis.', 'green'))
    except Exception as e:
        print (colored('Could not read specified scaling factor: ', 'red'))
        print (colored(e, 'red'))
    
    # Imaris metadata: voxel dimensions, arrangement of tiles
    try:    
       tree = ET.parse(args.imarisXMLinfile)
    except FileNotFoundError:
        sys.exit('XML metadata file does not exist.') 
    except Exception as e:
        if hasattr(e, 'message'):
            print(colored(e.message,'red'))
        else:
            print(e)    
        print (colored('XML file could not be parsed: '+ XMLinputfile,'green'))
        sys.exit(2)
    imarisTeraStitcherMetaData_root = tree.getroot()
    
    voxel_dimensions = imarisTeraStitcherMetaData_root.find('voxel_dims')
    if voxel_dimensions is None:
        print(colored("Error: No voxel dimension info found in Imaris xml file " + imarisXMLinfile.name, 'red'))
        sys.exit(1)        
    print(colored('Voxel dimensions: ' + str(voxel_dimensions.attrib),'green'))        
        
    import javabridge 
    from utils.bioformat_reader import BioFormatsReader   
    imarisImageReader = BioFormatsReader()

    """
    Script logic:
    1. Traverse tiles on lowest resolution level to generate masks for clustering
    2. Get resolution level below selected level for HnE generation(to get optimal z-value)
        Note: This level may or may not have lower z-resolution. Correct selected z if resolution is lower.
    3. Determine z_min and z_max from selected z, voxel size, and optional CLI parameters (if any)
    4. Compute max intensity projections and HnE images for specified resolution level.

    """ 
    
    # 3D array to be saved in TIFF format, RGB coded pixels; resize abd append to this
    #HnEimageTiles = np.empty((0,0,0,3), dtype = np.uint8) # (numTiles, Y_size, X_size, RGB channels)

    maskArrayInitialized = False
    maskTiles = np.empty((0,0,0), dtype = np.uint8)
    
    numberOfReferencedTiles = sum(1 for _ in imarisTeraStitcherMetaData_root.iter('Stack'))
    # # Include only those tiles that actually exist as readable image file/tile?

    imarisInputFileNames = []
    Tiff3DFilesForSegmentation = []
    tileIndices = []
    tilePhysicalDisplacements = []
    initialConfocalSubstackZcoordsSelection = []
    nonEmptyTiles = [] # 0 if tile with corresponding index does not have dapi signal, 1 otherwise.
    
    progressBar = tqdm(total=len([l for l in imarisTeraStitcherMetaData_root.iter('Stack')]), desc="Image tiles processed", \
                            bar_format="{l_bar}{bar} [ time left: {remaining} ]")
    
    # Extract masks for clustering, empty tiles and (initial) z-coord of NUCL signal maximum per tile
    resultImageIdx = 0 # Separate tile index to avoid saving empty images.
    for tileIdx, tile in enumerate(imarisTeraStitcherMetaData_root.iter('Stack')):
        progressBar.update(1)   
        ta = tile.attrib
        currentImarisTileFile = os.path.split(args.imarisXMLinfile.name)[0] + '/' + ta['IMG_REGEX']
        print(colored('Processing tile: ' + currentImarisTileFile,'green'))
        print(colored('Reading image stack: ' + currentImarisTileFile + 'at COL/ROW ' + str(ta['COL']) +','+ str(ta['ROW']) +  
            'with micron displacements HORIZ/VERT ' + str(ta['ABS_H']) +','+ str(ta['ABS_V']),'green'))
        
        try:
            print(f"Trying to open {currentImarisTileFile}")
            imarisImageReader.open(currentImarisTileFile) 
        except FileNotFoundError:
            print(colored('WARNING: File referenced in XML file does not exist. Skipping. Filename: ' + str(ta['IMG_REGEX']),'red'))
            continue

        if (imarisImageReader.get_n_channels() < 2):
            javabridge.kill_vm()
            print(colored('WARNING: Unsuitable image format: Expected 2 channels.','red'))  
            continue

        # Generate mask with lowest resolution level.
        lowestResolutionLevel = imarisImageReader.get_level_count() - 1        
        maskImageTile = confocalStackToMask(imarisImageReader, lowestResolutionLevel, args.maskThresh, args.cytoChannel, args.nuclearChannel) 
        
        if not maskArrayInitialized: 
            maskTiles.resize(tuple([numberOfReferencedTiles]) + maskImageTile.shape)
            maskArrayInitialized = True            
        maskTiles[resultImageIdx,:,:] = maskImageTile
        resultImageIdx = resultImageIdx + 1
        
        nonEmptyTiles.append(0 if not maskImageTile.any() else 1)
        
        # Get optimal substack location for max intensity projection from selected resolution -1 (if not already at lowest res)
        # Taking it from lower resolution level speeds up the process by factor 4 or 8 (depending on wether z has been downscaled)
        # ... thus creating at most 25 percent overhead, compared to almost 100 % overhead using original resolution.
        # Take care of wether the z-resolution has been actually downscaled in the lowest resolution level, rf:
        # https://github.com/imaris/ImarisWriter/blob/master/doc/Imaris5FileFormat.pdf
        # (skip this if the location is already fixed by the user, or using the entire stack has been specified).
        
        scaleFactor_ZselectLevel2currentLevel = 2 # Usual downscaling between resolution levels
        
        zSelectionResLevel = args.HnEresLevel + 1 # Select z coordinate of max intensity one resolution level below HNE resolution lvl
        imarisImageZdimSelect = 0
        if zSelectionResLevel >= lowestResolutionLevel:
            zSelectionResLevel = lowestResolutionLevel
            scaleFactor_ZselectLevel2currentLevel = 1
        else:
            scaleFactor_ZselectLevel2currentLevel = 2 # Usual downscaling between resolution levels
            # Check if that is really the case:
            _,_, imarisImageZdimHnE = imarisImageReader.get_3Ddimensions(args.HnEresLevel)            
            _,_, imarisImageZdimSelect = imarisImageReader.get_3Ddimensions(zSelectionResLevel)
            _,_, imarisImageZdimLower = imarisImageReader.get_3Ddimensions(zSelectionResLevel+1)
            scaleFactor_ZselectLevel2currentLevelFromData = imarisImageZdimHnE / imarisImageZdimSelect
            if 0.95 <= scaleFactor_ZselectLevel2currentLevelFromData <= 1.05:
                scaleFactor_ZselectLevel2currentLevel = 1

        print (colored('Current vs selected resolution level: ' + str(scaleFactor_ZselectLevel2currentLevelFromData) + ' ' + str(scaleFactor_ZselectLevel2currentLevel), 'green'))

        if not args.computeMaskOnly:
            selectedZinOriginalResolutionLevel = scaleFactor_ZselectLevel2currentLevel * \
                getMaxZIntensity(imarisImageReader, zSelectionResLevel, nuclearChannelIdx=args.nuclearChannel)               
            # int(...), because json cannot serialize numpy int64 type
            initialConfocalSubstackZcoordsSelection.append(int(selectedZinOriginalResolutionLevel))
                                
        tileIndices.append((int(ta['COL']), int(ta['ROW'])))        
        tilePhysicalDisplacements.append((int(ta['ABS_H']), int(ta['ABS_V'])))
               
        imarisImageReader.close()
      
    if not args.computeMaskOnly:     
        # Get median z coordinate over all non-empty tiles
        assert len(nonEmptyTiles) == len(initialConfocalSubstackZcoordsSelection), \
            "Length of nonEmptyTiles list should equal length of selected Z coordinates. Internal program error!"   
        # For empty tiles, the determined z-coordinate is meaningless. Exclude these (except there is no choice).
        #IF NO TILE IS NONEMPTY, THERE'S PROBABLY SOMETHING WRONG WITH MASK EXTRACTION
        zCoordsOfNonEmptyTiles = [x for x, y  in zip(initialConfocalSubstackZcoordsSelection, nonEmptyTiles) if y > 0 ] \
            if not all(v == 0 for v in nonEmptyTiles) else initialConfocalSubstackZcoordsSelection          
            
        midElemIdx = math.trunc(len(zCoordsOfNonEmptyTiles)/2)
        zCoordsOfNonEmptyTiles.sort()    
        
        print(colored('Initial confocal substack z-coordinate selection: ' + str(initialConfocalSubstackZcoordsSelection) +\
             '; Non empty tiles: '+ str(nonEmptyTiles), 'green'))
                
        # Override selected z coordinate, if user specified. 
        # # If set, could omit prior computation [but: still output 3D TIFFs for segmentation?].
        medianZoverTiles = zCoordsOfNonEmptyTiles[midElemIdx] if args.fixedZposition < 0 else args.fixedZposition
        print(colored('Selected Z (median): ' + str(medianZoverTiles), 'green'))

        # Compute start- and end-coordinates of substack; IN VOXEL UNITS OF USER SELECTED HNE RESOLUTION LEVEL ! 
        voxelDimResolutionLevelCorrectionFactor = math.pow(2, args.HnEresLevel)
        micronPerVoxelZ = abs(float(voxel_dimensions.attrib['D'])) * voxelDimResolutionLevelCorrectionFactor
        ZsubStackRangeInVoxels = round(args.substackThickness / micronPerVoxelZ)

        print(colored('Substack thickness in voxels: ' + str(ZsubStackRangeInVoxels) , 'green'))
        
        zMin = medianZoverTiles - math.trunc(ZsubStackRangeInVoxels/2)
        print(colored('voxelDimResolutionLevelCorrectionFactor: ' +  str(voxelDimResolutionLevelCorrectionFactor), 'green'))   
        zMax = zMin + ZsubStackRangeInVoxels    
        
        if (zMin < 0):
           zMax = zMax + abs(zMin)
           zMin = 0
           medianZoverTiles = medianZoverTiles + abs(zMin)       
        if (zMax > imarisImageZdimHnE  - 1):
           zMin = zMin - (zMax - imarisImageZdimHnE - 1)
           zMax = imarisImageZdimHnE - 1
           medianZoverTiles = medianZoverTiles - (zMax - imarisImageZdimHnE - 1)
        if (zMin < 0):
           zMin = 0
           print (colored('Warning: Section selected for maximum intensity projection is less than specified: zmin ' + str(zMin) + ' zmax: ' + str(zMax), 'yellow'))  
     
        # Extract max intensity projections on selected resolution level and mix HnE        
        HnEimageTilesMaxProj = np.empty((0,0,0,3), dtype = np.uint8) # (numTiles, Y_size, X_size, RGB channels)
        NuclTilesMaxProj = np.empty((0,0,0), dtype = np.uint16) # (numTiles, Y_size, X_size)
        CytoTilesMaxProj = np.empty((0,0,0), dtype = np.uint16) # (numTiles, Y_size, X_size)
        HnEresultArrayInitialized = False       
        
        resultImageIdx = 0 # Separate tile index to avoid saving empty images.
        for tileIdx, tile in enumerate(imarisTeraStitcherMetaData_root.iter('Stack')):
            ta = tile.attrib
            currentImarisTileFile = os.path.split(args.imarisXMLinfile.name)[0] + '/' + ta['IMG_REGEX']
            
            if not (args.imsFilenameFilter in ta['IMG_REGEX']):
                print(colored('Skipping with filter ' + args.imsFilenameFilter + ' not in ' + str(ta['IMG_REGEX']), 'green'))
                continue
            
            print(colored('Reading image stack: ' + currentImarisTileFile + ' at COLUMN '+ str(ta['COL']) + ', ROW ' + str(ta['ROW']) +\
                'with displacements HORIZ '+ str(ta['ABS_H']) + ' VERT ' + str(ta['ABS_V']), 'green'))
            
            try: 
                imarisImageReader.open(currentImarisTileFile) 
            except FileNotFoundError:
                print(colored('WARNING: File referenced in XML file does not exist. Skipping. Filename: ' +  str(ta['IMG_REGEX']),'red'))
                continue

            if (imarisImageReader.get_n_channels() < 2):
                javabridge.kill_vm()
                print(colored('WARNING: Unsuitable image format: Expected 2 channels.','red'))
                continue
            
            pseudoHnEimage, nuclImage, cytoImage, Tiff3DforSegm_filename = \
                confocalStackToHNE(imarisImageReader, args.HnEresLevel, zMin, zMax, args.TIFFwriteout, 
                    ta['COL'], ta['ROW'], args.colorWeightNUCL, args.colorWeightCYTO, 
                    args.cytoChannel, args.nuclearChannel) 
                    
            # inserted 251101 - TODO remove        
            temp_HnE_out_filename = Path(tiffWriteoutDir)/Path(f"pseudoHnE_COL_{ta['COL']}_ROW_{ta['ROW']}.tif")
            print(colored(f"Writing {temp_HnE_out_filename}","yellow"))
            imsave(temp_HnE_out_filename,pseudoHnEimage)  # tifffile.imwrite alias

            Tiff3DFilesForSegmentation.append(Tiff3DforSegm_filename)

            if not HnEresultArrayInitialized:
                NuclTilesMaxProj.resize(tuple([numberOfReferencedTiles]) + nuclImage.shape)        
                CytoTilesMaxProj.resize(tuple([numberOfReferencedTiles]) + cytoImage.shape)
                HnEimageTilesMaxProj.resize(tuple([numberOfReferencedTiles]) + pseudoHnEimage.shape)
                HnEresultArrayInitialized = True            
                  
            HnEimageTilesMaxProj[resultImageIdx,:,:,:] = pseudoHnEimage
            NuclTilesMaxProj[resultImageIdx,:,:] = nuclImage
            CytoTilesMaxProj[resultImageIdx,:,:] = cytoImage
            resultImageIdx = resultImageIdx + 1
            imarisInputFileNames.append(ta['IMG_REGEX'][-8:-3])
            
            imarisImageReader.close()

        if args.padImgDimsToPowerOf2:
            HnEimageTilesMaxProj= padXYtoPowerOf2(HnEimageTilesMaxProj, rgb = True)
            NuclTilesMaxProj = padXYtoPowerOf2(NuclTilesMaxProj)
            CytoTilesMaxProj = padXYtoPowerOf2(CytoTilesMaxProj)
            
        # Cut off unused image tiles (which were previously pre-allocated)
        HnEimageTilesMaxProj.resize(tuple([resultImageIdx]) + HnEimageTilesMaxProj[0].shape)
        NuclTilesMaxProj.resize(tuple([resultImageIdx]) + NuclTilesMaxProj[0].shape)
        CytoTilesMaxProj.resize(tuple([resultImageIdx]) + CytoTilesMaxProj[0].shape)        
        
        print(colored(('Shape of generated image: ' ,NuclTilesMaxProj.shape),'green'))
            
        imsave(Path(tiffWriteoutDir)/args.outfileHNEmaxIntensity.name, HnEimageTilesMaxProj, photometric='rgb')
        imsave(Path(tiffWriteoutDir)/args.outfileNUCLsubstackMaxIntensity.name, NuclTilesMaxProj, photometric='minisblack')
        imsave(Path(tiffWriteoutDir)/args.outfileCYTOsubstackMaxIntensity.name, CytoTilesMaxProj, photometric='minisblack')

    imsave(Path(tiffWriteoutDir)/args.maskFile.name, maskTiles, photometric='minisblack')

    tileLocations = {}
    tileLocations['tileIndices_COL-ROW'] = tileIndices
    tileLocations['tileDisplacementsInVoxelUnits'] = tilePhysicalDisplacements
    tileLocations['initialConfocalSubstackZcoordsSelection'] = initialConfocalSubstackZcoordsSelection    
    tileLocations['tileNotEmpty'] = nonEmptyTiles
    tileLocations['imarisInputFileNames'] = imarisInputFileNames
    tileLocations['tiff3DFilesForSegmentation'] = Tiff3DFilesForSegmentation    
    tileLocations['tiffFilesForSegmentation_dir-zScaling'] = args.TIFFwriteout
    if not args.computeMaskOnly:
        tileLocations['zCoordsOfNonEmptyTiles'] = zCoordsOfNonEmptyTiles
        tileLocations['selectedSubstackPixelCoordinates'] = [zMin, medianZoverTiles, zMax]
    voxDimD, voxDimV, voxDimH = abs(float(voxel_dimensions.attrib['D'])),\
        abs(float(voxel_dimensions.attrib['V'])), abs(float(voxel_dimensions.attrib['H']))
    tileLocations['voxelDimensions'] = [voxDimD, voxDimV, voxDimH]   
    resolutionLevelDownscalingFactor = 2 ** args.HnEresLevel 
    tileLocations['voxelDimensionsFeatures'] = \
        [voxDimD * resolutionLevelDownscalingFactor, \
        voxDimV *resolutionLevelDownscalingFactor, \
        voxDimH * resolutionLevelDownscalingFactor]  
    tileLocations['voxelDimensionsTIFFFiles'] = \
        [voxDimD * resolutionLevelDownscalingFactor, \
        voxDimV *resolutionLevelDownscalingFactor, \
        voxDimH * resolutionLevelDownscalingFactor * zScalingFactor]       

    featureOutputDir = Path(args.tileArrangementFile.name).parent
    featureOutputDir.mkdir(parents=True, exist_ok=True)        
    print(f"Writing features to  {featureOutputDir}")
    json.dump(tileLocations, args.tileArrangementFile)
    javabridge.kill_vm()
    
        
if __name__ == "__main__":
   main()
