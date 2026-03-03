import argparse
import os
import sys
import json
import math
import pickle
import numpy as np
#import vtk
from glob import glob
from geometryProcessor import gltfEncoder, morphologicalFeatures, textureFeatures, contextualFeatures
from termcolor import colored
from tifffile import imread
from utils import fs_dir_params

# Round float to n significant figures
def round_to_n(x, n):
    if not x: return 0
    power = -int(math.floor(math.log10(abs(x)))) + (n - 1)
    factor = (10 ** power)
    return round(x * factor) / factor

# Round the floats to reasonable precision to save mem (intended for use in json output)
def round_floats(o):
    if isinstance(o, float):
        return round_to_n(o,3) #round(o, 2)
    if isinstance(o, dict):
        return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_floats(x) for x in o]
    return o


# Not required if function removesuffix available (from Python 3.9)
def remove_suffix(text, suffix):
    return text[:-len(suffix)] if text.endswith(suffix) and len(suffix) != 0 else text    


def findImageFilesForTextureFeatureComputation(detailsFileName):
    
    #detailsFileName: e.g., .../Sec3HnE/StarDist3DOut/tile__H014_V014.tif_.pickle
    baseFilePath = remove_suffix(detailsFileName, '_.pickle')    
    
    #baseFilePath: e.g., .../Sec3HnE/StarDist3DOut/tile__H014_V014.tif
    dataBasePath = os.path.split(os.path.split(baseFilePath)[0])[0]

    #textureFileBasename: e.g., tile__H014_V014.tif
    textureFileBasename = baseFilePath.partition('.tif')[0] + '.tif'

    #dataBasePath: e.g., .../Sec3HnE/
    originalImageFileName = os.path.join(dataBasePath, 'TIFFtiles', os.path.split(textureFileBasename)[1])
    
    if baseFilePath.endswith('_seg.npy.pickle'):
        labelImageFileName = remove_suffix(baseFilePath, '_seg.npy.pickle') + '_cp_masks.tif'
    #For StarDist
    else:   
        labelImageFileName = baseFilePath + '_LABELS.tif'

    return originalImageFileName, labelImageFileName
    

def main(argv):
    # Nucleus surface meshes and morphological, textural and contextual features are generated
    # simultaneously, avoiding data parsing multiple times.

    parser = argparse.ArgumentParser(description='Compute gltf meshes from StarConvex nucleus geometry \
        and morphological, textural and contextual features from nucleus geometry and DAPI channel.')
    
    parser.add_argument('--inDir', metavar='inputDir', type=fs_dir_params.readable_dir, 
            help = 'Input directory with segmentation geometry (pickle) and textures.',
            default = os.path.join(os.getcwd() , 'sampleResults','Sec3HnE', 'StarDist3DOut'))
    parser.add_argument('--outDir', metavar='outputDir', type=fs_dir_params.writeable_dir,
            help = 'Output directory for geometry (gltf format) and computed features (json format).',
            default = os.path.join(os.getcwd() , 'sampleResults','Sec3HnE', 'StarDist3DGeometryAndFeatures'))
    parser.add_argument('--tileFile', metavar='tileArrangementsFileInput', type=argparse.FileType('r'), 
            help = 'Json file containing metadata of planar arrangement of extracted 2D tiles. \
            All voxel dimensions (before and after scaling) need to be present in this file.',
            default= os.path.join(os.getcwd() , 'sampleResults','Sec3HnE','tileArrangement.json'))
    parser.add_argument('--computeHaralick', 
            help = 'Whether or not to compute 3D Haralick features (169D) for each segmented nucleus.',
            action='store_true')            
    parser.add_argument('--separateFeatureFiles', 
            help = 'If set, derived features are not aggregated into a single json file, but split into files\
                containing morphological, textural and contextual features. ',
            action='store_true')             
    args = parser.parse_args()
    
    print('Extracting nucleus features and generating nucleus meshes.')
    print('Searching geomtry data in', os.path.join(args.inDir, '*.pickle'))
    print('Using tile arrangment metadata in: ', args.tileFile.name)
    print('Computing Haralick features: ', args.computeHaralick)

    if not os.path.exists(args.tileFile.name):
        print(colored('Tile arrangement file not speficied. Use CLI parameter --tileFile.','red'))
        sys.exit(1)
    with open(args.tileFile.name, 'r') as tF:
        tileFile=tF.read()
    try:
        tileData = json.loads(tileFile)        
    except:
        print(colored('Error: Cannot read tileFile: ' + args.tileFile.name,'red'))
        sys.exit(1)
    print("voxelDimensions: " + str(tileData['voxelDimensions'][0]))    
    print("voxelDimensionsFeatures: " + str(tileData['voxelDimensionsFeatures']))    
    print("voxelDimensionsTIFFFiles: " + str(tileData['voxelDimensionsTIFFFiles']))

    try:
        voxelDimensionsFeatures = tileData['voxelDimensionsFeatures']
        voxelDimensionsTIFFFiles = tileData['voxelDimensionsTIFFFiles']
    except:
        print(colored('Error: Cannot read voxel dimensions from tileArrangements json file.','red'))
        sys.exit(1)
    
    svx,svy,svz = voxelDimensionsFeatures[0],voxelDimensionsFeatures[1],voxelDimensionsFeatures[2]
    zAnisotropy = (svx/2.0 + svy/2.0) / svz
    
    print(colored('Using axial anisotropy factor: ' + str(zAnisotropy), 'yellow'))
        
    detailsFiles = glob(os.path.join(args.inDir, '*.pickle'))
    for detailsFile in detailsFiles:
        with open(detailsFile, 'rb') as f:          
            print('Generating mesh and features. Processing file: ', detailsFile)                        
            # 1) Generate and store gltf formatted geometry
            gltf_path = os.path.join(args.outDir , os.path.split(detailsFile)[1] + '_.gltf')
            bin_path = os.path.join(args.outDir , os.path.split(detailsFile)[1] + '_.bin')

            nucleusGeometry = pickle.load(f)          
            #zAnisotropyReverseTransform = np.array([[1/zAnisotropy,0,0],[0,1,0],[0,0,1]])
            zAnisotropyReverseTransform =  np.array([[0,1/zAnisotropy,0],[0,0,1],[1,0,0]])  #OK 
           
            starCenters = np.dot(nucleusGeometry['points'], zAnisotropyReverseTransform)
          
            ray_verts = np.dot(nucleusGeometry['rays_vertices'], zAnisotropyReverseTransform)
            ray_dists = nucleusGeometry['dist']
            
            vertex_data, index_data = gltfEncoder.GLTFstoreGeneratedMesh(starCenters, \
                ray_dists, ray_verts,nucleusGeometry['rays_faces'])
                                      
            nucleusFeaturesStruct = {}
            segmentationConfidence = nucleusGeometry['prob']
            nucleusFeaturesStruct['segmentationConfidence'] = round_floats(segmentationConfidence.tolist())
            document, buffers = gltfEncoder.stardist3d_to_gltf(vertex_data, index_data, gltf_path, bin_path)
            gltfEncoder.saveGLTF(gltf_path, bin_path, document, buffers)

            # 2) Compute and store features for nucleus visualization: Diameters, volumes, ellipsoid orientations, elongation
            #       morphological: elongation, volume
            
            # Compute vertex data separately for ellipsoid fit and elongation computing, i.e.,
            # don't rely on data extracted by GLTFexporter.
            cellHullPointsArray = []    
            for i in range(len(starCenters)): # for every cell                                
                cellHullPoints = starCenters[i] + np.swapaxes(ray_dists[i] * ray_verts.T, 0, 1)
                cellHullPointsArray.append(cellHullPoints)    
            
            print('Computing diameters ...', end=' ')   
            voxelDiameters = morphologicalFeatures.computeDiametersNP(cellHullPointsArray)
            #Conversion to metric, assuming isotropic sampling in feature space
            micronPerVoxel = abs(voxelDimensionsFeatures[0])
            nucleusFeaturesStruct['nucleusDiameters'] = round_floats([micronPerVoxel*x for x in voxelDiameters])
            
            print('Computing volumes ....', end=' ')
            nucleusVolumes = morphologicalFeatures.computeVolumes(cellHullPointsArray)
            #Convert from voxel to metric units:
            cubeMicronPerVoxel = abs(voxelDimensionsFeatures[0]*voxelDimensionsFeatures[1]*voxelDimensionsFeatures[2])
            nucleusFeaturesStruct['nucleusVolumes'] = round_floats([cubeMicronPerVoxel*x for x in nucleusVolumes])
                        
            ellipsoidFits = morphologicalFeatures.computeEllipsoidFits(cellHullPointsArray) #vertex_data
            nucleusEllipsoidCenters = [x[0].tolist() for x in ellipsoidFits]
            nucleusFeaturesStruct['nucleusEllipsoidCenters'] = round_floats(nucleusEllipsoidCenters)
            nucleusOrientations = [x[1].tolist() for x in ellipsoidFits]
            nucleusFeaturesStruct['nucleusEllipsoidAxes'] = round_floats(nucleusOrientations)
            nucleusRadii = [x[2].tolist() for x in ellipsoidFits]
            nucleusFeaturesStruct['nucleusEllipsoidRadii'] = round_floats(nucleusRadii)
            nucleusElongations, mainAxisIndices, longestAxisIndices = \
                    morphologicalFeatures.computeBiAxialElongations(ellipsoidFits)
            nucleusFeaturesStruct['elongations'] = round_floats(nucleusElongations)
            nucleusFeaturesStruct['mainAxisIndices'] = mainAxisIndices           
            nucleusFeaturesStruct['longestAxisIndices'] = longestAxisIndices
            
            features_file_path = os.path.join(args.outDir , os.path.split(detailsFile)[1] + '_.json')
            
            # 3) Compute/store texture features: 
            #       textural: Haralick, Punctateness histogram

            # TODO This can be made more robust by storing/retrieving paths in/from tileArrangements.json.
            originalImageFilePath, labelImageFilePath = findImageFilesForTextureFeatureComputation(detailsFile)
            if not (os.path.isfile(originalImageFilePath) and os.path.isfile(labelImageFilePath)):
                print('Searching for: ' + originalImageFilePath + ' and ' + labelImageFilePath, end = ' --> ')  
                print ('Cannot open source image or label image file. \
                    Skipping texture feature computation for image tile.')            
            else:

                print('Computing texture features, using files ' + originalImageFilePath + \
                    ' ' + labelImageFilePath, end = ' ')                
                textureFeatures_file_path = os.path.join(args.outDir , os.path.split(detailsFile)[1] + '_textureFeatures.json')
                
                textureFeaturesStruct = {}                     
                
                origImg = imread(originalImageFilePath) 
                labelImg = imread(labelImageFilePath)
                
                nuclBboxes = textureFeatures.cellBoundingBoxes(nucleusGeometry, zAnisotropy = zAnisotropy)
                
                haralickFeatures, punctatenessFeatures = \
                    textureFeatures.computeTextureFeatures(nuclBboxes, imTexture = origImg, \
                    imLabels = labelImg, includeHaralickFeatures=args.computeHaralick)
                
                if args.computeHaralick:
                    if args.separateFeatureFiles:
                        textureFeaturesStruct['haralickFeatures'] = round_floats(haralickFeatures)
                    else:
                        nucleusFeaturesStruct['haralickFeatures'] = round_floats(haralickFeatures)
                if args.separateFeatureFiles:
                    textureFeaturesStruct['punctuatenessFeatures'] = punctatenessFeatures
                else:
                    nucleusFeaturesStruct['punctuatenessFeatures'] = punctatenessFeatures
                                
                nucleusIrragularityScores = \
                    textureFeatures.computeNucleusIrregalurityScores(nuclBboxes, labelImg, nucleusOrientations,\
                                        zAnisotropy = zAnisotropy)
                
                if args.separateFeatureFiles:
                    with open(textureFeatures_file_path, 'w') as f2:
                        json.dump(textureFeaturesStruct, f2, indent=2)
            
                # It might be a bit dirty to do this write inside this scope:
                nucleusFeaturesStruct['nucleusIrregularityScores'] = round_floats(nucleusIrragularityScores)

            # 4) Compute/store neighborhood features: 
            #       contextual: knn concat of (distance, elongation, volume, relative orientation)

            neighborhoodFeatures = contextualFeatures.constructTissueNeighborhoodGraph(\
                    nucleusEllipsoidCenters, nucleusVolumes, nucleusElongations, nucleusOrientations,\
                    mainAxisIndices, desiredNumberOfNeighbors = 10)
            # # tissueInterfaceNucleusLikelihoods = computeTissueInterfaceLikelihoods(neighborhoodFeatures)

            mesenchymalScores, epithelialScores = contextualFeatures.aggregateCelltypeScores(\
                                                    contextFeatureArray = neighborhoodFeatures)

            neighborhoodFeatures_file_path = os.path.join(args.outDir , \
                os.path.split(detailsFile)[1] + '_neighborhoodFeatures.json')
            neighborhoodFeaturesStruct = {}   
            if args.separateFeatureFiles:                        
                neighborhoodFeaturesStruct['concatenatedNeighborhoodFeatures'] = round_floats(neighborhoodFeatures)
                neighborhoodFeaturesStruct['mesenchymalScores'] = round_floats(mesenchymalScores)
                neighborhoodFeaturesStruct['epithelialScores'] = round_floats(epithelialScores)
            else:
                nucleusFeaturesStruct['concatenatedNeighborhoodFeatures'] = round_floats(neighborhoodFeatures)
                nucleusFeaturesStruct['mesenchymalScores'] = round_floats(mesenchymalScores)
                nucleusFeaturesStruct['epithelialScores'] = round_floats(epithelialScores)

            if args.separateFeatureFiles:
               with open(neighborhoodFeatures_file_path, 'w') as f2:
                   json.dump(neighborhoodFeaturesStruct, f2, indent=2)

            with open(features_file_path, 'w') as f2:
                json.dump(nucleusFeaturesStruct, f2, indent=2)
                        
            
if __name__ == "__main__":
   main(sys.argv[1:])
