import numpy as np
import time, os, sys
import json, pickle
import argparse

from os.path import exists
from glob import glob
from tifffile import imread,imwrite

from cellpose import utils, io, models

from utils import fs_dir_params


parser = argparse.ArgumentParser(description='Predict 3D Cellpose segmentation. \
    Consumes 3D TIFF images.')

parser.add_argument('--inDir', metavar='inputDir', type=fs_dir_params.readable_dir, 
        help = 'Input directory with segmentation geometry (pickle) and textures.', # note: Textures may be located in separate dir
        default = os.path.join(os.getcwd() , 'Cellpose3DOut'))
parser.add_argument('--outDir', metavar='outputDir', type=fs_dir_params.writeable_dir,
        help = 'Output directory for geometry (gltf format) and computed features (json format).',
        default = os.path.join(os.getcwd() , 'Cellpose3DGeometryAndFeatures'))
parser.add_argument('--tileJsonFilename', metavar='tileArrangementsFileInput', type=argparse.FileType('r'), 
            help = 'Json file containing metadata of planar arrangement of extracted 2D tiles. \
            All voxel dimensions (before and after scaling) need to be present in this file.',
            default= os.path.join(os.getcwd() , 'sampleResults','Sec3HnE','tileArrangement.json'))   #TODO replace this default - may cause trouble when used with other datasets.      
parser.add_argument('--cellposeModel',  metavar='cellposeModel', type=str,
            help = 'Cellpose segmentation model to use.',
            default='cyto')             
parser.add_argument('--cellposeDiameter',  metavar='cp_D', type=float,
            help = 'Nucleus size estimate for Cellpose segmentation.',
            default=30.0)
parser.add_argument('--batchSize',  metavar='bS', type=int,
            help = 'Batchsize for inference of segmentation masks and gradient maps.',
            default=2)            
parser.add_argument('--anisotropy',  metavar='cp_a', type=float,
            help = 'Estimated anisotropy in spatial sampling of 3D volumes.',
            default=0.0)    # default 0 causes reading anisotropy from tileArangement.json file
parser.add_argument('--useGPU', 
            help = 'If set, compute Cellpose segmentation masks and gradient maps on GPU.',
            action='store_true')     
parser.add_argument(
        "--gpu_device", required=False, default="0", type=str,
        help="which gpu device to use, use an integer for torch, or mps for M1")                                    
args = parser.parse_args()

if not os.path.isdir(args.outDir):
    os.mkdir(args.outDir)

tileDataDict = json.load(args.tileJsonFilename)   

#device, gpu = models.assign_device(use_torch=True, gpu=args.useGPU, device=args.gpu_device)
device, gpu = models.assign_device(use_torch=True, gpu=False, device=args.gpu_device)

#model = models.Cellpose(gpu=args.useGPU, model_type=args.cellposeModel)

#model = models.CellposeModel(device=device, pretrained_model=args.cellposeModel, )
model = models.CellposeModel(gpu=True)

print('Starting Cellpose segmentation. ')
print('Model used:')
print(model)
print('Reading from:')
print(args.inDir)
print('Writing to:')
print(args.outDir)
print('Parameter cellposeDiameter:', args.cellposeDiameter)
print('Parameter anisotropy:', args.anisotropy)
if not 'voxelDimensionsTIFFFiles' in tileDataDict:
    print('Key voxelDimensionsTIFFFiles not available!')
    sys.exit(1)
segmentationDataVoxelDimensions = tileDataDict['voxelDimensionsTIFFFiles']
svx,svy,svz = segmentationDataVoxelDimensions[0], segmentationDataVoxelDimensions[1], segmentationDataVoxelDimensions[2]
voxelAnisotropy = (svx/2.0 + svy/2.0) / svz
print('Segmentation data anisotropy:', voxelAnisotropy)
if args.anisotropy > 0.0:
    voxelAnisotropy = args.anisotropy

SEGMENTATION_MASK_FILE_SUFFIX = "_cp_masks"

# Iterating through the tileArrangement.json
for imageFileName, isNotEmpty in zip(tileDataDict['tiff3DFilesForSegmentation'], tileDataDict['tileNotEmpty']):
    print('Processing image: ', imageFileName, end = ' ')
   
    if not isNotEmpty:
        print('  --> Image is empty - skipping ...')
        continue
   
    inFileImagePath = os.path.join(args.inDir, imageFileName)
    outFileLabelImagePath = os.path.join(args.outDir,os.path.splitext(imageFileName)[0] + '_CELLPOSE-LABELS'+SEGMENTATION_MASK_FILE_SUFFIX+os.path.splitext(imageFileName)[1])
    print(f"{outFileLabelImagePath=}")
    if exists(outFileLabelImagePath):
        print('  --> Cellpose segmentation exists. Skipping ...')
        continue           
    print('Writing segmentation to: ', outFileLabelImagePath)

    channel = [0,0]

    print('Reading img ', imageFileName)
    img = io.imread(inFileImagePath)

    print("Image shape: ", img.shape)
    print("voxelAnisotropy", voxelAnisotropy)

    z_axis = 0
    masks, flows, styles = model.eval(img, batch_size=args.batchSize,
        diameter=args.cellposeDiameter, do_3D=True,anisotropy=voxelAnisotropy,
        flow_threshold=0.4, z_axis=z_axis, progress=True) 
    print('Eval done.')
    # Omit saving entire model output to npy file (required for Cellpose GUI only)
    # io.masks_flows_to_seg(img, masks, flows, outFileLabelImagePath, channel)
    print('Processed ', outFileLabelImagePath)
    
    # save results as png
    #io.save_to_png(img, masks, flows, filename)
    io.save_masks(img, masks, flows, outFileLabelImagePath, suffix=SEGMENTATION_MASK_FILE_SUFFIX, png=False, tif=True,  save_flows=False, save_outlines=True, save_txt=True, save_mpl=True)    

    #dY,dX,cellprob = flows[3]
    
     
