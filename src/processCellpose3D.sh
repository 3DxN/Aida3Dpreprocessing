#!/bin/bash

# Runs the following pipeline: 
# 1) Read Imaris/Bitplane 3D image tiles data (specified together with metadata (voxel resolution, tile locations, tile file names) in xml), generate intermediate 3D image tiles in TIFF format for segmentation and texture feature extraction, generate 2D H&E tiles from specified DAPI and cyto channels, 
# 2) Run segmentation (StarDist3D here)
# 3) Generate 3D meshes from segmented data and compute morphological, textural and contextual features


# Required CLI arguments:
# $1 Source directory
# $2 Target directory
#
# Optional CLI arguments:
# -h Help message
# -r Resolution level to select from Imaris HDF5 image data. For details see below.
r_arg="1" # Default value
# -z Axial downscaling factor (target anisotropy).
z_arg="2.0" # Default value
# DAPI channel
d_arg="1" # Default value - Make sure the channel exists!
# Cyto channel
c_arg="0" # Default value - Make sure the channel exists!


while getopts 'hr:z:c:d:' OPTION; do
    case "$OPTION" in
        h)
            echo "Usage: $0 [-h] [-r resolution level value] [-z axial downscaling factor (target anisotropy)]"
            ;;
        r)
            r_arg="$OPTARG"
            echo "Using Bitplane resolution level: $r_arg"
            ;;
        z)
            z_arg="$OPTARG"
            echo "Target axial anisotropy downscaling factor for 3D segmentation: $z_arg"
            ;;
        d)
            d_arg="$OPTARG"
            echo "Using DAPI channel number: $d_arg"
            ;;
        c)
            c_arg="$OPTARG"
            echo "Using CYTO channel number: $c_arg"
            ;;            
        *)
            echo "Usage: $0 [-h] [-r resolution level value] [-c CYTO channel number] [-d DAPI channel number] [-z axial downscaling factor (target anisotropy)] ImarisSourceXMLfile destinationDir" >&2
            exit 1
        ;;
    esac
done

shift "$(($OPTIND -1))"



if [ ! -z $1 ]; then
    echo "Reading image data from: $1"
else
    echo "No data source location (Imaris Bitplane xml metadata) provided."  >&2
    echo "Usage: $0 [-h] [-r resolution level value] [-z axial downscaling factor (target anisotropy)] ImarisSourceXMLfile destinationDir" >&2
    exit 1
fi

if [ ! -z $2 ]; then
    echo "Writing results to: $2"
else
    echo "No location for computed results provided."  >&2
    echo "Usage: $0 [-h] [-r resolution level value] [-z axial downscaling factor (target anisotropy)] ImarisSourceXMLfile destinationDir" >&2
    exit 1
fi

echo "HnE generation command:\
python $PHENO_SRC/src/genHnE.py \
--HnEresLevel $r_arg \
--nuclearChannel $d_arg \
--cytoChannel $c_arg \
--maskFile $2/clusteringMask.tif \
--outfileNUCLsubstackMaxIntensity $2/NUCLmaxIntensity.tif \
--outfileCYTOsubstackMaxIntensity $2/CYTOmaxIntensity.tif \
--outfileHNEmaxIntensity $2/HNEmaxIntensity.tif \
--tileArrangementFile $2/tileArrangement.json \
--TIFFwriteout $2/TIFFtiles/ $z_arg \
$1
"



# Extract virtual HnE sections and volumes for segmentation
mkdir -p $2

python $PHENO_SRC/src/genHnE.py \
--HnEresLevel "$r_arg" \
--nuclearChannel "$d_arg" \
--cytoChannel "$c_arg" \
--maskFile "$2/clusteringMask.tif" \
--outfileNUCLsubstackMaxIntensity "$2/NUCLmaxIntensity.tif" \
--outfileCYTOsubstackMaxIntensity "$2/CYTOmaxIntensity.tif" \
--outfileHNEmaxIntensity "$2/HNEmaxIntensity.tif" \
--tileArrangementFile "$2/tileArrangement.json" \
--TIFFwriteout "$2/TIFFtiles/" "$z_arg" \
"$1"


# HnE resolution level 1 implies downscaling in XY and Z planes of factor 2. TIFFwriteout parameter 2.0 applies and additional downscaling of factor 2 along optical axis. Hence, original voxel dimensions [0.2 0.2 0.2] pix/um transform into voxel dimensions [0.4 0.4 0.8] pix/um in the TIFF files used for 3D segmentation. Such anisotropic scaling may be desirable to match a segmentation model (StarDist) or make segmentation computationally more efficient (cellpose).
# For computing morphological features, the scaling along the optical axis is reversed, so metric dimensions of derived features will be [0.4 0.4 0.4] pix/um.






#Run Cellpose3D segmentation
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate cellpose
mkdir -p $2/Cellpose3DOut
echo "Segmentation command: python $PHENO_SRC/src/predCellpose3D.py  --inDir $2/TIFFtiles/ --outDir $2/Cellpose3DOut --tileJsonFilename $2/tileArrangement.json --useGPU"
python $PHENO_SRC/src/predCellpose3D.py --inDir "$2/TIFFtiles/" --outDir "$2/Cellpose3DOut" --tileJsonFilename "$2/tileArrangement.json" --useGPU
#conda activate base


python $PHENO_SRC/src/labels2starconvex3D.py "$2/Cellpose3DOut" 0 "$2/tileArrangement.json"


# Mesh generation and feature computation
mkdir -p "$2/Cellpose3DGeometryAndFeatures"
echo "FeatureExtraction command: python $PHENO_SRC/src/extractVisData.py --inDir $2/Cellpose3DOut --outDir $2/Cellpose3DGeometryAndFeatures --tileFile $2/tileArrangement.json"
python $PHENO_SRC/src/extractVisData.py --inDir "$2/Cellpose3DOut" --outDir "$2/Cellpose3DGeometryAndFeatures" --tileFile "$2/tileArrangement.json" #--computeHaralick
