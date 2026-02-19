

CLASSES = ['control','treated']
SEGMENTATION_PATH = ['../012523_303_1_63X_stitchable','../013023_332_001_63X']
RAW_IMAGE_PATH = ['../012523_303_1_63X_stitchable','../013023_332_001_63X']
XML_FILES_PATH = ['../012523_303_1_63X_stitchable/012523_303_1_63X_stitchable.xml','../013023_332_001_63X/013023_332_001_63X.xml']

# Intermediate dummy config for testing
CLASSES = [''] # Provided by argparse
SEGMENTATION_PATH = ['./data/CellposeSegmentations','./data/CellposeSegmentations']
RAW_IMAGE_PATH = ['./data/raw']
XML_FILES_PATH = ['./data/raw/012523_303_1_63X_stitchable.xml','./data/raw/012523_303_1_63X_stitchable.xml']



gH2AX_threshold = 1300,1300
cd8_threshold = 750,1000
gH2AX__pixel_count_threshold = 100 
cell_size_threshold = 200

radii = [32,48,64,128]
cd8_dapi_box_overlap_threshold = 0.2

N_PSEUDO_CLASSES = 5
