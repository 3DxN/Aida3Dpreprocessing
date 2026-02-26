# AIDA-3D preprocessing

Scripts for preparing data (Imaris format) for visualisation in [AIDA-3D](https://github.com/3DxN/AIDA-3D-OMAL)

## Deployment:

1. Clone repository:
```
https://github.com/3DxN/Aida3Dpreprocessing
```

2. Create and activate a Python virtual environment (required Python version: ...):

Linux:
```
python -m venv venv
source venv/bin/activate
```

Windows:
```
python -m venv venv
venv/bin/Activate.ps
```

3. Install dependencies:
```
pip install -r requirements.txt 
```


## Data processing

Steps include:
- Converting Imaris data to TIFF 
- Generating HnE




1. Convert Imaris data
Imaris data is assumed to include a metadata `xml` file describing the geometric arrangement of tiles and relative file paths to data of individual tiles. Tile image data is stored in `.ims` files referenced by the `.xml` file.

`~/OMAL/250916aida3dData/012523_303_1_63X_stitchable.xml`


`python genHnE.py data/012523_303_1_63X_stitchable.xml --TIFFwriteout ./data/TIFFtiles 1 --nuclearChannel 3 --cytoChannel 4 --fixedZposition 11`


2. Run segmentation
```
python src/segmentation/predCellpose3D.py --inDir data/TIFFtiles/ --outDir data/CellposeSegementations --tileJsonFilename data/features/tileArrangement.json
```

3. Run CD8/GH2AX analysis
```
python src/proximityAnalysis/analyze_multi.py 

```
Note: Set parameters in file `src/proximityAnalysis/config.py` (to be replaced by `argparse` interface)

4. Plot results
```
python src/proximityAnalysis/plotRatios.py 
usage: PlotRatios [-h] [-p PLOT_OUTPUT_FORMAT] [-c [CLASSLIST ...]] [--num_pseudo_classes NUM_PSEUDO_CLASSES]
                  pct_cd8_json_file pct_gh2ax_json_file mean_intensity_gh2ax_json_file
PlotRatios: error: the following arguments are required: pct_cd8_json_file, pct_gh2ax_json_file, mean_intensity_gh2ax_json_file
```
e.g.,
```
python src/proximityAnalysis/plotRatios.py pct_cd8_2026_02_19-11_24_40.json pct_gh2ax_2026_02_19-10_55_23.json mean_intensity_gh2ax_2026_02_19-11_24_40.json -c control treated -p pdf
```

