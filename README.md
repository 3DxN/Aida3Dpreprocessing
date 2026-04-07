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
python src/proximityAnalysis/plotRatios.py pct_cd8_2026_02_19-11_24_40.json pct_gh2ax_2026_02_19-10_55_23.json mean_intensity_gh2ax_2026_02_19-11_24_40.json -c control control -p pdf
```

## Snakemake workflow

Use the Snakemake pipeline to run the full workflow from a single config file.

1. Install Snakemake (in the same environment as the repo dependencies):
```
pip install snakemake
```
2. Edit `config.yaml`.
3. Run individual steps or the full pipeline:
```
snakemake -j 1 hne
snakemake -j 1 segmentation
snakemake -j 1 analysis
snakemake -j 1 plot
snakemake -j 1 labels2starconvex3d
snakemake -j 1 extract_vis_data
snakemake -j 1
```
To override input/output roots from the CLI:
```
snakemake -j 1 --config input_dir=/path/to/inputs output_dir=/path/to/outputs
```

Notes:
- The analysis step writes a JSON config for `src/proximityAnalysis/config.py` under `data/workflow_state`.
- The plot step selects the most recent analysis outputs and creates `*_texture_features_w_neighbors.csv` aliases to match `plotRatios.py` expectations.
- Geometry steps are controlled via the `geometry` section in `config.yaml`. Set `geometry.enabled: true` to include them in the default `snakemake -j 1` run.
- `labels2starconvex3d` expects Cellpose `.npy` outputs (masks/flows). If you already have StarDist `*.pickle` outputs, set `geometry.run_labels2starconvex: false` and point `geometry.input_dir` to those pickles.
- Input paths (e.g., `imaris_xml`, `raw_image_path`, `xml_files_path`) resolve relative to `input_dir` when not absolute.
- Output paths (e.g., TIFFs, segmentations, analysis outputs) resolve relative to `output_dir` when not absolute.
- Use `snakemake -n -p` for a dry run with command printing.
