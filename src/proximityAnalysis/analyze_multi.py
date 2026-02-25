import tifffile
import xml.etree.ElementTree as ET
import numpy as np
import os
import time
# from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.cluster import KMeans
from tqdm import tqdm
import json
from imaris_ims_file_reader.ims import ims_reader
from config import SEGMENTATION_PATH, RAW_IMAGE_PATH, XML_FILES_PATH, CLASSES, gH2AX_threshold, cd8_threshold, gH2AX__pixel_count_threshold, N_PSEUDO_CLASSES, radii
from utils_multi import get_cd8_segmentation_by_dilation, compute_features_for_roi, find_neighbors, numbers_of_neighbors_within_radius
from joblib import Parallel, delayed

import warnings
 
warnings.filterwarnings("ignore", message=".*scipy.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*GLCM is symmetrical.*", category=UserWarning)

print('STARTING THE ANALYSIS')
start_time0 = time.time()
start_time = time.time()
################# parse arguments ##########################

print('Applying configuration...')
print('Using segmentation masks in ', SEGMENTATION_PATH)
print('Using raw images in', RAW_IMAGE_PATH)
print('Using xml files', XML_FILES_PATH)
elapsed_time = time.time() - start_time
print(f"Time taken for parse arguments: {elapsed_time} seconds")
print('')

timestampstring = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
PCT_GH2AX_FILE = f'pct_gh2ax_{timestampstring}.json'
PCT_CD8_FILE = f'pct_cd8_{timestampstring}.json'
MEAN_INTENSITY_GH2AX_FILE = f'mean_intensity_gh2ax_{timestampstring}.json'
################# gh2ax analysis ###########################

print('Starting gammaH2AX and CD8 analysis...')
print('')
pct_gh2ax = {}
pct_cd8 = {}
mean_intensity_gh2ax = {}

for idx, treatmentClass in enumerate(CLASSES):
    print(f'CLASS: {treatmentClass}')
    tree = ET.parse(XML_FILES_PATH[idx])
    root = tree.getroot()
    pct_gh2ax[treatmentClass] = []
    pct_cd8[treatmentClass] = []
    mean_intensity_gh2ax[treatmentClass] = []
    gh2ax_ctr = 0
    cd8_ctr = 0

    GENERATED_FEATURE_FILE = treatmentClass + f'_texture_features_w_neighbors_{timestampstring}.csv'
    featureFile = open(GENERATED_FEATURE_FILE, 'w')

    for stack in root.iter('Stack'):
        col, row = int(stack.get('COL')), int(stack.get('ROW'))
        print(f'PROCESSING TILE: col: {col}, row: {row}')

        tile_file_root = 'tile__H%03d_V%03d' % (col, row)
        segmentation_file = tile_file_root + '_CELLPOSE-LABELS_cp_masks.tif'
        raw_file = str(stack.get('IMG_REGEX'))

        segmentationFilePath = os.path.join(SEGMENTATION_PATH[idx], segmentation_file)
        segmentation = tifffile.imread(segmentationFilePath)
        print(f'USING SEGMENTATION MASK: {segmentationFilePath = } with # unique cell labels = {len(np.unique(segmentation))}')

        # print(os.path.join(DIR, c, raw_file))
        # img = tifffile.imread(os.path.join(RAW_IMAGE_PATH[idx], raw_file[:-4] + '.tiff'))
        # print(os.path.join(RAW_IMAGE_PATH[idx], raw_file))
        start_time = time.time()
        ims_file = ims_reader(os.path.join(RAW_IMAGE_PATH[idx], raw_file))
        elapsed_time = time.time() - start_time
        ims_file.change_resolution_lock(0)
        img = ims_file[0, :, :, :, :]
        print(f'LOADED FILE {raw_file = } in {elapsed_time} seconds. Being of shape: {img.shape}')

        ## gh2ax and cd8 seems to be the raw fluorescent image
        ## They will become the masks using each threshold's value
        gh2ax = img[1, :, :, :]
        cd8 = img[0, :, :, :]
        #print(f'gh2ax shape: {gh2ax.shape}')
        #print(f'cd8 shape: {cd8.shape}')

        start_time = time.time()
        ## Generating masks for gh2ax and cd8
        gh2ax_ctr += 1
        gh2ax_mask = gh2ax > gH2AX_threshold[idx]

        cd8_ctr += 1
        cd8_mask = cd8 > cd8_threshold[idx]
        print(f'cd8_mask.shape: {cd8_mask.shape}')

        elapsed_time = time.time() - start_time
        print(f'GENERATED MASKS OF cd8 in {elapsed_time} seconds.')

        ## segmentation_masked => gh2ax mask with decent intensity
        segmentation_masked = segmentation * gh2ax_mask
        ## segmentation_cd8 => dilation and erosion boundary computation for cd8 segmentation
        ## Does this funcdtion taking too long to compute?
        start_time = time.time()
        
        segmentation_cd8 = get_cd8_segmentation_by_dilation(segmentation, cd8_mask)
        elapsed_time = time.time() - start_time
        print(f"CALLED get_cd8_segmentation_by_dilation: {elapsed_time} seconds")
        CD8_MASK_FILENAME = f"CD8_mask_col_{col}_row_{row}.tif"
        SEGMENTATION_FILENAME = f"SEGMENTATION_col_{col}_row_{row}.tif"
        CD8_SEGMENTATION_FILENAME = f"CD8_SEGM_col_{col}_row_{row}.tif"
        tifffile.imwrite(CD8_MASK_FILENAME,cd8_mask)
        tifffile.imwrite(SEGMENTATION_FILENAME,segmentation)
        tifffile.imwrite(CD8_SEGMENTATION_FILENAME,segmentation_cd8)

        ## Serial processing ROI labels in chunk
        segmentationMaskElements = np.unique(segmentation_masked)
        print(f"SERIALLY PROCESSING WITH  gH2AX__pixel_count_threshold with # unique labels in segmentation_masked: {len(segmentationMaskElements)}")
        start_time = time.time()
        for lbl in tqdm(segmentationMaskElements):
            if lbl == 0:
                continue
            roi = segmentation_masked == lbl
            if np.count_nonzero(roi) < gH2AX__pixel_count_threshold:
                segmentation_masked[roi] = 0
        elapsed_time = time.time() - start_time
        print(f'PROCESSED gH2AX__pixel_count_threshold in: {elapsed_time} seconds')

        ## Processing in parallel in batch
        print('FEATURE CALCULATION: ', end='')
        print(f'Number of unique labels in segmentation_masked: {len(np.unique(segmentation_masked))}')
        start_time = time.time()
        segmentation_gh2ax = np.zeros(segmentation.shape)
        tasks = []
        unique_labels = np.unique(segmentation_masked)
        # Collect tasks for parallel processing
        for lbl in tqdm(unique_labels):
            if lbl == 0:
                continue
            roi = segmentation == lbl
            segmentation_gh2ax[segmentation == lbl] = lbl
            mean_intensity_gh2ax[treatmentClass].append((gh2ax * roi).sum() / np.count_nonzero(roi))
            tasks.append((gh2ax, roi, lbl))
        print('Tasks gathering is finished')
        # Process in parallel
        # Optimize n_jobs parameter by getting the number of available CPU cores
        n_jobs = os.cpu_count() // 2
        print(f'os.cpu_count: {os.cpu_count()}, n_jobs: {n_jobs}')

        print('PARALLEL FEATURE PROCESSING in ...', end='')
        # Process in parallel using joblib
        results = Parallel(n_jobs=n_jobs)(delayed(compute_features_for_roi)(t[0], t[1], t[2]) for t in tqdm(tasks))
        elapsed_time = time.time() - start_time
        print(f' {elapsed_time} seconds')

        # Finding neighbors and writing results to file
        print('PARALLEL PROCESSING: FIND NEIGHBORS ', end='')
        start_time = time.time()
        neighbor_tasks = [(lbl, segmentation_masked) for lbl, _ in results]
        neighbor_results = Parallel(n_jobs=n_jobs)(delayed(find_neighbors)(t[0], t[1]) for t in tqdm(neighbor_tasks))
        elapsed_time = time.time() - start_time
        print(f"in {elapsed_time} seconds")

        print(f'SAVING RESULTS TO {GENERATED_FEATURE_FILE} ', end='')
        start_time = time.time()
        for (lbl_feature, props), (lbl_neighbor, neighbors) in zip(results, neighbor_results):
            # print(f'lbl_feature: {lbl_feature}, lbl_neighbor: {lbl_neighbor}')
            assert lbl_feature == lbl_neighbor, 'lbl_feature and lbl_neighbor are different'
            line = f"{segmentation_file},{lbl_feature},{','.join(map(str, props))}"
            line += f",{','.join(map(str, neighbors[0]))}\n"
            featureFile.write(line)
        elapsed_time = time.time() - start_time
        print(f'in {elapsed_time} seconds.')

        GH2AX_IMAGE_FILE = os.path.join(SEGMENTATION_PATH[idx], tile_file_root + '_gh2ax.tif')
        CD8_IMAGE_FILE = os.path.join(SEGMENTATION_PATH[idx], tile_file_root + '_cd8.tif')
        tifffile.imwrite(GH2AX_IMAGE_FILE, segmentation_gh2ax, compression='zlib', compressionargs={'level': 8})
        tifffile.imwrite(CD8_IMAGE_FILE, segmentation_cd8, compression='zlib', compressionargs={'level': 8})
        print(f'WROTE GH2AX image to: {GH2AX_IMAGE_FILE}')
        print(f'WROTE CD8 image to: {CD8_IMAGE_FILE}')

        # Append computed features to json files
        ORIGINAL_FEATURE_FILE = os.path.join(SEGMENTATION_PATH[idx], 'Cellpose3DGeometryAndFeatures', 'tile__H%03d_V%03d.tif_CELLPOSE-LABELS_seg.npy.pickle_.json' % (col,row))
        UPDATED_FEATURE_FILE = os.path.join(SEGMENTATION_PATH[idx], 'Cellpose3DGeometryAndFeatures', 'tile__H%03d_V%03d.tif_CELLPOSE-LABELS_seg.npy.pickle_w_gh2ax.json' % (col,row))
        try:
            with open(ORIGINAL_FEATURE_FILE) as originalFeatureFile:
                json_data = json.load(originalFeatureFile)
            gH2AX_vector = [0]*(len(np.unique(segmentation))-1)
            CD8_vector = [0]*(len(np.unique(segmentation))-1)
            for cellidx in np.unique(segmentation_gh2ax):
                gH2AX_vector[int(cellidx)-1] = 1
            for cellidx in np.unique(segmentation_cd8):
                CD8_vector[int(cellidx)-1] = 1
            json_data['is_gH2AX'] = gH2AX_vector
            json_data['is_CD8'] = CD8_vector
            # print(len(np.unique(segmentation)), len(json_data['segmentationConfidence']))
            with open(UPDATED_FEATURE_FILE) as updatedFeatureFile:
                json.dump(json_data, updatedFeatureFile)
        except FileNotFoundError:
            pass

        pct_cd8[treatmentClass].append((len(np.unique(segmentation_cd8))-1)/(len(np.unique(segmentation))-1))
        pct_gh2ax[treatmentClass].append((len(np.unique(segmentation_masked))-1)/(len(np.unique(segmentation))-1))

    # print(gh2ax_sum/gh2ax_ctr)
    featureFile.close()

print(f'CELL ANALYSIS COMPLETED IN {time.time() - start_time0} seconds')


# Save (for plotting) pct_cd8, pct_gh2ax

with open(PCT_GH2AX_FILE, 'w') as pctGH2AXFile:
    json.dump(pct_gh2ax, pctGH2AXFile)
with open(PCT_CD8_FILE, 'w') as pctCD8File:
    json.dump(pct_cd8, pctCD8File)
with open(MEAN_INTENSITY_GH2AX_FILE, 'w') as meanIntensityGH2AXFile:
    json.dump(mean_intensity_gh2ax, meanIntensityGH2AXFile)

