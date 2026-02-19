import tifffile
import xml.etree.ElementTree as ET
import numpy as np
import os
import time
# from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.cluster import KMeans
from tqdm import tqdm
import seaborn
import json
import umap
from imaris_ims_file_reader.ims import ims_reader
from config import SEGMENTATION_PATH, RAW_IMAGE_PATH, XML_FILES_PATH, CLASSES, gH2AX_threshold, cd8_threshold, gH2AX__pixel_count_threshold, N_PSEUDO_CLASSES, radii
from utils_multi import get_cd8_segmentation_by_dilation, compute_features_for_roi, find_neighbors, process_label, pseudo_class_to_heatmap
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

print('CELL ANALYSIS COMPLETED IN ')
elapsed_time = time.time() - start_time0
print(f' {elapsed_time} seconds')

"""
TODO:
    Save (for plotting) pct_cd8, pct_gh2ax
"""
with open(PCT_GH2AX_FILE, 'w') as pctGH2AXFile:
    json.dump(pct_gh2ax, pctGH2AXFile)
with open(PCT_CD8_FILE, 'w') as pctCD8File:
    json.dump(pct_cd8, pctCD8File)
with open(MEAN_INTENSITY_GH2AX_FILE, 'w') as meanIntensityGH2AXFile:
    json.dump(mean_intensity_gh2ax, meanIntensityGH2AXFile)


"""
print('STARTING Rest of the analysis')
start_time1 = time.time()
t,p = stats.ttest_ind(pct_gh2ax[CLASSES[0]],pct_gh2ax[CLASSES[1]],equal_var=False)
print('gamma H2AX+ cell ratio p-value:', p)

t,p = stats.ttest_ind(pct_cd8[CLASSES[0]],pct_cd8[CLASSES[1]],equal_var=False)
print('CD8+ cell ratio p-value:' ,p)

# gh2ax ratio
plt.boxplot([pct_gh2ax[i] for i in CLASSES], showfliers=False)
plt.ylim(0,0.15)
plt.xticks([1,2], CLASSES)
plt.title('gH2AX ratio')
plt.savefig('gh2ax_ratio.png')
plt.clf()

# cd8 ratio
plt.boxplot([pct_cd8[i] for i in CLASSES], showfliers=False)
# plt.ylim(0,0.15)
plt.xticks([1,2], CLASSES)
plt.title('cd8+ ratio')
plt.savefig('cd8_ratio.png')
plt.clf()

# mean gammaH2AX intensity
plt.boxplot([mean_intensity_gh2ax[i] for i in CLASSES], showfliers=False)
plt.xticks([1,2], CLASSES)
plt.title('Mean Intensity')
plt.savefig('mean_intensity_gh2ax.png') 
plt.clf()

################### gammaH2AX texture analysis #############################
print('Extracting gammaH2AX texture for analysis...')
start_time = time.time()
DATA = []
NEIGHBORS = []
CELLIDS = []
N_CLASS = {}
combine = True
scale = True
for idx,c in enumerate(CLASSES):
    f = open(c + '_texture_features_w_neighbors.csv', 'r')
    lines = f.readlines()
    d = [i.strip().split(',')[2:8] for i in lines]
    n = [[int(float(j)) for j in i.strip().split(',')[8:]] for i in lines]

    cids = [[c] + i.strip().split(',')[:2] for i in lines]

    N_CLASS[idx] = len(d)

    if combine:
        DATA += d
        CELLIDS += cids
        NEIGHBORS += n
    else:
        DATA.append(d)
        CELLIDS.append(cids)
        NEIGHBORS.append(n)

if combine:
    DATA = [np.array(DATA).astype(float)]
else:
    for i in DATA:
        DATA[i] = np.array(DATA[i].astype(float))

clf = KMeans(n_clusters=N_PSEUDO_CLASSES, algorithm='full', max_iter=1000, tol=1e-7)

for idx,data in enumerate(DATA):
    #scale to 0-1
    if scale:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)

    features = ['AutocorrelationFeatureValue', 'ClusterProminenceFeatureValue', 'ContrastFeatureValue', 'CorrelationFeatureValue', 'JointEnergyFeatureValue', 'IdFeatureValue']
    features_idx = [3,5,2,1,4,0]

    lbl = clf.fit_predict(data[:,features_idx])

    pca = PCA(n_components=2)
    data_transformed = pca.fit_transform(data[:,features_idx])
    print(data_transformed.shape)

    plt.scatter(data_transformed[:,0], data_transformed[:,1], c=lbl)
    plt.title('PCA')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.savefig((CLASSES[idx] if len(DATA)>1 else 'combined') + '_all_pca_500_700.png')

    umap = umap.UMAP(n_components=2)
    data_transformed = umap.fit_transform(data[:,features_idx])
    plt.clf()
    plt.scatter(data_transformed[:,0], data_transformed[:,1], c=lbl)
    plt.title('UMAP')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.savefig((CLASSES[idx] if len(DATA)>1 else 'combined') + '_all_umap_500_700.png')

    plt.clf()
    hmap = pseudo_class_to_heatmap(lbl[:N_CLASS[0]], CELLIDS[:N_CLASS[0]], NEIGHBORS[:N_CLASS[0]],n_classes=N_PSEUDO_CLASSES)
    seaborn.heatmap(hmap,vmin=0,vmax=0.7)
    plt.savefig('heatmap_treated.png')
    plt.clf()
    hmap = pseudo_class_to_heatmap(lbl[N_CLASS[0]:], CELLIDS[N_CLASS[0]:], NEIGHBORS[N_CLASS[0]:],n_classes=N_PSEUDO_CLASSES)
    seaborn.heatmap(hmap,vmin=0,vmax=0.7)
    plt.savefig('heatmap_control.png')

elapsed_time = time.time() - start_time
print(f"Time taken for the gammaH2AX texture analysis: {elapsed_time} seconds")
print('')

####################### gammaH2AX - CD8 proximity ##################################

print('Analyzing gammaH2AX and CD8 proximity...')
start_time = time.time()
neighbors_gh2ax = {}
neighbors_cd8 = {}
neighbors_random_gh2ax = {}
neighbors_random_cd8 = {}
for idx,c in enumerate(CLASSES):
    print(f'CLASS: {c}')
    tree = ET.parse(XML_FILES_PATH[idx])
    root = tree.getroot()
    neighbors_gh2ax[c] = []
    neighbors_random_gh2ax[c] = []
    for stack in root.iter('Stack'):
        col,row = int(stack.get('COL')), int(stack.get('ROW'))
        print(f'col: {col}, row: {row}')

        cd8_segment_file = 'tile__H%03d_V%03d_cd8.tif' % (col,row)
        gh2ax_segment_file = 'tile__H%03d_V%03d_gh2ax.tif' % (col,row)
        nuclei_segment_file = 'tile__H%03d_V%03d_CELLPOSE-LABELS_cp_masks.tif' % (col,row)

        print(f"READING: {cd8_segment_file=}, {gh2ax_segment_file=}, {nuclei_segment_file=}")

        cd8_segmentation = tifffile.imread(os.path.join(SEGMENTATION_PATH[idx], cd8_segment_file))
        gh2ax_segmentation = tifffile.imread(os.path.join(SEGMENTATION_PATH[idx], gh2ax_segment_file))
        nuclei_segmentation = tifffile.imread(os.path.join(SEGMENTATION_PATH[idx], nuclei_segment_file))

        labels = np.unique(gh2ax_segmentation)
        labels = labels[labels != 0]  # Exclude background label

        # Parallelize the label processing
        n_jobs = os.cpu_count() // 2  # Adjust based on your system
        print(f'os.cpu_count: {os.cpu_count()}, n_jobs: {n_jobs}')
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_label)(lbl, gh2ax_segmentation, cd8_segmentation, radii) for lbl in tqdm(labels))

        # Filter out None results and append to neighbors_gh2ax
        start_time_inner = time.time()
        for result in results:
            if result is not None:
                lbl, neighbors = result
                neighbors_gh2ax[c].append(neighbors)

        elapsed_time = time.time() - start_time_inner
        print(f"Time taken for finding neighbors of {c}, row: {row}, col: {col}: {elapsed_time} seconds")

        # random_labels = np.random.permutation(nuclei_segmentation.max())[:100] + 1
        # for lbl in random_labels:
        # 	n = get_neighbors(lbl, nuclei_segmentation, cd8_segmentation, radius=radii)
        # 	neighbors_random_gh2ax[c].append(n)
        # 	n = get_neighbors(lbl, nuclei_segmentation, gh2ax_segmentation, radius=radii)
        # 	neighbors_random_cd8[c].append(n)

for i in CLASSES:
    neighbors_gh2ax[i] = np.array(neighbors_gh2ax[i])
    np.savetxt('gh2ax_to_proximity_%s.csv'%(i), neighbors_gh2ax[i], delimiter=',')

t,p = stats.ttest_ind(neighbors_gh2ax[CLASSES[0]],neighbors_gh2ax[CLASSES[1]],equal_var=False)
print('gammaH2AX (control vs treated) u-test p-values:', p)

for idx,v in enumerate(radii):
    plt.boxplot([neighbors_gh2ax[i][:,idx].flatten()-1 for i in CLASSES], showfliers=False, meanline=True)
    # plt.ylim(0,0.15)
    plt.xticks([1,2], CLASSES)
    plt.ylabel('# of neighbor CD8+ cells')
    plt.title('gammaH2AX proximity to CD8 (%dpx)' % (v))
    plt.savefig('gammaH2AX proximity to CD8 (%dpx).png' % (v))
    plt.clf()

elapsed_time = time.time() - start_time
print(f"Time taken for the rest of the gammaH2AX - CD8 proximity analysis: {elapsed_time} seconds")
print('')

elapsed_time = time.time() - start_time0
print(f'Time taken for the entire analysis.py run: {elapsed_time} seconds')
"""


