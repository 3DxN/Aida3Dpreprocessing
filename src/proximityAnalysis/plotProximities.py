import argparse
import json
import os
import sys
import tifffile
import time
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime
from joblib import Parallel, delayed
from scipy import stats
from tqdm import tqdm

from config import SEGMENTATION_PATH, XML_FILES_PATH
from utils_multi import numbers_of_neighbors_within_radius,     get_neighbors_by_centroids
####################### gammaH2AX - CD8 proximity ##################################


parser = argparse.ArgumentParser(
                    prog='PlotProximities',
                    description='Plots proximities - CD8 and GH2AX',
                    epilog='Text at the bottom of help')
parser.add_argument('-p','--plot_output_format', type=str, default='png') 
parser.add_argument('-c', '--classes', action='store', dest='classesList',
                    type=str, nargs='*', default=['control', 'treated'],
                    help="Examples: -c control -c treated")
parser.add_argument('-r', '--radii', action='store', dest='radiiList',
                    type=int, nargs='*', default=[32,48,64,128],
                    help="Examples: -r 32 48 64 128")
args = parser.parse_args()

OUT_FILE_EXTENTION = args.plot_output_format
if not OUT_FILE_EXTENTION in ['pdf','png']:
    print(f"Unsupported file format for plot output: {OUT_FILE_EXTENTION}. Select png or pdf")
    sys.exit(1)

CLASSES = args.classesList
radii = args.radiiList
print(f"{CLASSES=} {radii=}")
"""
NOTES:
Code counts numbers of CD8 positive segments with a set of distances from GH2AX positive segment, for each GH2AX positve segments.
Based on label images. Computation per tile individually.

CONSIDER:
- Instead of using label segments, compute distances between centroids of label. Results are slightly different, but computation is much faster. 
- To improve statistics, register all centroids above into global coordinate frame. Can this be done with sufficient accuracy based on instrument encoder readings?
"""

print('Analyzing gammaH2AX and CD8 proximity...')
timestampstring = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
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

        ####TODO remove!
        # time1 = time.time()
        # print("Neighborhood computation ...")
        # get_neighbors_by_centroids(gh2ax_segmentation, cd8_segmentation)
        # print(f"Neighbors computed in {time.time()-time1} sec.")

        gh2ax_labels = np.unique(gh2ax_segmentation)
        gh2ax_labels = gh2ax_labels[gh2ax_labels != 0]  # Exclude background label

        # Parallelize the label processing
        n_jobs = os.cpu_count() // 2  # Adjust based on your system
        print(f'os.cpu_count: {os.cpu_count()}, n_jobs: {n_jobs}')
        results = Parallel(n_jobs=n_jobs)(
            delayed(numbers_of_neighbors_within_radius)(lbl, gh2ax_segmentation, cd8_segmentation, radii) for lbl in tqdm(gh2ax_labels))

        # Filter out None results and append to neighbors_gh2ax
        start_time_inner = time.time()
        for result in results:
            if result is not None:
                lbl, neighbors = result
                neighbors_gh2ax[c].append(neighbors)

        elapsed_time = time.time() - start_time_inner
        print(f"Time taken for finding neighbors of {c}, row: {row}, col: {col}: {elapsed_time} seconds")

        # nuclei_segmentation = tifffile.imread(os.path.join(SEGMENTATION_PATH[idx], nuclei_segment_file))
        #
        # random_labels = np.random.permutation(nuclei_segmentation.max())[:100] + 1
        # for lbl in random_labels:
        # 	n = get_neighbors(lbl, nuclei_segmentation, cd8_segmentation, radius=radii)
        # 	neighbors_random_gh2ax[c].append(n)
        # 	n = get_neighbors(lbl, nuclei_segmentation, gh2ax_segmentation, radius=radii)
        # 	neighbors_random_cd8[c].append(n)


# New output for CD8 neighbors - note: all classes must have unique names to prevent data loss (json/dict cannot hold multiple keys with same name)
with open(f'gh2ax_cd8_neighbor_proximity_{timestampstring}.json', 'w') as fp:
    json.dump(neighbors_gh2ax, fp)

# Old output format for CD8 neighbors:
for i in CLASSES:
    neighbors_gh2ax[i] = np.array(neighbors_gh2ax[i])
    np.savetxt('gh2ax_to_proximity_%s_{timestampstring}.csv'%(i), neighbors_gh2ax[i], delimiter=',')

t,p = stats.ttest_ind(neighbors_gh2ax[CLASSES[0]],neighbors_gh2ax[CLASSES[1]],equal_var=False)
print(f"Result of T-Test (scipy stats.ttest_ind)")
print("DATA:")
print(f"{neighbors_gh2ax[CLASSES[0]]=}")
print(f"{neighbors_gh2ax[CLASSES[1]]=}")
print('gammaH2AX (control vs treated) u-test p-values:', p)

for idx,radius in enumerate(radii):
    plt.boxplot([neighbors_gh2ax[i][:,idx].flatten()-1 for i in CLASSES], showfliers=False, meanline=True)
    # plt.ylim(0,0.15)
    plt.xticks([1,2], CLASSES)
    plt.ylabel('# of neighbor CD8+ cells')
    plt.title('gammaH2AX proximity to CD8 (radius %dpx)' % (radius))
    plt.savefig(f'gammaH2AX proximity to CD8 (radius %dpx)_{timestampstring}.{OUT_FILE_EXTENTION}' % (radius))
    plt.clf()

elapsed_time = time.time() - start_time
print(f"Time taken for the rest of the gammaH2AX - CD8 proximity analysis: {elapsed_time} seconds")
print('')
