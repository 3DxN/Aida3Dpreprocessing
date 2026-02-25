import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from time import time
import argparse
import json
import seaborn
import sys
import umap

from utils_multi import pseudo_class_to_heatmap

"""
INPUT: pct_cd8, pct_gh2ax, mean_intensity_gh2ax, CLASSES
"""
# Invoke for testing purposes: python src/proximityAnalysis/plotRatios.py pct_cd8_2026_02_19-11_24_40.json pct_gh2ax_2026_02_19-10_55_23.json mean_intensity_gh2ax_2026_02_19-11_24_40.json -c control control
parser = argparse.ArgumentParser(
                    prog='PlotRatios',
                    description='Plots ratios between CD8 and GH2AX',
                    epilog='Text at the bottom of help')
parser.add_argument('pct_cd8_json_file') 
parser.add_argument('pct_gh2ax_json_file') 
parser.add_argument('mean_intensity_gh2ax_json_file') 
parser.add_argument('-p','--plot_output_format', type=str, default='pdf') 
parser.add_argument('-c', '--classes', action='store', dest='classList',
                    type=str, nargs=2, default=['control', 'treated'],
                    help="Examples: -c control -c treated")
parser.add_argument('--num_pseudo_classes', type=int, default = 5)

args = parser.parse_args()

OUT_FILE_EXTENTION = args.plot_output_format
if not OUT_FILE_EXTENTION in ['pdf','png']:
    print(f"Unsupported file format for plot output: {OUT_FILE_EXTENTION}. Select png or pdf")
    sys.exit(1)

with open(args.pct_cd8_json_file, 'r') as pct_cd8_json_file, open(args.pct_gh2ax_json_file, 'r') as pct_gh2ax_json_file, open(args.mean_intensity_gh2ax_json_file, 'r') as mean_intensity_gh2ax_json_file:
    try:
        pct_cd8 = json.load(pct_cd8_json_file)
        pct_gh2ax = json.load(pct_gh2ax_json_file)
        mean_intensity_gh2ax = json.load(mean_intensity_gh2ax_json_file)

    except json.JSONDecodeError as ex:
        print("ERROR: Could not decode json: {ex}")
        sys.exit(1)

CLASSES= args.classList
if len(CLASSES)!=2:
    print("ERROR: Exactly two classes required for plotting. Specify with CLI parameters: -c class1 class2")
    sys.exit(1)
N_PSEUDO_CLASSES = args.num_pseudo_classes

print('CD8/GH2AX ratio analysis ...')
timestampstring = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
start_time1 = time()
t,p = stats.ttest_ind(pct_gh2ax[CLASSES[0]],pct_gh2ax[CLASSES[1]],equal_var=False)
print('gamma H2AX+ cell ratio p-value:', p)

t,p = stats.ttest_ind(pct_cd8[CLASSES[0]],pct_cd8[CLASSES[1]],equal_var=False)
print('CD8+ cell ratio p-value:' ,p)

# gh2ax ratio
plt.boxplot([pct_gh2ax[i] for i in CLASSES], showfliers=False)
#plt.ylim(0,0.15)
plt.xticks([1,2], CLASSES)
plt.title('gH2AX ratio')
plt.savefig(f'gh2ax_ratio_{timestampstring}.{OUT_FILE_EXTENTION}')
plt.clf()

# cd8 ratio
plt.boxplot([pct_cd8[i] for i in CLASSES], showfliers=False)
# plt.ylim(0,0.15)
plt.xticks([1,2], CLASSES)
plt.title('cd8+ ratio')
plt.savefig(f'cd8_ratio_{timestampstring}.{OUT_FILE_EXTENTION}')
plt.clf()

# mean gammaH2AX intensity
plt.boxplot([mean_intensity_gh2ax[i] for i in CLASSES], showfliers=False)
plt.xticks([1,2], CLASSES)
plt.title('Mean Intensity')
plt.savefig(f'mean_intensity_gh2ax_{timestampstring}.{OUT_FILE_EXTENTION}') 
plt.clf()




################### gammaH2AX texture analysis #############################
print('Extracting gammaH2AX texture for analysis...')
texture_analysis_start_time = time()
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

clf = KMeans(n_clusters=N_PSEUDO_CLASSES, algorithm='lloyd', max_iter=1000, tol=1e-7) #invalid parameter: algorithm='full' (must be in ['lloyd','elkan'])

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
    plt.savefig((CLASSES[idx] if len(DATA)>1 else 'combined') + f'_all_pca_500_700_{timestampstring}.{OUT_FILE_EXTENTION}')

    umap = umap.UMAP(n_components=2)
    data_transformed = umap.fit_transform(data[:,features_idx])
    plt.clf()
    plt.scatter(data_transformed[:,0], data_transformed[:,1], c=lbl)
    plt.title('UMAP')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.savefig((CLASSES[idx] if len(DATA)>1 else 'combined') + f'_all_umap_500_700_{timestampstring}.{OUT_FILE_EXTENTION}')

    plt.clf()
    hmap = pseudo_class_to_heatmap(lbl[:N_CLASS[0]], CELLIDS[:N_CLASS[0]], NEIGHBORS[:N_CLASS[0]],n_classes=N_PSEUDO_CLASSES)
    seaborn.heatmap(hmap,vmin=0,vmax=0.7)
    plt.savefig(f'heatmap_treated_{timestampstring}.{OUT_FILE_EXTENTION}')
    plt.clf()
    hmap = pseudo_class_to_heatmap(lbl[N_CLASS[0]:], CELLIDS[N_CLASS[0]:], NEIGHBORS[N_CLASS[0]:],n_classes=N_PSEUDO_CLASSES)
    seaborn.heatmap(hmap,vmin=0,vmax=0.7)
    plt.savefig(f'heatmap_control_{timestampstring}.{OUT_FILE_EXTENTION}')

print(f"GammaH2AX texture analysis in {time() - texture_analysis_start_time} seconds")
print('')
