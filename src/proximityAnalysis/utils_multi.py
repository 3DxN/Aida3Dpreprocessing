from radiomics.glcm import RadiomicsGLCM as glcm
import SimpleITK as sitk
from scipy import ndimage
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.spatial.distance import cdist
import numpy as np
from config import *
from tqdm import tqdm
from skimage.measure import regionprops, label


def get_bounding_boxes(image):
	props = regionprops(image)
	bboxes = {}
	for prop in props:
        # prop.bbox = (min_row, min_col, max_row, max_col) for 2‑D
        # 3D result: (min_z, min_row, min_col, max_z, max_row, max_col)
		bboxes[prop.label] = prop.bbox
	return bboxes

# gateway functions to compute features for roi using in parallel
# Parallelize neighbor finding
def numbers_of_neighbors_within_radius(lbl, gh2ax_segmentation, cd8_segmentation, radii):
    if lbl == 0:
        return None
    n = get_neighbors(lbl, gh2ax_segmentation, cd8_segmentation, radius=radii)
    return lbl, [len(i) for i in n]

def find_neighbors(lbl, segmentation_masked):
    neighbors = get_neighbors(lbl, segmentation_masked)
    return lbl, neighbors

def compute_features_for_roi(img, roi, lbl):
    props = get_features(img, roi).flatten()
    return lbl, props

def process_gH2AX(segmentation_masked_chunk, labels_chunk, threshold):
    results = []
    for lbl in labels_chunk:
        roi = segmentation_masked_chunk == lbl
        if np.count_nonzero(roi) < threshold:
            results.append((lbl, roi))
        else:
            results.append((lbl, None))
    return results
   
# compute glcm features
def get_features(img, roi):
	# --- START: Added Image Normalization Algorithm ---
    # To prevent memory errors caused by high dynamic range, this section
    # normalizes the image intensities using the Z-score method. The statistics
    # (mean and standard deviation) are calculated *only* from the voxels
    # within the provided Region of Interest (ROI) to ensure that the
    # normalization is relevant to the tissue being analyzed.

    # 1. Extract the intensity values from the voxels within the ROI mask.
    roi_voxels = img[roi > 0]

    # 2. Calculate the mean and standard deviation of the ROI voxels.
    # This check prevents errors if the ROI is unexpectedly empty.
    if roi_voxels.size > 0:
        mean_roi = np.mean(roi_voxels)
        std_roi = np.std(roi_voxels)
    else:
        mean_roi = 0
        std_roi = 1.0 # Avoid division by zero in the unlikely case of an empty ROI

    # 3. Apply the Z-score normalization to the entire image array.
    # We check if std_roi is close to zero to avoid division by zero errors,
    # which can happen if the ROI contains voxels of a single intensity value.
    if std_roi > 1e-6:
        # The original img_array is overwritten with the normalized version.
        img = (img - mean_roi) / std_roi
    else:
        # If standard deviation is zero, all values are the mean.
        # The normalized image becomes an array of zeros.
        img = img - mean_roi
    
    print(f"DEBUG: Image normalized using ROI mean={mean_roi:.2f}, std={std_roi:.2f}")
    # --- END: Added Image Normalization Algorithm ---

    img = sitk.GetImageFromArray(img, isVector=True)
    roi = sitk.GetImageFromArray(roi.astype(int), isVector=True)
    g = glcm(inputImage=img, inputMask=roi, distances=[1,3,5,7], dirs=[1,2,3,4,5,6,7,8], normed=False, bins=256)
    g.enableAllFeatures()
# edits on 07/28/25  See email from Gefei on 7/25/25
#g.execute()
    # compute props
#	props = [g.getAutocorrelationFeatureValue(), g.getClusterProminenceFeatureValue(), g.getContrastFeatureValue(), g.getCorrelationFeatureValue(), g.getJointEnergyFeatureValue(), g.getIdFeatureValue()]
    try:
        g.execute()
    except:
        props=[0,0,0,0,0,0]
    else:
        props = [g.getAutocorrelationFeatureValue(), g.getClusterProminenceFeatureValue(), g.getContrastFeatureValue(), g.getCorrelationFeatureValue(), g.getJointEnergyFeatureValue(), g.getIdFeatureValue()]
    return np.array(props)

# create circular neighborhood(s) centered on the specified cell (lbl)
def get_neighbors(lbl, img, img2=None, radius=[64]):
	roi = img==lbl

	# get centroid
	zc,yc,xc = ndimage.measurements.center_of_mass(roi)

	# draw sphere with given radius
	z,y,x = np.mgrid[0:img.shape[0]:1, 0:img.shape[1]:1, 0:img.shape[2]:1]

	neighbors = []
	for r in radius:
		mask = np.sqrt((z - zc)**2 + (y - yc)**2 + (x - xc)**2)
		mask[mask > r] = 0
		mask[mask > 0] = 1

		# use circle as mask
		neighborhood = (img if img2 is None else img2)*mask
		if img2 is None:
			neighborhood[neighborhood==lbl] = 0

		# return neighbors
		#neighbors.append(len(np.unique(neighborhood)))
		neighbors.append(np.unique(neighborhood))

	return neighbors

"""
Instead of defining neighborhood criterium by distance from centroid of label in image 1 to any point of label in image 2 (function `get_neighbors`):
May replace this by working with centroids of labels in two images (initially) - and correct by, e.g., diameter of labels if necessary (likely it is not).
Note: Search restricted to segmentation labels actually identified as corresponding to, e.g., CD8 or GH2AX
"""
def get_neighbors_by_centroids(im1, im2, radius=[64]):

	#im1_bboxes=get_bounding_boxes(im1)
	#im2_bboxes=get_bounding_boxes(im2)	

	regions1 = regionprops(label(im1)) 
	regions2 = regionprops(label(im2))


	pairwise_distances = {}

	for r1 in regions1:
		id1 = r1.label
		coords1 = r1.coords  # (N1, 2) array of (row, col)

		pairwise_distances[id1] = {}

		for r2 in regions2:
			id2 = r2.label
			coords2 = r2.coords  # (N2, 2)

			# Compute all pairwise distances between pixels of the two regions
			d = cdist(coords1, coords2)  # shape (N1, N2)

			# Shortest distance between the two regions
			min_dist = d.min()

			pairwise_distances[id1][id2] = min_dist

	print("PAIRWISE DISTANCES")
	for id1, row in pairwise_distances.items(): 
		for id2, dist in row.items(): 
			print(f"Region {id1} (img1) → Region {id2} (img2): {dist:.2f} pixels")



	"""
	roi = im1==lbl

	# get centroid
	zc,yc,xc = ndimage.measurements.center_of_mass(roi)

	# draw sphere with given radius
	z,y,x = np.mgrid[0:im1.shape[0]:1, 0:im1.shape[1]:1, 0:im1.shape[2]:1]

	neighbors = []
	for r in radius:
		mask = np.sqrt((z - zc)**2 + (y - yc)**2 + (x - xc)**2)
		mask[mask > r] = 0
		mask[mask > 0] = 1

		# use circle as mask
		neighborhood = (im1 if im2 is None else im2)*mask
		if im2 is None:
			neighborhood[neighborhood==lbl] = 0

		# return neighbors
		#neighbors.append(len(np.unique(neighborhood)))
		neighbors.append(np.unique(neighborhood))

	return neighbors
	"""


"""
# Superseeded by `get_cd8_segmentation_by_dilation`

from concurrent.futures import ProcessPoolExecutor

def process_chunk(chunk, segmentation, cd8_mask, cd8_dapi_box_overlap_threshold):
	results = []
	#print(f'{chunk=}')
	for lbl in chunk:
		if lbl == 0:
			continue
		roi = segmentation == lbl
		roi_dilated = binary_dilation(roi, iterations=3)
		roi_eroded = binary_erosion(roi, iterations=3)
		roi_boundary = np.logical_xor(roi_dilated, roi_eroded)
		cd8_roi = cd8_mask * roi_boundary
		cover = np.count_nonzero(cd8_roi) / np.count_nonzero(roi_boundary)
		if cover < cd8_dapi_box_overlap_threshold:
			results.append((lbl, 0))
		else:
			results.append((lbl, lbl))
	return results


def get_cd8_segmentation_by_dilation(segmentation, cd8_mask, num_chunks=16):
	cd8_nuclei_segmentation = segmentation.copy()
	unique_labels = np.unique(segmentation)

	print(f'Number of unique labels: {len(unique_labels)}')

	chunks = np.array_split(unique_labels, num_chunks)

	with ProcessPoolExecutor() as executor:
		futures = [executor.submit(process_chunk, chunk, segmentation, cd8_mask, cd8_dapi_box_overlap_threshold) for
		           chunk in tqdm(chunks)]
		results = []
		for future in futures:
			results.extend(future.result())

	for lbl, result in results:
		if lbl is not None:
			cd8_nuclei_segmentation[segmentation == lbl] = result

	return cd8_nuclei_segmentation

"""


def get_cd8_segmentation_by_dilation(segmentation, cd8_mask):
	cd8_nuclei_segmentation = segmentation.copy()
	labelIDs = np.unique(segmentation)
	print(f'CD8 analysis with segmentation label image. Number of segments: {len(labelIDs)-1}')
	## Faster version of function `get_cd8_segmentation_by_dilation`. 
	## 500x speed up by restricting morphological ops to neighborhood of labels.
	# INPUT: semantic segmentation label map, CD8 mask (thresholded CD8 image)
	# OUTPUT: - list of labels, where there is sufficient overlap between label envelope (i.e., CD8 positive) 
	#		  - label image retaining only segments where CD8 is positive
	retainedLabels = []	
	
	props = regionprops(segmentation)
	bboxes = {}
	for prop in props:
        # prop.bbox = (min_row, min_col, max_row, max_col) for 2‑D
        # 3D result: (min_z, min_row, min_col, max_z, max_row, max_col)
		bboxes[prop.label] = prop.bbox

	for lbl in labelIDs[1:]: # Value 0 corresponds to background label (by convention)
		# Crop segmentation by bounding box, then extend for dilation
		DL = 3 #DILATION_RANGE
		
		bbox = bboxes[lbl]
		dilated_bbox = (bbox[0]-DL,bbox[1]-DL,bbox[2]-DL,bbox[3]+DL,bbox[4]+DL,bbox[5]+DL)
		xmin,ymin,zmin,xmax,ymax,zmax=dilated_bbox

		# Crop at margins, no padding
		xmin,ymin,zmin = max(0,xmin), max(0,ymin), max(0,zmin)
		segmentationImgDim= segmentation.shape
		xmax,ymax,zmax = min(segmentationImgDim[0],zmax),min(segmentationImgDim[1],ymax),min(segmentationImgDim[2],zmax)
	
		segmentation_crop_with_margin =  segmentation[xmin:xmax,ymin:ymax,zmin:zmax]
		cd8_mask_crop_with_margins = cd8_mask[xmin:xmax,ymin:ymax,zmin:zmax]

		roi = segmentation_crop_with_margin==lbl
		
		if segmentation_crop_with_margin.size == 0:
			pass #continue
		roi_dilated = binary_dilation(roi, iterations=3)
		roi_eroded = binary_erosion(roi, iterations=3)
		roi_boundary = np.logical_xor(roi_dilated,roi_eroded)
		cd8_roi = cd8_mask_crop_with_margins*roi_boundary
		roi_boundary_count = np.count_nonzero(roi_boundary)
		if roi_boundary_count == 0: # avoid div0
			continue 
		overlap = np.count_nonzero(cd8_roi)/roi_boundary_count
		###print(overlap, end=' ')
		if overlap > cd8_dapi_box_overlap_threshold:
			print(f"Label retained: {lbl}")  # Labels to retain for the CD8 segment image!
			retainedLabels.append(lbl)
	
	# Generate segmentation label image where only `retainedLabels` values are present		
	mask = np.isin(segmentation, retainedLabels)
	cd8_nuclei_segmentation = np.full_like(segmentation, fill_value=0)
	cd8_nuclei_segmentation[mask] = segmentation[mask]

	return cd8_nuclei_segmentation


def pseudo_class_to_histogram_neighbors(pseudo_labels, cellids, neighbors, n_classes=None, normalize=False):
	if not n_classes:
		n_classes = len(np.unique(pseudo_labels))

	histogram = np.zeros((len(cellids),n_classes))

	neighbor_indices = {}
	for i in range(len(cellids)):
		neighbor_indices['_'.join(cellids[i])] = i

	for idx,cellid in enumerate(cellids):
		for neighbor in neighbors[idx]:
			if neighbor == 0:
				continue
			n_idx = neighbor_indices[cellid[0] + '_' + cellid[1] + '_' + str(neighbor)]
			histogram[idx,pseudo_labels[n_idx]] += 1

	return histogram

def pseudo_class_to_heatmap(pseudo_labels, cellids, neighbors, n_classes=None, normalize=False):
	if not n_classes:
		n_classes = len(np.unique(pseudo_labels))
	heatmap = np.zeros((n_classes,n_classes))

	neighbor_indices = {}
	for i in range(len(cellids)):
		neighbor_indices['_'.join(cellids[i])] = i

	for idx,cellid in enumerate(cellids):
		for neighbor in neighbors[idx]:
			if neighbor == 0:
				continue
			n_idx = neighbor_indices[cellid[0] + '_' + cellid[1] + '_' + str(neighbor)]
			heatmap[pseudo_labels[n_idx],pseudo_labels[idx]] += 1

	heatmap /= heatmap.sum()

	return heatmap
