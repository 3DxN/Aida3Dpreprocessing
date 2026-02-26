import math
import numpy as np
from sklearn.neighbors import NearestNeighbors

"""
3. Contextual features (kNN): Concatenation of features (3a)-(3d) for k nearest nuclei (Euclidean distance). Question: How many neighbors are optimal? Jens indicated up to 10.

3a. Distance: Euclidean distance between nucleus centers in voxel units.
3b. Volume: As defined in (1a).
3c. Elongation: As defined in (1b).
3d. relative Orientation: Angle between main axes of nucleus and neighbor nucleus under consideration; a scalar value in range [0,90] degrees. Using orientations defined in (1c). The rationale of this simplicifation is to make features invariant to the orienation of the local coordinate system, and invariant to permutation of axis direction, while retaining a measure for parallelism in nucleus alignment. This assumes the difference between skew axes and (almost) intersecting axes is impertinent to classifying nucleus alignments. 
3e. absolute Orientation: local nucleus coordinate system triads, adjusted for semi-acis lengths
"""


# Returns the unit vector of the vector. 
def unit_vector(vector):    
    return vector / np.linalg.norm(vector)


# Returns the angle in radians between vectors 'v1' and 'v2'::
def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.inner(v1_u, v2_u).item(), -1.0, 1.0))


"""
#TODO: filter margins in XY plane, but retain indices (but not necessarily required at this stage; it can be done at the end, there's little overhead choosing this option)
"""

"""From the 3 orientation vectors describing the nucleus ellisoid fit semiaxis, choose one:
Either the longest one, or the 'main axis'""" 
# # 
def selectOrientationAxis(orientationAxisTriplet):
    o1,o2,o3 = orientationAxisTriplet[0],orientationAxisTriplet[1],orientationAxisTriplet[2]
    return max(np.linalg.norm(o1),np.linalg.norm(o2),np.linalg.norm(o3))


def selectMainAxis(orientationTriplet,mainAxisIndex):
    return orientationTriplet[mainAxisIndex]


def euclLength(v):
    return math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])


# Function returns scores describing mesenchymal or epithelial character based on context.
# Larger return value (not exceeding 1) -> higher probability of mesenchymal or epithelial respectively
def computeCellTypeScores(neighborTranslation,currentCellOrientation,neighborCellOrientation,angle):
    mesenchymalScore, epithelialScore = 0.0, 0.0
    # translation vector more orthogonal to orientation vectors -> rather epithelial
    # translation vector more parallel to orientation vectors -> rather mesenchymal
    
    angleWeight = 1-(angle/90)
    neighborDist = euclLength(neighborTranslation) ## Use value from knn graph directly?
    if neighborDist < 1: #avoid nan result in computaion of angle_between when distance is zero
        return 0,0        
    distanceWeight = 1 / neighborDist #if neighborDist > 1 else 0
    translationVectorWeight = angle_between(neighborTranslation,currentCellOrientation)/3.1415

    mesenchymalScore = angleWeight * distanceWeight * (1.0 - translationVectorWeight)
    epithelialScore = angleWeight * distanceWeight * translationVectorWeight
    
    return mesenchymalScore, epithelialScore


# Returns concatenated neighborhood features for each nucleus, as describe above (3)
def constructTissueNeighborhoodGraph(nucleusLocations, nucleusVolumes, nucleusElongations, nucleusOrientations,\
        mainAxisIndices, desiredNumberOfNeighbors = 10): 
    # nucleusLocations expected to have original anisotropy
    # Following https://scikit-learn.org/stable/modules/neighbors.html#unsupervised-neighbors:
    
    numNeighbors = min(len(nucleusLocations), desiredNumberOfNeighbors)

    neighbrs = NearestNeighbors(n_neighbors=numNeighbors, algorithm='ball_tree').fit(nucleusLocations)
    distances, indices = neighbrs.kneighbors(nucleusLocations)

    contextFeatureArray = [] #Each entry corresponds to exactly one nucleus, but refers to n_neighbors nuclei.

    for i, (cellIdx, celldistances) in enumerate(zip(indices,distances)):
        singleCellContextFeatures = []
        currentCellOrientations = nucleusOrientations[i]
        currentNucleusLocation = nucleusLocations[i]
        currentCellOrientation = selectMainAxis(currentCellOrientations,mainAxisIndices[i])
        neighborFeatureArray = []
        for (neighborIdx,distanceToNeighbor) in zip(cellIdx,celldistances): 
            neighborCellOrientationS = nucleusOrientations[neighborIdx]  
            neighborNucleusLocation = nucleusLocations[neighborIdx]
            neighborCellOrientation = selectMainAxis(neighborCellOrientationS,mainAxisIndices[neighborIdx])
            #print("neighborCellOrientation ", neighborCellOrientation)
            #angle in [0,90] degrees! 
            angle = round((angle_between(currentCellOrientation,neighborCellOrientation) * (180/3.14159265)),1)
            angle = angle if angle < 90 else 180.0 - angle            
            neighborTranslation = np.array(neighborNucleusLocation) - np.array(currentNucleusLocation)

            #larger -> higher probability of mesenchymal or epithelial
            mesenchymalScore, epithelialScore = computeCellTypeScores(neighborTranslation,\
                                    currentCellOrientation,neighborCellOrientation,angle)
            
            singleCellContextFeatures = (distanceToNeighbor,\
                                         nucleusVolumes[neighborIdx],\
                                         nucleusElongations[neighborIdx],\
                                         angle,\
                                         mesenchymalScore,\
                                         epithelialScore,\
                                         nucleusOrientations[neighborIdx]) #All 3 orientation, no main axis selection.
                                         
            neighborFeatureArray.append(singleCellContextFeatures)
                
        contextFeatureArray.append(neighborFeatureArray)

    return contextFeatureArray 
    

# Function to select nuclei with a well defined orientation.
# Returns true, if 1 semiaxis is significantly longer than the other two; default threshold is 0.8.
def isElongated(elongationValue, relativeThreshold = 0.8):
    return elongationValue < relativeThreshold or elongationValue > 1 / relativeThreshold
       
       
def computeTissueInterfaceLikelihoods(contextFeatureArray):

    tissueInterfaceLikelihoodArray = []
    for nucleus in contextFeatureArray:
        tissueInterfaceLikelihood = 0    
        if len(contextFeatureArray) < 1:
            tissueInterfaceLikelihoodArray.append(tissueInterfaceLikelihood)
            continue
        currentNucleusFeatures = contextFeatureArray[0] 
        if not isElongated(currentNucleusFeatures[2]):
            tissueInterfaceLikelihoodArray.append(tissueInterfaceLikelihood)
            continue    

        for i,neighborNucleus in enumerate(nucleus):
            #First nucleus is 'self' - not required
            if i == 0:
                continue
        
        #1) Compute table of relative orientations: neigborIdx -> relOrientation    
        
        #2) Fit bi-model distribution. One mode should have mean close to 0, the other one at least 50?,60?,70? degrees away
                
        #3) Simple heuristic: 
       
        tissueInterfaceLikelihoodArray.append(tissueInterfaceLikelihood)
    
    return tissueInterfaceLikelihoodArray
  

# Compute a score for the likelihood that a nucleus is part of an arrangement
# lined of in a 'string' of nuclei, i.e, neighboring nuclei in a line, facing pole-to-pole (mesenchymal)
# or lined up side-by-side (epithelial)
# For each nucleus, filters a subset of non-round nuclei with similar orientation
# Score derived from this subset: Sum of scores of all neighbors with suitable elongation.  
def aggregateCelltypeScores(contextFeatureArray):
    mesenchymalScores = []
    epithelialScores = []
    for nucleusNeighborsArray in contextFeatureArray:
        # Note: contextFeatureArray[0] is nucleus 'self', i.e., not a neighbor
        selfNucleus = nucleusNeighborsArray[0]
        selfElongation = selfNucleus[2]
        
        mesenchymalScoreAggregate = 0
        epithelialScoreAggregate = 0
        if not isElongated(selfElongation):
            mesenchymalScores.append(mesenchymalScoreAggregate)
            epithelialScores.append(epithelialScoreAggregate)
            continue
            
        for neighborNucleus in nucleusNeighborsArray[1:]:            
            #neighborDistance = neighborNucleus[0]
            neighborElongation = neighborNucleus[2]
            #neighborRelativeOrientation = neighborNucleus[3]
            mesenchymalScore = neighborNucleus[4]
            epithelialScore = neighborNucleus[5]
            
            if isElongated(neighborElongation):
                mesenchymalScoreAggregate = mesenchymalScoreAggregate + mesenchymalScore
                epithelialScoreAggregate = epithelialScoreAggregate + epithelialScore
                        
        mesenchymalScores.append(mesenchymalScoreAggregate)
        epithelialScores.append(epithelialScoreAggregate)
        
    return mesenchymalScores, epithelialScores
       
       
       
