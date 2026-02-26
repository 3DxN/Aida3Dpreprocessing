import math
import numpy as np
from scipy.spatial import distance_matrix, ConvexHull
from geometryProcessor import ellipsoid_fit
from termcolor import colored
#import itertools

"""
1. Morphological features:

1a. Volume: Number of segmented voxels, corrected for anisotropy.
1b. Elongation: Main axis length divided by arithmetic mean of remaining axes. Then: value < 1 -> oblate, value > 1 -> prolate. Value close to 1: Spherical if ratio between remaining axes is also close to 1 (expected to be the case).
1c. Orientation: Represented as direction of main axis, by convention pointing upward wrt. z-coordinates. Deemed useful for registering adjacent sections, less so for phenotyping.
1d. Diameter: Maximum length of segment within nucleus, e.g., double of length of longest semi-axis. (Perhaps not required for phenotyping, as volume and elongation in conjunction already express this information, but potentially useful for visualization purposes, as discussed in meeting on 16/3/2021.)
"""

# General method to compute volume from nucleus vertices, computionally heavy using qhull (http://www.qhull.org/)
# StarDist3D based method could be more efficient by summing tetrahedron (defined by triange face vertices and nucleus center) volumes.
def computeVolumes(nucleusCollectionVertices):
    nucleusVolumes = []    
    for nucleusVertices in nucleusCollectionVertices:
        #pVertexData = np.array([[x[0][0],x[0][1],x[0][2]] for x in nucleusVertices])
        pVertexData = np.array([[x[0],x[1],x[2]] for x in nucleusVertices])        
        hull = ConvexHull(pVertexData)               
        nucleusVolumes.append(hull.volume)
    return nucleusVolumes


"""
# Baseline implementation (slow) for diameter computation
# Rotating calipers method (https://en.wikipedia.org/wiki/Rotating_calipers) impractical for 3D case
def computeDiameters(npVertexDataList):
    nucleusDiameters = []
    for nucleusEnvelopeVertices in npVertexDataList:
        print('*',end=' ',flush=True)
        sqMaxDist=0

        factor = 100 # Increase accuracy when using ints
        intVerts = [[factor*int(x[0][0]),factor*int(x[0][1]),factor*int(x[0][2])] for x in nucleusEnvelopeVertices]

        for vertex1,vertex2 in itertools.combinations(intVerts, 2):
            v1x_v2x=vertex1[0]-vertex2[0]
            v1y_v2y=vertex1[1]-vertex2[1]
            v1z_v2z=vertex1[2]-vertex2[2]                                
            sqPointDist = (v1x_v2x)*(v1x_v2x) + (v1y_v2y)*(v1y_v2y) + (v1z_v2z)*(v1z_v2z)                
            if sqPointDist > sqMaxDist:
                sqMaxDist = sqPointDist                           
        
        nucleusDiameters.append(math.sqrt(sqMaxDist)/factor)
        
        #parallel loop:
       
        #from joblib import Parallel, delayed
        #def yourfunction(k):   
        #    s=3.14*k*k
        #    print "Area of a circle with a radius ", k, " is:", s
        #
        #element_run = Parallel(n_jobs=-1)(delayed(yourfunction)(k) for k in range(1,10))

    return nucleusDiameters
"""

def computeDiametersNP(npVertexDataList):    
    nucleusDiameters = []
    factor = 100 # Increase accuracy when using ints        
    for nucleusEnvelopeVertices in npVertexDataList:
        #intVerts = np.array([[factor*int(x[0][0]),factor*int(x[0][1]),factor*int(x[0][2])] for x in nucleusEnvelopeVertices])
        intVerts = np.array([[factor*int(x[0]),factor*int(x[1]),factor*int(x[2])] for x in nucleusEnvelopeVertices])        
        nucleusDiameters.append(np.max(distance_matrix(intVerts,intVerts))/factor)        
    return nucleusDiameters
    
    
# Compute centroid, axes and radii of fitted ellipsoid.
# Using https://github.com/marksemple/pyEllipsoid_Fit, 
# following Yury Petrov's method: https://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit
# Following encoding conventions described here: https://math.stackexchange.com/questions/2816986/plotting-an-ellipsoid-using-eigenvectors-and-eigenvalues     
# Alternative implementation, including visualisation: https://github.com/aleksandrbazhin/ellipsoid_fit_python  
def computeEllipsoidFits(nucleusCollectionVertices):
    ellipsoids = []
    for nucleusVertices in nucleusCollectionVertices:
        #pVertexData = np.array([[x[0][0],x[0][1],x[0][2]] for x in nucleusVertices])
        pVertexData = np.array([[x[0],x[1],x[2]] for x in nucleusVertices])    
        ellipsoid = ellipsoid_fit.fit(pVertexData) # (centre, evecs, radii)                          
        #print('EL', ellipsoid.centre, ellipsoid.evecs, ellipsoid.radii)
        ellipsoids.append(ellipsoid)
    return ellipsoids


#Params: Three semi axis length (radii)
def ellipsoidElongation(a,b,c): 
    diffAB = abs(a-b)
    diffAC = abs(a-c)
    diffBC = abs(b-c)    
    remainingAxis, mergedAxis, mainAxisIndex  = 0, 0, -1
    
    # Case: remaining axis a
    if diffBC < diffAC and diffBC < diffAB:
        remainingAxis = a
        mainAxisIndex = 0
        mergedAxis = (b+c)/2        

    # Case: remaining axis B
    if diffAC < diffBC and diffAC < diffBC:
        remainingAxis = b
        mainAxisIndex = 1
        mergedAxis = (a+c)/2        

    # Case: remaining axis c
    if diffAB < diffAC and diffAB < diffBC:
        remainingAxis = c
        mainAxisIndex = 2
        mergedAxis = (a+b)/2       
        
    elongation = abs(remainingAxis / mergedAxis) 
        
    return abs(remainingAxis / mergedAxis), mainAxisIndex 


def computeBiAxialElongations(ellipsoids):
    elongations, mainAxisIndices, longestAxisIndices = [], [], []
    for ellipsoid in ellipsoids:
        radii = ellipsoid[2]
        SX,SY,SZ = radii[0],radii[1],radii[2]
        evecs = ellipsoid[1] #ellipsoid axes (unit vector)
        
        r1 = np.linalg.norm([SX*evecs[0][0], SX*evecs[1][0], SX*evecs[2][0]])
        r2 = np.linalg.norm([SY*evecs[0][1], SY*evecs[1][1], SY*evecs[2][1]])
        r3 = np.linalg.norm([SZ*evecs[0][2], SZ*evecs[1][2], SZ*evecs[2][2]])
        
        longestElementIndex = np.array([r1,r2,r3]).argmax()
        longestAxisIndices.append(int(longestElementIndex))
        
        ellipsoidElongationValue, mainAxisIndex = ellipsoidElongation(r1,r2,r3)
        mainAxisIndices.append(mainAxisIndex)
        elongations.append(ellipsoidElongationValue)
        
    return elongations, mainAxisIndices, longestAxisIndices    

