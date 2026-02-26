import numpy as np
import math
from numpy.linalg import eig, inv
#from scipy.spatial.transform import Rotation as R

# http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/
# Includes treatment of singularities!
def rotMatToAxisAngle(m):
    epsilon = 0.01 # margin to allow for rounding errors
    epsilon2 = 0.1 # margin to distinguish between 0 and 180 degrees
    x, y, z = 0, 0, 0
    if abs(m[0][1]-m[1][0])< epsilon and abs(m[0][2]-m[2][0])< epsilon and abs(m[1][2]-m[2][1])< epsilon :
        if abs(m[0][1]+m[1][0]) < epsilon2 and abs(m[0][2]+m[2][0]) < epsilon2 and \
            abs(m[1][2]+m[2][1]) < epsilon2 and abs(m[0][0]+m[1][1]+m[2][2]-3) < epsilon2 :
            return 0,1,0,0, True # zero angle, arbitrary axis
        angle = math.pi;
        xx = (m[0][0]+1)/2
        yy = (m[1][1]+1)/2
        zz = (m[2][2]+1)/2
        xy = (m[0][1]+m[1][0])/4
        xz = (m[0][2]+m[2][0])/4
        yz = (m[1][2]+m[2][1])/4
        if xx > yy and xx > zz : 
            if xx< epsilon :
                x = 0
                y = 0.7071
                z = 0.7071
            else :
                x = math.sqrt(xx)
                y = xy/x
                z = xz/x
        else : 
            if yy > zz :
                if yy< epsilon :
                    x = 0.7071
                    y = 0
                    z = 0.7071
                else :
                    y = math.sqrt(yy)
                    x = xy/y
                    z = yz/y
            else :
                if zz< epsilon :
                    x = 0.7071
                    y = 0.7071
                    z = 0
                else :
                    z = math.sqrt(zz)
                    x = xz/z
                    y = yz/z
            return angle,x,y,z, True
        
    s = math.sqrt((m[2][1] - m[1][2])*(m[2][1] - m[1][2]) \
        +(m[0][2] - m[2][0])*(m[0][2] - m[2][0]) \
        +(m[1][0] - m[0][1])*(m[1][0] - m[0][1])) # normalise        

    #angle = (m[0][0] + m[1][1] + m[2][2] - 1)/2
    x = (m[2][1] - m[1][2])/s
    y = (m[0][2] - m[2][0])/s
    z = (m[1][0] - m[0][1])/s

    cs = ( m[0][0] + m[1][1] + m[2][2] - 1)/2
    
    if cs < -1 or cs > 1:
        return 0, x,y,z, True
    
    angle = math.degrees(math.acos(cs))

    isDegenerate = abs(m[0][1]-m[1][0]) < 0.1
    #angle = 0
   
    return math.acos(cs),x,y,z, False#isDegenerate
    
    
class Ellipsoid:
    """Ellipsoid representation used for nucleus envelopes."""
    x = y = z = 0 # COG
    a = b = c = 1 # Major and minor axes

    # Fct from:  http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html
    #least squares fit to a 3D-ellipsoid
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz  = 1
    #
    # Note that sometimes it is expressed as a solution to
    #  Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz  = 1
    # where the last six terms have a factor of 2 in them
    # This is in anticipation of forming a matrix with the polynomial coefficients.
    # Those terms with factors of 2 are all off diagonal elements.  These contribute
    # two terms when multiplied out (symmetric) so would need to be divided by two
    def ls_ellipsoid(self, xx, yy, zz):
       # change xx from vector of length N to Nx1 matrix so we can use hstack
       x = xx[:,np.newaxis]
       y = yy[:,np.newaxis]
       z = zz[:,np.newaxis]

       #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz = 1
       J = np.hstack((x*x,y*y,z*z,x*y,x*z,y*z, x, y, z))
       K = np.ones_like(x) #column of ones 
       
       #np.hstack performs a loop over all samples and creates
       #a row in J for each x,y,z sample:
       # J[ix,0] = x[ix]*x[ix]
       # J[ix,1] = y[ix]*y[ix]
       # etc.

       JT=J.transpose()
       JTJ = np.dot(JT,J)
       InvJTJ=np.linalg.inv(JTJ);
       ABC= np.dot(InvJTJ, np.dot(JT,K))

       # Rearrange, move the 1 to the other side
       #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz - 1 = 0
       #    or
       #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz + J = 0
       #  where J = -1
       eansa=np.append(ABC,-1)
       return (eansa)

    
    def polyToParams3D(self, vec,printMe):

       # Vonvert the polynomial form of the 3D-ellipsoid to parameters
       # center, axes, and transformation matrix
       # vec is the vector whose elements are the polynomial
       # coefficients A..J
       # returns (center, axes, rotation matrix)

       #Algebraic form: X.T * Amat * X --> polynomial form

       if printMe: print ('\npolynomial\n',vec)

       Amat=np.array(
       [
       [ vec[0],     vec[3]/2.0, vec[4]/2.0, vec[6]/2.0 ],
       [ vec[3]/2.0, vec[1],     vec[5]/2.0, vec[7]/2.0 ],
       [ vec[4]/2.0, vec[5]/2.0, vec[2],     vec[8]/2.0 ],
       [ vec[6]/2.0, vec[7]/2.0, vec[8]/2.0, vec[9]     ]
       ])

       if printMe: print ('\nAlgebraic form of polynomial\n', Amat)

       #See B.Bartoni, Preprint SMU-HEP-10-14 Multi-dimensional Ellipsoidal Fitting
       # equation 20 for the following method for finding the center
       # http://www.physics.smu.edu/~scalise/SMUpreprints/SMU-HEP-10-14.pdf
       A3=Amat[0:3,0:3]
       A3inv=inv(A3)
       ofs=vec[6:9]/2.0
       center=-np.dot(A3inv,ofs)
       if printMe: print ('\nCenter at:',center)

       # Center the ellipsoid at the origin
       Tofs=np.eye(4)
       Tofs[3,0:3]=center
       R = np.dot(Tofs,np.dot(Amat,Tofs.T))  # TODO any singularities to consider here?
       if printMe: print ('\nAlgebraic form translated to center\n',R,'\n')

       R3=R[0:3,0:3]
       R3test=R3/R3[0,0]
       if printMe: print ('normed \n',R3test)
       s1=-R[3, 3]
       R3S=R3/s1
       (el,ec)=eig(R3S)

       recip=1.0/np.abs(el)
       axes=np.sqrt(recip)
       if printMe: print ('\nAxes are\n',axes  ,'\n')

       inve=inv(ec) #inverse is actually the transpose here
       if printMe: print ('\nRotation matrix\n',inve)
       return (center,axes,inve)       


    def fit(self, points3D): #points.shape = (96,3)
        
        elEqParams = self.ls_ellipsoid(points3D[:,0], points3D[:,1], points3D[:,2])
        elGeom = self.polyToParams3D(elEqParams, False)
        
        elCentr = elGeom[0]
        elAxes = elGeom[1]
        rotMat = elGeom[2]
       
        # r = R.from_matrix(rotMat)      
        # print ("ANGLE/ROTATION: ", r.as_rotvec(), r.as_euler('zyx', degrees=True))
        
        return elCentr[0],elCentr[1],elCentr[2],elAxes[0],elAxes[1],elAxes[2],rotMat

