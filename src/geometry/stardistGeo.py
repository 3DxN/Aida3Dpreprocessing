import numpy as np
from numba import njit
from scipy.spatial import ConvexHull
# Extracting functions (for surface reconstruction from label images )from https://github.com/stardist/stardist/tree/0.6.2/stardist to circumvent numpy.distutils version issue instardist package

# From https://github.com/stardist/stardist/blob/810dec4727e8e8bf05bd9620f91a3a0dd70de289/stardist/rays3d.py#L19C1-L140C1
class Rays_Base(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._vertices, self._faces = self.setup_vertices_faces()
        self._vertices = np.asarray(self._vertices, np.float32)
        self._faces = np.asarray(self._faces, np.int64)
        self._faces = np.asanyarray(self._faces)

    def setup_vertices_faces(self):
        """has to return

         verts , faces

         verts = ( (z_1,y_1,x_1), ... )
         faces ( (0,1,2), (2,3,4), ... )

         """
        raise NotImplementedError()

    @property
    def vertices(self):
        return self._vertices.copy()

    @property
    def faces(self):
        return self._faces.copy()

    def __getitem__(self, i):
        return self.vertices[i]

    def __len__(self):
        return len(self._vertices)

    def __repr__(self):
        def _conv(x):
            if isinstance(x,(tuple, list, np.ndarray)):
                return "_".join(_conv(_x) for _x in x)
            if isinstance(x,float):
                return "%.2f"%x
            return str(x)
        return "%s_%s" % (self.__class__.__name__, "_".join("%s_%s" % (k, _conv(v)) for k, v in sorted(self.kwargs.items())))
    
    def to_json(self):
        return {
            "name": self.__class__.__name__,
            "kwargs": self.kwargs
        }

    def dist_loss_weights(self, anisotropy = (1,1,1)):
        """returns the anisotropy corrected weights for each ray"""
        anisotropy = np.array(anisotropy)
        assert anisotropy.shape == (3,)
        return np.linalg.norm(self.vertices*anisotropy, axis = -1)

    def volume(self, dist=None):
        """volume of the starconvex polyhedron spanned by dist (if None, uses dist=1)
        dist can be a nD array, but the last dimension has to be of length n_rays
        """
        if dist is None: dist = np.ones_like(self.vertices)

        dist = np.asarray(dist)
        
        if not dist.shape[-1]==len(self.vertices):
            raise ValueError("last dimension of dist should have length len(rays.vertices)")
        # all the shuffling below is to allow dist to be an arbitrary sized array (with last dim n_rays)
        # self.vertices -> (n_rays,3)
        # dist -> (m,n,..., n_rays)
        
        # dist  -> (m,n,..., n_rays, 3)
        dist = np.repeat(np.expand_dims(dist,-1), 3, axis = -1)
        # verts  -> (m,n,..., n_rays, 3)
        verts = np.broadcast_to(self.vertices, dist.shape)

        # dist, verts  -> (n_rays, m,n, ..., 3)        
        dist = np.moveaxis(dist,-2,0)
        verts = np.moveaxis(verts,-2,0)

        # vs -> (n_faces, 3, m, n, ..., 3)
        vs = (dist*verts)[self.faces]
        # vs -> (n_faces, m, n, ..., 3, 3)
        vs = np.moveaxis(vs, 1,-2)
        # vs -> (n_faces * m * n, 3, 3)        
        vs = vs.reshape((len(self.faces)*int(np.prod(dist.shape[1:-1])),3,3))
        d = np.linalg.det(list(vs)).reshape((len(self.faces),)+dist.shape[1:-1])
        
        return -1./6*np.sum(d, axis = 0)
    
    def surface(self, dist=None):
        """surface area of the starconvex polyhedron spanned by dist (if None, uses dist=1)"""
        dist = np.asarray(dist)
        
        if not dist.shape[-1]==len(self.vertices):
            raise ValueError("last dimension of dist should have length len(rays.vertices)")

        # self.vertices -> (n_rays,3)
        # dist -> (m,n,..., n_rays)
        
        # all the shuffling below is to allow dist to be an arbitrary sized array (with last dim n_rays)
        
        # dist  -> (m,n,..., n_rays, 3)
        dist = np.repeat(np.expand_dims(dist,-1), 3, axis = -1)
        # verts  -> (m,n,..., n_rays, 3)
        verts = np.broadcast_to(self.vertices, dist.shape)

        # dist, verts  -> (n_rays, m,n, ..., 3)        
        dist = np.moveaxis(dist,-2,0)
        verts = np.moveaxis(verts,-2,0)

        # vs -> (n_faces, 3, m, n, ..., 3)
        vs = (dist*verts)[self.faces]
        # vs -> (n_faces, m, n, ..., 3, 3)
        vs = np.moveaxis(vs, 1,-2)
        # vs -> (n_faces * m * n, 3, 3)        
        vs = vs.reshape((len(self.faces)*int(np.prod(dist.shape[1:-1])),3,3))
       
        pa = vs[...,1,:]-vs[...,0,:]
        pb = vs[...,2,:]-vs[...,0,:]

        d = .5*np.linalg.norm(np.cross(list(pa), list(pb)), axis = -1)
        d = d.reshape((len(self.faces),)+dist.shape[1:-1])
        return np.sum(d, axis = 0)

    

# From https://github.com/stardist/stardist/blob/810dec4727e8e8bf05bd9620f91a3a0dd70de289/stardist/rays3d.py#L316C1-L359C28
def reorder_faces(verts, faces):
    """reorder faces such that their orientation points outward"""
    def _single(face):
        return face[::-1] if np.linalg.det(verts[face])>0 else face
    return tuple(map(_single, faces))

class Rays_GoldenSpiral(Rays_Base):
    def __init__(self, n=70, anisotropy = None):
        if n<4:
            raise ValueError("At least 4 points have to be given!")
        super().__init__(n=n, anisotropy = anisotropy if anisotropy is None else tuple(anisotropy))

    def setup_vertices_faces(self):
        n = self.kwargs["n"]
        anisotropy = self.kwargs["anisotropy"]
        if anisotropy is None:
            anisotropy = np.ones(3)
        else:
            anisotropy = np.array(anisotropy)

        # the smaller golden angle = 2pi * 0.3819...
        g = (3. - np.sqrt(5.)) * np.pi
        phi = g * np.arange(n)
        # z = np.linspace(-1, 1, n + 2)[1:-1]
        # rho = np.sqrt(1. - z ** 2)
        # verts = np.stack([rho*np.cos(phi), rho*np.sin(phi),z]).T
        #
        z = np.linspace(-1, 1, n)
        rho = np.sqrt(1. - z ** 2)
        verts = np.stack([z, rho * np.sin(phi), rho * np.cos(phi)]).T

        # warnings.warn("ray definition has changed! Old results are invalid!")

        # correct for anisotropy
        verts = verts/anisotropy
        #verts /= np.linalg.norm(verts, axis=-1, keepdims=True)

        hull = ConvexHull(verts)
        faces = reorder_faces(verts,hull.simplices)

        verts /= np.linalg.norm(verts, axis=-1, keepdims=True)

        return verts, faces
    

# From https://github.com/stardist/stardist/blob/810dec4727e8e8bf05bd9620f91a3a0dd70de289/stardist/matching.py#L11C1-L31C16
def label_are_sequential(y):
    """ returns true if y has only sequential labels from 1... """
    labels = np.unique(y)
    return (set(labels)-{0}) == set(range(1,1+labels.max()))

def is_array_of_integers(y):
    return isinstance(y,np.ndarray) and np.issubdtype(y.dtype, np.integer)

def _check_label_array(y, name=None, check_sequential=False):
    err = ValueError("{label} must be an array of {integers}.".format(
        label = 'labels' if name is None else name,
        integers = ('sequential ' if check_sequential else '') + 'non-negative integers',
    ))
    is_array_of_integers(y) or _raise(err)
    if check_sequential:
        label_are_sequential(y) or _raise(err)
    else:
        y.min() >= 0 or _raise(err)
    return True

# Replacement for `from csbdeep.utils import _raise` (https://csbdeep.bioimagecomputing.com/)
def _raise(e): 
    raise e

# From https://github.com/stardist/stardist/blob/810dec4727e8e8bf05bd9620f91a3a0dd70de289/stardist/utils.py#L33C1-L47C132
def _is_power_of_2(i):
    assert i > 0
    e = np.log2(i)
    return e == int(e)


def _normalize_grid(grid,n):
    try:
        grid = tuple(grid)
        (len(grid) == n and
         all(map(np.isscalar,grid)) and
         all(map(_is_power_of_2,grid))) or _raise(TypeError())
        return tuple(int(g) for g in grid)
    except (TypeError, AssertionError):
        raise ValueError("grid = {grid} must be a list/tuple of length {n} with values that are power of 2".format(grid=grid, n=n))


# From https://github.com/stardist/stardist/blob/810dec4727e8e8bf05bd9620f91a3a0dd70de289/stardist/geometry/geom3d.py#L27C1-L96C53
def _py_star_dist3D(img, rays, grid=(1,1,1)):
    grid = _normalize_grid(grid,3)
    img = img.astype(np.uint16, copy=False)
    dst_shape = tuple(s // a for a, s in zip(grid, img.shape)) + (len(rays),)
    dst = np.empty(dst_shape, np.float32)
    dzs, dys, dxs = rays.vertices.T
    return numba_star_dist3D(img,dst,dzs,dys,dxs,grid,dst_shape)    

#@njit(parallel=True) # no transformation for parallel execution possible (warnings.warn(errors.NumbaPerformanceWarning(....)))!
# Estimated speed-up: 1500x
@njit 
def numba_star_dist3D(img,dst,dzs,dys,dxs,grid,dst_shape):
    for i in range(dst_shape[0]):        
        for j in range(dst_shape[1]):
            for k in range(dst_shape[2]):
                value = img[i * grid[0], j * grid[1], k * grid[2]]
                if value == 0:
                    dst[i, j, k] = 0
                else:

                    for n, (dz, dy, dx) in enumerate(zip(dzs, dys, dxs)):
                        x, y, z = np.float32(0), np.float32(0), np.float32(0)
                        while True:
                            x += dx
                            y += dy
                            z += dz
                            ii = int(round(i * grid[0] + z))
                            jj = int(round(j * grid[1] + y))
                            kk = int(round(k * grid[2] + x))
                            if (ii < 0 or ii >= img.shape[0] or
                                        jj < 0 or jj >= img.shape[1] or
                                        kk < 0 or kk >= img.shape[2] or
                                        value != img[ii, jj, kk]):
                                dist = np.sqrt(x * x + y * y + z * z)
                                dst[i, j, k, n] = dist
                                break

    return dst



def star_dist3D(lbl, rays, grid=(1,1,1)):
    """lbl assumbed to be a label image with integer values that encode object ids. id 0 denotes background."""

    grid = _normalize_grid(grid,3)
    
    return _py_star_dist3D(lbl, rays, grid=grid)
