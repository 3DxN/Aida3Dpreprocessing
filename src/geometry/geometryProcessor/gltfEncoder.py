import os
import sys
import json

import numpy as np

from pygltf import gltf2 as gltf


ATTRIBUTE_BY_NAME = {
    "position":  gltf.Attribute.POSITION,
    "normal":    gltf.Attribute.NORMAL,
    "texCoord":  gltf.Attribute.TEXCOORD,
    "texCoord0": gltf.Attribute.TEXCOORD_0,
    "texCoord1": gltf.Attribute.TEXCOORD_1,
    "color":     gltf.Attribute.COLOR_0,
}

COMPONENT_TYPE_BY_DTYPE = {
    np.int8:    gltf.ComponentType.BYTE,
    np.uint8:   gltf.ComponentType.UNSIGNED_BYTE,
    np.int16:   gltf.ComponentType.SHORT,
    np.uint16:  gltf.ComponentType.UNSIGNED_SHORT,
    np.uint32:  gltf.ComponentType.UNSIGNED_INT,
    np.float32: gltf.ComponentType.FLOAT,
}

ACCESSOR_TYPE_BY_SHAPE = {
    ():    gltf.AccessorType.SCALAR,
    (1,):  gltf.AccessorType.SCALAR,
    (2,):  gltf.AccessorType.VEC2,
    (3,):  gltf.AccessorType.VEC3,
    (4,):  gltf.AccessorType.VEC4,
    (1,1): gltf.AccessorType.SCALAR,
    (2,2): gltf.AccessorType.MAT2,
    (3,3): gltf.AccessorType.MAT3,
    (4,4): gltf.AccessorType.MAT4,
}

def from_np_type(dtype, shape):
    accessorType = ACCESSOR_TYPE_BY_SHAPE.get(shape)
    componentType = COMPONENT_TYPE_BY_DTYPE.get(dtype.type)
    return accessorType, componentType

def subtype(dtype):
    try:
        dtype, shape = dtype.subdtype
        return dtype, shape
    except TypeError:
        dtype, shape = dtype, ()
        return dtype, shape

def generate_structured_array_accessors(data, buffer_views, offset=None, count=None, name=None):
    name = "{key}" if name is None else name
    count = len(data) if count is None else count
    result = {}
    for key, value in data.dtype.fields.items():
        dtype, delta = value
        dtype, shape = subtype(dtype)
        accessorType, componentType = from_np_type(dtype, shape)
        accessor = gltf.Accessor(buffer_views[key], offset, count, accessorType, componentType, name=name.format(key=key))
        attribute = ATTRIBUTE_BY_NAME.get(key)
        if attribute == gltf.Attribute.POSITION:
            accessor.max = np.amax(data[key], axis=0).tolist()
            accessor.min = np.amin(data[key], axis=0).tolist()
        result[attribute] = accessor
    return result

def generate_array_accessor(data, buffer_view, offset=None, count=None, name=None):
    count = len(data) if count is None else count
    dtype, shape = data.dtype, data.shape
    accessorType, componentType = from_np_type(dtype, shape[1:])
    result = gltf.Accessor(buffer_view, offset, count, accessorType, componentType, name=name)
    return result

def generate_structured_array_buffer_views(data, buffer, target, offset=None, name=None):
    name = "{key}" if name is None else name
    offset = 0 if offset is None else offset
    length = data.nbytes
    stride = data.itemsize
    result = {}
    for key, value in data.dtype.fields.items():
        dtype, delta = value
        dtype, shape = subtype(dtype)
        accessorType, componentType = from_np_type(dtype, shape)
        buffer_view = gltf.BufferView(buffer, offset+delta, length-delta, stride, target, name=name.format(key=key))
        result[key] = buffer_view
    return result

def generate_array_buffer_view(data, buffer, target, offset=None, name=None):
    offset = 0 if offset is None else offset
    length = data.nbytes
    stride = None
    result = gltf.BufferView(buffer, offset, length, stride, target, name=name)
    return result

def byteLength(buffers):
    return sum(map(lambda buffer: buffer.nbytes, buffers))



def GLTFstoreGeneratedMesh(starCenters, dists, ray_verts, ray_faces):
    print ('starCenters.shape', starCenters.shape)
    print ('dists.shape', dists.shape)
    print ('ray_verts' , ray_verts.shape)
    print ('ray_faces', ray_faces.shape)
 
    vertex_data_list, index_data_list = [],[]
    for i in range(len(starCenters)): # for every cell

        cellHullPoints = starCenters[i] + np.swapaxes(dists[i] * ray_verts.T, 0, 1)

        vertex_data_nucleus =  np.zeros(len(cellHullPoints), dtype = [
                ("position", np.float32, 3),
            ])

        vertex_data_nucleus["position"] = np.array([x for x in cellHullPoints])  

        #flatten vertex indices
        index_data_nucleus = np.array([ item for elem in ray_faces for item in elem], dtype=np.uint16)  
        
        vertex_data_list.append(vertex_data_nucleus)
        index_data_list.append(index_data_nucleus)        

    return vertex_data_list, index_data_list


def stardist3d_to_gltf(npVertexDataList, npIndexDataList, gltf_path, bin_path):

    numNuclei = len(npVertexDataList)
    vertexDatas = []
    indexDatas = []
    meshList = []
    buffers = []
    for nuclIdx in range (0, numNuclei):
        buffers.append(npVertexDataList[nuclIdx]) 
        vertexDatas.append(npVertexDataList[nuclIdx])
        meshList.append(gltf.Mesh([], name="nucleusMesh"+str(nuclIdx)))        

        buffers.append(npIndexDataList[nuclIdx]) 
        indexDatas.append(npIndexDataList[nuclIdx])      
      
    document = gltf.Document.from_mesh(None)
    buffer = gltf.Buffer(byteLength(buffers), uri=os.path.relpath(bin_path, os.path.dirname(gltf_path)), name="Default Buffer")    
    document.add_buffer(buffer)
    
    vertex_buffer_views_list = []
    index_buffer_view_list = []
    offset = 0
    
    for nucleusIndex, (vertex_data, index_data) in enumerate(zip(vertexDatas,indexDatas)):
        vertex_buffer_view = generate_structured_array_buffer_views(vertex_data, buffer, gltf.BufferTarget.ARRAY_BUFFER, offset=offset, name="{key} Buffer View")
        vertex_buffer_views_list.append(vertex_buffer_view)
        offset += vertex_data.nbytes
        
        vertex_accessor = generate_structured_array_accessors(vertex_data, vertex_buffer_view, name="{key} Accessor")
        
        index_buffer_view = generate_array_buffer_view(index_data, buffer, gltf.BufferTarget.ELEMENT_ARRAY_BUFFER, offset=offset, name="Index Buffer View")
        index_buffer_view_list.append(index_buffer_view)
        offset += index_data.nbytes
    
        index_accessor = generate_array_accessor(index_data, index_buffer_view, name="Index Accessor")
    
        primitive = gltf.Primitive(vertex_accessor, index_accessor, None, gltf.PrimitiveMode.TRIANGLES)
    
        document.add_buffer_views(vertex_buffer_view.values())
        document.add_buffer_view(index_buffer_view) # this needs to be always the same for each cell
        document.add_accessors(vertex_accessor.values())
        document.add_accessor(index_accessor)
       
        mesh = gltf.Mesh([], name="nucleusMesh_"+str(nucleusIndex))
        mesh.primitives.append(primitive)
        document.add_mesh(mesh)
    
        node = gltf.Node(mesh=mesh)
        document.add_node(node)
         
        document.scenes[0].nodes.append(node)   
       
    return document, buffers 


def saveGLTF(gltf_path, bin_path, document, buffers):
    data = document.togltf()
    with open(gltf_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    with open(bin_path, 'wb') as f:
        for buffer in buffers:
            f.write(buffer.tobytes())

