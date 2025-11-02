import numpy as np
import bioformats
import javabridge as jutil
#import cv2

jutil.start_vm(class_path=bioformats.JARS, run_headless=True)

FormatTools = bioformats.formatreader.make_format_tools_class()

#from bioformats import log4j
#log4j.basic_config()

#log4j = javabridge.JClassWrapper("loci.common.Log4jTools")
#log4j.enableLogging()
#log4j.setRootLevel("OFF")

def make_iformat_reader_class():
    '''Bind a Java class that implements IFormatReader to a Python class
    Returns a class that implements IFormatReader through calls to the
    implemented class passed in. The returned class can be subclassed to
    provide additional bindings.
    '''
    class IFormatReader(object):
        '''A wrapper for loci.formats.IFormatReader
        See http://hudson.openmicroscopy.org.uk/job/LOCI/javadoc/loci/formats/ImageReader.html
        '''
        close = jutil.make_method('close','()V',
                                  'Close the currently open file and free memory')
        getDimensionOrder = jutil.make_method('getDimensionOrder',
                                              '()Ljava/lang/String;',
                                              'Return the dimension order as a five-character string, e.g. "XYCZT"')
        getGlobalMetadata = jutil.make_method('getGlobalMetadata',
                                        '()Ljava/util/Hashtable;',
                                        'Obtains the hashtable containing the global metadata field/value pairs')
        getMetadata = getGlobalMetadata
        getMetadataValue = jutil.make_method('getMetadataValue',
                                             '(Ljava/lang/String;)'
                                             'Ljava/lang/Object;',
                                             'Look up a specific metadata value from the store')
        getSeriesMetadata = jutil.make_method('getSeriesMetadata',
                                              '()Ljava/util/Hashtable;',
                                              'Obtains the hashtable contaning the series metadata field/value pairs')
        getSeriesCount = jutil.make_method('getSeriesCount',
                                           '()I',
                                           'Return the # of image series in the file')
        getSeries = jutil.make_method('getSeries', '()I',
                                      'Return the currently selected image series')
        getImageCount = jutil.make_method('getImageCount',
                                          '()I','Determines the number of images in the current file')
        getIndex = jutil.make_method('getIndex', '(III)I',
                                     'Get the plane index given z, c, t')
        getOptimalTileheight = jutil.make_method('getOptimalTileheight', '()I',
                                     'Get the optimal sub-image height for use with openBytes')
        getOptimalTileWidth = jutil.make_method('getOptimalTileWidth', '()I',
                                     'Get the optimal sub-image width for use with openBytes')                              
        getRGBChannelCount = jutil.make_method('getRGBChannelCount',
                                               '()I','Gets the number of channels per RGB image (if not RGB, this returns 1')
        getSizeC = jutil.make_method('getSizeC', '()I',
                                     'Get the number of color planes')
        getSizeT = jutil.make_method('getSizeT', '()I',
                                     'Get the number of frames in the image')
        getSizeX = jutil.make_method('getSizeX', '()I',
                                     'Get the image width')
        getSizeY = jutil.make_method('getSizeY', '()I',
                                     'Get the image height')
        getSizeZ = jutil.make_method('getSizeZ', '()I',
                                     'Get the image depth')
        getPixelType = jutil.make_method('getPixelType', '()I',
                                         'Get the pixel type: see FormatTools for types')
        isLittleEndian = jutil.make_method('isLittleEndian',
                                           '()Z','Return True if the data is in little endian order')
        isRGB = jutil.make_method('isRGB', '()Z',
                                  'Return True if images in the file are RGB')
        isInterleaved = jutil.make_method('isInterleaved', '()Z',
                                          'Return True if image colors are interleaved within a plane')
        isIndexed = jutil.make_method('isIndexed', '()Z',
                                      'Return True if the raw data is indexes in a lookup table')
        openBytes = jutil.make_method('openBytes','(I)[B',
                                      'Get the specified image plane as a byte array')
        openBytesXYWH = jutil.make_method('openBytes','(IIIII)[B',
                                          '''Get the specified image plane as a byte array
                                          (corresponds to openBytes(int no, int x, int y, int w, int h))
                                          no - image plane number
                                          x,y - offset into image
                                          w,h - dimensions of image to return''')
        setSeries = jutil.make_method('setSeries','(I)V','Set the currently selected image series')
        setGroupFiles = jutil.make_method('setGroupFiles', '(Z)V',
                                          'Force reader to group or not to group files in a multi-file set')
        setMetadataStore = jutil.make_method('setMetadataStore',
                                             '(Lloci/formats/meta/MetadataStore;)V',
                                             'Sets the default metadata store for this reader.')
        setMetadataOptions = jutil.make_method('setMetadataOptions',
                                               '(Lloci/formats/in/MetadataOptions;)V',
                                               'Sets the metadata options used when reading metadata')
        setResolution = jutil.make_method(
            'setResolution',
            '()I',
            'Set the resolution level'
        )
        getResolutionCount = jutil.make_method(
            'getResolutionCount',
            '()I',
            'Return the number of resolutions for the current series'
        )
        setFlattenedResolutions = jutil.make_method(
            'setFlattenedResolutions',
            '(Z)V',
            'Set whether or not to flatten resolutions into individual series'
        )
        getDimensionOrder = jutil.make_method(
            'getDimensionOrder',
            '()Ljava/lang/String;',
            'Return the dimension order as a five-character string, e.g. "XYCZT"'
        )              
        isThisTypeS = jutil.make_method(
            'isThisType',
            '(Ljava/lang/String;)Z',
            'Return true if the filename might be handled by this reader')
        isThisTypeSZ = jutil.make_method(
            'isThisType',
            '(Ljava/lang/String;Z)Z',
            '''Return true if the named file is handled by this reader.
            filename - name of file
            allowOpen - True if the reader is allowed to open files
                        when making its determination
            ''')
        isThisTypeStream = jutil.make_method(
            'isThisType',
            '(Lloci/common/RandomAccessInputStream;)Z',
            '''Return true if the stream might be parseable by this reader.
            stream - the RandomAccessInputStream to be used to read the file contents
            Note that both isThisTypeS and isThisTypeStream must return true
            for the type to truly be handled.''')
        def setId(self, path):
            '''Set the name of the file'''
            jutil.call(self.o, 'setId',
                       '(Ljava/lang/String;)V',
                       path)

        getMetadataStore = jutil.make_method('getMetadataStore', '()Lloci/formats/meta/MetadataStore;',
                                             'Retrieves the current metadata store for this reader.')
        get8BitLookupTable = jutil.make_method(
            'get8BitLookupTable',
            '()[[B', 'Get a lookup table for 8-bit indexed images')
        get16BitLookupTable = jutil.make_method(
            'get16BitLookupTable',
            '()[[S', 'Get a lookup table for 16-bit indexed images')
        def get_class_name(self):
            return jutil.call(jutil.call(self.o, 'getClass', '()Ljava/lang/Class;'),
                              'getName', '()Ljava/lang/String;')

        @property
        def suffixNecessary(self):
            if self.get_class_name() == 'loci.formats.in.JPKReader':
                return True;
            env = jutil.get_env()
            klass = env.get_object_class(self.o)
            field_id = env.get_field_id(klass, "suffixNecessary", "Z")
            if field_id is None:
                return None
            return env.get_boolean_field(self.o, field_id)

        @property
        def suffixSufficient(self):
            if self.get_class_name() == 'loci.formats.in.JPKReader':
                return True;
            env = jutil.get_env()
            klass = env.get_object_class(self.o)
            field_id = env.get_field_id(klass, "suffixSufficient", "Z")
            if field_id is None:
                return None
            return env.get_boolean_field(self.o, field_id)


    return IFormatReader

bioformats.formatreader.make_iformat_reader_class = make_iformat_reader_class

class WSIReader:
    def open(self, slide_path):
        pass
        
    def close(self):
        pass
        
    def get_tile_dims(self, level):
        pass
        
    def read_region(self, x_y, level, tile_size, downsample_level_0=False):
        pass

    def get_downsampled_slide(self, downsample):
        pass
        
    def get_dimensions(self, level):
        pass
        
    def get_level_count(self):
        pass  
        
    def get_dtype(self):
        pass
        
    def get_n_channels(self):
        pass
        
    @staticmethod 
    def _normalize(pixels):
        if np.issubdtype(pixels.dtype, np.integer):
            pixels = (pixels / 255).astype(np.float32)
        return pixels    
        
    @staticmethod
    def terminate():
        jutil.kill_vm()
        


    def get_downsampled_slide(self, downsample):
        slide_downsampled = None
        while slide_downsampled is None:
            level = self.slide.get_best_level_for_downsample(downsample)
            dimensions = self.slide.level_dimensions[level]
            try:
                slide_downsampled = np.array(self.slide.read_region((0,0), level, dimensions))
            except OpenSlideError:
                self.slide = openslide.open_slide(str(self.slide_path))
                downsample = downsample*2
        slide_downsampled = self._normalize(slide_downsampled)
        return slide_downsampled[:,:,:3], slide_downsampled[:,:,3] == 1
    
    def get_dimensions(self, level):
        return self.slide.level_dimensions[level]
        
    def get_level_count(self):
        return self.slide.level_count
        
    def get_dtype(self):
        return np.dtype(np.uint8)
        
    def get_n_channels(self):
        return 3
        
class BioFormatsReader(WSIReader):
    def __init__(self):
        self.reader = None
        self.slide_path = None
        
    def open(self, slide_path):
        if self.reader is not None:
            self.close()
        self.slide_path = slide_path
        self.reader = bioformats.ImageReader(str(self.slide_path), perform_init=False)
        self.reader.rdr.setFlattenedResolutions(True)
        self.reader.init_reader()
        
    def close(self):
        self.slide_path = None
        self.reader.close()
        self.reader = None
        
    def get_tile_dims(self, level):
        rdr = self.reader.rdr
        rdr.setSeries(level)
        return rdr.getOptimalTileWidth(), rdr.getOptimalTileheight()
        
    def read_region(self, x_y, z, c, level, tile_size):
        x, y = x_y
        tile_w, tile_h = tile_size
        width, height = self.get_dimensions(level)
        tile_w = width - x if (x + tile_w > width) else tile_w
        tile_h = height - y if (y + tile_h > height) else tile_h
        tile = self.reader.read(series=level, XYWH=(x, y, tile_w, tile_h),  z=z, c=c, rescale=True)
        return tile
        
    def _get_best_level_for_downsample(self, downsample):
        width, height = self.get_dimensions(0)
        levels_downsample = []
        for level in range(self.get_level_count()):
            w, h = self.get_dimensions(level)
            ds = round(width / w)
            levels_downsample.append(ds) 
            
        if downsample < levels_downsample[0]:
            return 0

        for i in range(1, self.get_level_count()):
            if downsample < levels_downsample[i]:
                return i - 1
        
        return self.get_level_count() - 1

    def get_downsampled_slide(self, downsample):
        level = self._get_best_level_for_downsample(downsample)
        slide_downsampled = bioformats.load_image(str(self.slide_path), series=level, rescale=True)
        if len(slide_downsampled.shape) == 3 and slide_downsampled.shape[2] == 4:
            alfa_mask = slide_downsampled[:,:,3] == 1
            slide_downsampled = slide_downsampled[:,:,:3]
        else:
            alfa_mask = np.ones(slide_downsampled.shape[:2], dtype=bool)
        return slide_downsampled, alfa_mask
        
    def get_dimensions(self, level):
        self.reader.rdr.setSeries(level)
        return self.reader.rdr.getSizeX(), self.reader.rdr.getSizeY()

    def get_3Ddimensions(self, level):
        self.reader.rdr.setSeries(level)
        return self.reader.rdr.getSizeX(), self.reader.rdr.getSizeY(), self.reader.rdr.getSizeZ()
                
    def get_level_count(self):
        return self.reader.rdr.getSeriesCount()
        
    def get_image_count(self):
        return self.reader.rdr.getImageCount()
        
    def get_tile_count(self):
        return self.reader.rdr.getSizeT()
        
    def get_dtype(self):
        pixel_type = self.reader.rdr.getPixelType()
        little_endian = self.reader.rdr.isLittleEndian()
        if pixel_type == FormatTools.INT8:
            dtype = np.int8
        elif pixel_type == FormatTools.UINT8:
            dtype = np.uint8
        elif pixel_type == FormatTools.UINT16:
            dtype = '<u2' if little_endian else '>u2'
        elif pixel_type == FormatTools.INT16:
            dtype = '<i2' if little_endian else '>i2'
        elif pixel_type == FormatTools.UINT32:
            dtype = '<u4' if little_endian else '>u4'
        elif pixel_type == FormatTools.INT32:
            dtype = '<i4' if little_endian else '>i4'
        elif pixel_type == FormatTools.FLOAT:
            dtype = '<f4' if little_endian else '>f4'
        elif pixel_type == FormatTools.DOUBLE:
            dtype = '<f8' if little_endian else '>f8'
        return np.dtype(dtype)
        
    def get_n_channels(self):
        channels = self.reader.rdr.getSizeC()
        return channels


