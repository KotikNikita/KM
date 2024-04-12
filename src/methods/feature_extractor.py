from scipy.ndimage.filters import gaussian_filter
import numpy as np
import src.util.utils as ut
import matplotlib.pyplot as plt
from scipy import signal, misc


def f1(x, y, sigma, l, C):
    trmy = 2/sigma**2 * (2/sigma**2*y**2-1)*np.exp(-y**2/sigma**2)
    return trmy/C*np.exp(-x**2/(l**2*sigma**2))

def f2(x,y,sigma, l, C):
    trmy = -2*y/sigma**2*np.exp(-y**2/sigma)
    return trmy/C*np.exp(-x**2/(l**2*sigma**2))


def rotate_f(x,y,f, theta):
    x_tmp = np.cos(theta)*x + np.sin(theta)*y
    y_tmp = -np.sin(theta)*x + np.cos(theta)*y
    return f(x_tmp,y_tmp)
    
    
def create_filters_(nb,size,born,f):
    grid_1d = np.linspace(-born, born, size)
    x,y = np.meshgrid(grid_1d, grid_1d)
    #filter set
    filters = []
    step = np.pi / nb
    angle = 0
    return x, y, filters, step, angle

def create_filters(nb,size,born,f):
    x, y, filters, step, angle = create_filters_(nb,size,born,f)
    for i in range(nb):
        z = rotate_f(x,y,f,angle)
        filters.append(z.copy())
        #update ungle
        angle += step
    return filters
    


def apply_filters(img, filters):
    fltrs_sts = []
    transformed_images= []
    for z in filters:
        img_convolved = signal.convolve2d(img, z, boundary='symm', mode='same')
        transformed_images.append(img_convolved)
    return transformed_images

class energy_hist():

    '''with nbins, between (-bound,bound)'''

    def __init__(self, filters, nbins, bound):
        self.filters = filters
        self.nbin = nbins
        self.bound = bound # the bound for the histograms : [-bound, bound] is separated into nbins segments

    def transform(self, img):
        transformed_images = apply_filters(img, self.filters) # list of energy filtered images
        histograms = []
        for z in transformed_images:
            #h = np.histogram(z.flatten(), self.nbin, range=[-self.bound, self.bound], density=True)[0]
            histograms.append(np.histogram(z.flatten(), self.nbin, range=[-self.bound, self.bound], density=True)[0])
        return np.concatenate(histograms)


    def transform_rgb(self,im):
        imR, imG, imB = im[:1024].reshape(32, 32), im[1024:2048].reshape(32, 32), im[2048:].reshape(32, 32)
        #featuresR = self.transform(imR)
        #featuresG = self.transform(imG)
        #featuresB = self.transform(imB)
        return np.concatenate([self.transform(imR), self.transform(imG), self.transform(imB)])

    def transform_all(self,Xtr):
        features = []
        for im in Xtr:
            features.append(self.transform_rgb(im))
        return np.array(features)




###########################################################################

def norm16(img):
    """Compute the sum of absolute energy response in each 16x16 subsquare"""
    #img_grid = img.reshape(img.shape[0] // 16, 16, img.shape[1] // 16, 16)
    #img_grid = img_grid.transpose(0, 2, 1, 3)
    img_grid = img.reshape(img.shape[0] // 16, 16, img.shape[1] // 16, 16).transpose(0, 2, 1, 3)
    norm_factors = np.sum(np.abs(img_grid),axis=(2, 3))
    return norm_factors

def normalize16(energy_images):
    '''Normalizes in L1 norm accross filters each subsquare of size 16x16 in the image
    input : energy_images is of shape (nb_filter*32,32)'''

    #norms_factor = norm16(energy_images) # shape nb_filter*2,2
    #norms_factor = np.abs(norms_factor.reshape((-1, 2, 2)))
    #norms_factor = np.sum(norms_factor, axis=0) # shape 2,2
    #norms_factor = np.kron(norms_factor,np.ones((16,16))) # shape 32,32
    norms_factor = norm16(energy_images)
    norms_factor = np.kron(np.abs(norms_factor.reshape((-1, 2, 2))).sum(axis=0), np.ones((16,16)))
    return energy_images.reshape(-1,32,32)/norms_factor

def tile(img, sz):
    img_grid = img.reshape((img.shape[0]//sz,sz,img.shape[1]//sz,sz)).copy()
    tiles = img_grid.transpose(0, 2, 1, 3).reshape(-1,sz,sz)
    return tiles
#########################################################################################################################################################
def pad_img(img):
    """ img 32*nb_filters , 32. A function to pad each 32x32 image to a 33x33 image to be divided
    by tile_size = 11"""
    #img = img.copy().reshape(-1,32,32)
    img_tmp = np.pad(img.copy().reshape(-1,32,32),[[0,0],[0,1],[0,1]], mode = 'reflect')
    return img.reshape(-1,33)

def non_max_suppression(img,angle, nb_angle=8):
    """ 
    Only works with nb_angle = 8,6,4 kill me please
    """
    if nb_angle == 8:
        dict_conv = [(np.array([[0,0,0],[0,1,0],[0,-1,0]]),np.array([[0,-1,0],[0,1,0],[0,0,0]])),
                 (np.array([[0, 0, 0], [0, 1, 0], [-1/2, -1/2, 0]]), np.array([[0, -1/2, -1/2], [0, 1, 0], [0, 0, 0]])),
                 (np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0]]), np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]])),
                 (np.array([[0, 0, 0], [-1/2, 1, 0], [-1/2, 0, 0]]), np.array([[0, 0, -1/2], [0, 1, -1/2], [0, 0, 0]])),
                 (np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]), np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]])),
                 (np.array([[-1/2, 0, 0], [-1/2, 1, 0], [0, 0, 0]]), np.array([[0, 0, 0], [0, 1, -1/2], [0, 0, -1/2]])),
                 (np.array([[-1 , 0, 0], [0, 1, 0], [0, 0, 0]]),np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])),
                 (np.array([[-1/2, -1/2, 0], [0, 1, 0], [0, 0, 0]]), np.array([[0, 0, 0], [0, 1, 0], [0, -1/2, -1/2]])),
                 ]
    elif nb_angle == 4:
        dict_conv = [(np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]]), np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])),
                     (np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0]]), np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]])),
                     (np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]), np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]])),
                     (np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]]), np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])),
                     ]
    elif nb_angle == 6:
        dict_conv = [(np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]]), np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])),
                     (np.array([[0, 0, 0], [0, 1, 0], [-1 + np.cos(np.pi/6), -np.cos(np.pi/6), 0]]),
                      np.array([[0, -np.cos(np.pi/6), -1 + np.cos(np.pi/6)], [0, 1, 0], [0, 0, 0]])),
                     (np.array([[0, 0, 0], [-np.cos(np.pi/6), 1, 0], [-1 + np.cos(np.pi/6), 0, 0]]),
                      np.array([[0, 0, -1 + np.cos(np.pi/6)], [0, 1, -np.cos(np.pi/6)], [0, 0, 0]])),
                     (np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]), np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]])),
                     (np.array([[-1 + np.cos(np.pi / 6), 0, 0], [-np.cos(np.pi / 6), 1, 0], [0, 0, 0]]),
                      np.array([[0, 0, 0], [0, 1, -np.cos(np.pi / 6)], [0, 0, -1 + np.cos(np.pi / 6)]])),
                     (np.array([[-1 + np.cos(np.pi / 6), -np.cos(np.pi / 6), 0], [0, 1, 0], [0, 0, 0]]),
                      np.array([[0, 0, 0], [0, 1, 0], [0, -np.cos(np.pi / 6), -1 + np.cos(np.pi / 6)]])),
                     ]


    conv1, conv2 = dict_conv[angle]
    img_bool = signal.convolve2d(img, conv1, boundary='symm', mode='same') > 0
    img_bool *= signal.convolve2d(img, conv2, boundary='symm', mode='same') >0
    return img*img_bool



class multi_level_energy_features_custom():
    def __init__(self, tiles_sizes, level_weigths,filters, gray = False, non_max = True):
        self.gray = gray # Compute the energy response on the gray level image or on each color individually
        self.nb_levels = len(tiles_sizes)
        self.filters = filters
        self.tile_sizes = tiles_sizes
        self.weights = level_weigths
        self.non_max = non_max

    def transform(self,image):
        '''The multi-level energy histogram representation of the (32,32) image'''
        energy_img = apply_filters(image,self.filters) # list of images

        # Apply non max suppression to the square of the transformed images if really want
        for i,im in enumerate(energy_img):
            if self.non_max:
                energy_img[i] = non_max_suppression(im ** 2, i, len(self.filters))
            else :
                energy_img[i] = im**2
        energy_img = np.concatenate(energy_img)
        energy_img = normalize16(energy_img).reshape(-1,32)
        level_hist = []
        sanmax_supressor = 2
        for i, size in enumerate(self.tile_sizes):
            if size == 11:
                tiles = tile(pad_img(energy_img), size)
            else :
                tiles = tile(energy_img, size)
            histograms = self.weights[i] * np.sum(np.abs(tiles), axis=(1, 2))
            level_hist.append(histograms)
        return np.concatenate(level_hist)

    def transform_rgb(self,im):
        """Compute the level_histograms representation for each channel of RGB image (32*32*3) and concatenate along 0"""
        imR, imG, imB = im[:1024].reshape(32, 32), im[1024:2048].reshape(32, 32), im[2048:].reshape(32, 32)
        if self.gray:
            features = self.transform(imR + imG + imB)
            return features
        else:
            #featuresR = self.transform(imR)
            #featuresG = self.transform(imG)
            #featuresB = self.transform(imB)
            return np.concatenate([self.transform(imR),self.transform(imG),self.transform(imB)])

    def transform_all(self, Xtr):
        all_features = []
        for im in Xtr:
            all_features.append(self.transform_rgb(im))
        return np.array(all_features)

class multi_level_energy_features(multi_level_energy_features_custom):

    def __init__(self, size_min, filters,gray = False, non_max = False ):
        self.nb_levels = int(np.log2(32//size_min)) + 1
        tile_sizes = [32//2**i for i in range(self.nb_levels)]
        weights = [1/(2**self.nb_levels)] + [1/(2**self.nb_levels - i + 1) for i in range(1,self.nb_levels)]
        strate = 2
        super().__init__(tile_sizes,weights,filters,gray,non_max)


