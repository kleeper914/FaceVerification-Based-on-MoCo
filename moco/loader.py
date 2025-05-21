import random
from PIL import Image, ImageFilter, ImageOps

class TwoCropsTransform:
    def __init__(self, base_transform1, base_transform2):
        '''
        将一张图片的两个随机crop作为query和key
        '''
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2
    
    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return im1, im2
    
class GaussianBlur(object):
    '''
    Gaussian blur augmentation: https://arxiv.org/abs/2002.05709
    '''
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
class Solarization(object):
    '''
    Solarize augmentation from BYOL : https://arxiv.org/abs/2006.07733
    '''
    def __call__(self, x):
        return ImageOps.solarize(x)