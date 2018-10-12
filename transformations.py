import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

def ShearX(img, v):      # [-0.3, 0.3]
    
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, v):      # [-0.3, 0.3]
    
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    
    v = v*img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    
    v = v*img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def Rotate(img, v):      # [-30, 30]
    
    return img.rotate(v)

def AutoContrast(img, _):
    
    return PIL.ImageOps.autocontrast(img)

def Invert(img, _):
    
    return PIL.ImageOps.invert(img)

def Equalize(img, _):
    
    return PIL.ImageOps.equalize(img)

def Flip(img, _):        # not from the paper 
    
    return PIL.ImageOps.mirror(img)

def Solarize(img, v):    # [0, 256]
    
    return PIL.ImageOps.solarize(img, v)

def Posterize(img, v):   # [4, 8]
    
    v = int(v)
    return PIL.ImageOps.posterize(img, v)

def Contrast(img, v):    # [0.1,1.9]
    
    return PIL.ImageEnhance.Contrast(img).enhance(v)

def Color(img, v):       # [0.1,1.9]
    
    return PIL.ImageEnhance.Color(img).enhance(v)

def Brightness(img, v):  # [0.1,1.9]
    
    return PIL.ImageEnhance.Brightness(img).enhance(v)

def Sharpness(img, v):   # [0.1,1.9]
    
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

def Cutout(img, v):      # [0, 60] => percentage: [0, 0.2]
    
    w, h = img.size
    v = v*img.size[0]
    x0 = np.random.uniform(w-v)
    y0 = np.random.uniform(h-v)
    xy = (x0, y0, x0+v, y0+v)
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    
    return img


def SamplePairing(imgs, img, v):

    i = np.random.choice(len(imgs))
    img2 = PIL.Image.fromarray(imgs[i])

    return PIL.Image.blend(img, img2, v)

    