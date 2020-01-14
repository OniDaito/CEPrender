""" 
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/  
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/
Author : Benjamin Blundell - k1803390@kcl.ac.uk

util_image.py - save out our images, load images and 
normalise fits images.
"""

import torch
import numpy as np
from PIL import Image, ImageFilter

def save_image(img_tensor, name = "ten.jpg"):
  """ Save a particular tensor to an image. We add a clamp here
  to make sure it falls in range."""
  img_tensor = img_tensor.detach().cpu()
  mm = torch.max(img_tensor)
  img_tensor = img_tensor / mm
  pp = np.uint8(img_tensor.numpy() * 255)
  #img = Image.fromarray(np.uint8(torch.clamp(ten.detach(),\
  #        min= 0.0, max=1.0).numpy() * 255))
  img = Image.fromarray(np.uint8(img_tensor.numpy() * 255))
  img.save(name, "JPEG")

def save_fits(img_tensor, name = "ten.fits"):
  # Also save fits versions too
  from astropy.io import fits
  hdu = fits.PrimaryHDU(img_tensor.detach().cpu().numpy())
  hdul = fits.HDUList([hdu])  
  hdul.writeto(name)
 
def load_fits(path) :
  from astropy.io import fits
  hdul = fits.open(path)
  data = np.array(hdul[0].data, dtype=np.float32)
  #print("data",data)
  return torch.tensor(data, dtype=torch.float32)

def load_image(path) :
  im = Image.open(path)
  nm = np.asarray(im, dtype=np.float32)
  nm = nm / 255.0
  return torch.tensor(nm, dtype=torch.float32)


# A null normaliser that just returns the image
# Useful if we want to test renderer with no
# normalisation
class NormaliseNull(object):
  def normalise(self, img):
    return img

# we use this in the dataloader too so we have it here
# It needs to be a class so it can be pickled, which is 
# less elegant than a closure :/

class NormaliseTorch(object):
  '''Our simulator does no balancing so we perform a scaling
  and a sort-of-normalising to get things into the right 
  range. This matches the same output as our network. '''

  def normalise(self, img):
    # Replaced this with a new one for the real data
    intensity = torch.sum(img)
    dimg = img / intensity * 100.0
    #dimg = img / 10.0 # really simple but might just work.
    return dimg

class NormaliseMinMax(object):
  def __init__(self, min_intensity, max_intensity, image_size=(128,128)) :
    self.min_intensity = min_intensity
    self.max_intensity = max_intensity
    self.image_size = image_size

  def normalise(self, img) :
    ''' Given a min max, scale each pixel so the entire count sums to 1.'''
    
    if self.min_intensity != self.max_intensity :
      di = (torch.sum(img) - self.min_intensity) / (self.max_intensity - self.min_intensity)
      dimg = img * (1.0 / (self.image_size[0] * self.image_size[1]) * di)
      return dimg
    else:
      return img / self.max_intensity