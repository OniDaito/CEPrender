""" 
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/  
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/
Author : Benjamin Blundell - k1803390@kcl.ac.uk

renderer.py - Perform splatting of gaussians with torch
functions. Based on the DirectX graphics pipeline.
"""

import torch
import numpy as np
import random
import math
import torch.nn.functional as F

from PIL import Image, ImageFilter
from util_math import angles_to_axis, mat_to_rod,\
  gen_rot_rod, gen_rot, gen_trans_xy, gen_identity,\
  gen_perspective, gen_ndc, gen_scale
from util_image import NormaliseTorch, NormaliseNull

class Splat(object):
  """ Our splatter class that generates matrices, loads 3D 
  points and spits them out to a 2D image with gaussian 
  blobs. The gaussians are computed in screen/pixel space
  with everything else following the DirectX style 
  pipeline. """

  def __init__(self, fov, aspect, near, far, size = (128, 128),\
      device="cpu", normalise_func = NormaliseNull()): 
    self.size = size
    self.near = near
    self.far = far
    self.device = device
    self.perspective = gen_perspective(fov, aspect, near, far)
    self.modelview = gen_identity(device = self.device)
    self.trans_mat = gen_identity(device = self.device)
    self.rot_mat = gen_identity(device = self.device)
    self.scale_mat = gen_scale(0.5, 0.5, 0.5, device = self.device)
    self.ndc = gen_ndc(self.size, device = self.device)
    self.xs = torch.tensor([0], dtype=torch.float32)
    self.ys = torch.tensor([0], dtype=torch.float32)
    self.normalise_func = normalise_func
    #self.w_mask = torch.tensor([0])
 
  def _gen_mats(self, points):
    """ Generate the matrices we need to do the rendering.
    These are support matrices needed by pytorch in order
    to convert out points to 2D ones all in the same 
    final tensor. """
    numbers = list(range(0, self.size[0]))
    square = [ numbers for x in numbers]
    cube = []

    for i in range(0, points.shape[0]):
      cube.append(square)
   
    self.xs = torch.tensor(cube, dtype = torch.float32,\
        device = self.device)
    self.ys = self.xs.permute([0, 2, 1])

  def to(self, device):
    """ Move this class and all it's associated data from
    one device to another. """
    self.device = device
    self.perspective = self.perspective.to(device)
    self.modelview = self.modelview.to(device)
    self.trans_mat = self.trans_mat.to(device)
    self.rot_mat = self.rot_mat.to(device)
    self.scale_mat = self.scale_mat.to(device)
    self.ndc = self.ndc.to(device)
    self.xs = self.xs.to(device)
    self.ys = self.ys.to(device)
    #self.w_mask = self.w_mask.to(device)
    return self

  def set_normalise_func(self, nfunc) :
    """ What function should we use to normalise our 
    output tensor image?"""
    self.normalise_func = nfunc
  
  def render(self, points : torch.Tensor, xr : torch.Tensor,\
    yr : torch.Tensor, zr : torch.Tensor, xt : torch.Tensor,\
    yt : torch.Tensor, sigma = 1.25):
    """ Generate a test baseline image to try and find. We take the 
    points, a mask, an output filename and 5 values that represent
    the rodrigues vector and the translation. Sigma referss to the
    spread of the gaussian and dropout is the chance that this 
    point is ignored. """
    if self.xs.shape[0] != points.shape[0]: 
      self._gen_mats(points)
    
    self.rot_mat = gen_rot_rod(xr, yr, zr)
    self.trans_mat = gen_trans_xy(xt, yt)
    self.rot_mat.requires_grad_(True)
    self.trans_mat.requires_grad_(True)
    self.modelview = torch.matmul(torch.matmul(self.scale_mat, self.rot_mat),\
      self.trans_mat)
    o = torch.matmul(self.modelview, points)

    # Divide through by W seems to work with a handy mask and sum
    #q = torch.matmul(self.perspective, o)
    #w = q * mask
    #w = torch.sum(w, 1, keepdim=True)
    #r = q / w

    s = torch.matmul(self.ndc, o)
    px = s.narrow(1, 0, 1)
    py = s.narrow(1, 1, 1)
    ex = px.expand(points.shape[0], self.xs.shape[1],\
        self.xs.shape[2])
    ey = py.expand(points.shape[0], self.ys.shape[1],\
        self.ys.shape[2])

    model =  1.0 / (2.0 * math.pi * sigma**2) * torch.sum(\
        torch.exp(-((ex - self.xs)**2 + \
        (ey-self.ys)**2)/(2*sigma**2)), dim=0)

    #normed = self.normalise_func.normalise(model)
    #return normed
    return model

  def is_valid_point(self, point):      
    ''' Check if this point is valid. We use just the standard offset
    so no rotations are applied, just the camera.'''
    o = torch.matmul(self.modelview, point)
    q = torch.matmul(self.perspective, point)
    if q[0] > q[3] or q[0] < -q[3]:
      return False
    if q[1] > q[3] or q[1] < -q[3]:
      return False
    if q[2] > q[3] or q[2] < 0.0:
      return False
    return True

if __name__ == "__main__" :
  """ Test our splat function. Can do this as follows:
  python3 splat_torch.py --sigma 2.0 --num-points 50 --rots 20 20 20
  """
  import argparse
  parser = argparse.ArgumentParser(description='PyTorch Splat')
  parser.add_argument('--cuda', action='store_true', default=False,
    help='Use cuda for this program.')
  parser.add_argument('--rots', metavar='R', type=float, nargs=3,\
      help='Rotation around X, Y, Z axis') 
  parser.add_argument('--rod', metavar='S', type=float, nargs=3,\
      help='Rodrigues vector') 
  parser.add_argument('--obj', default="",
    help='An alternative model to the torus')  
  parser.add_argument('--ply', default="",
    help='An alternative model to the torus')  
  parser.add_argument('--sigma', type=float,\
    default=1.2, help='Sigma value') 
  parser.add_argument('--num-points', type=int,\
    default=500, help='Number of points') 
  args = parser.parse_args()
  use_cuda = args.cuda
  device = torch.device("cuda" if use_cuda else "cpu")  
  
  base_points = []
  if len(args.obj) > 0:
    base_points = plyobj.load_obj(args.obj)

  elif len(args.ply) > 0:
    base_points = plyobj.load_ply(args.ply)

  splat = Splat(math.radians(90), 1.0, 1.0, 10.0, device = device)
  if args.rod != None:
    model = splat.gen_baseline_rod(args.rod[0], args.rod[1],\
      args.rod[2], base_points,\
      num_points = args.num_points, sigma = float(args.sigma))
    save_image(model, name = "gen.jpg")
    save_fits(model, name = "gen.fits")
  elif args.rots != None:
    model = splat.gen_baseline(math.radians(args.rots[0]),\
      math.radians(args.rots[1]), math.radians(args.rots[2]),\
      base_points,\
      num_points = args.num_points, sigma = float(args.sigma))
    save_image(model, name = "gen.jpg")
    save_fits(model, name = "gen.fits")
  else:
      model = splat.gen_baseline(base_points,\
        num_points = args.num_points, sigma = float(args.sigma))
      save_image(model, name = "gen.jpg")
      save_fits(model, name = "gen.fits")
    
    #point = torch.tensor([0, 0, -3.0, 1.0])
    #print(splat.is_valid_point(point))
    
    #point = torch.tensor([-2.0, 0, 3.0, 1.0])
    #print(splat.is_valid_point(point))
   
    #point = torch.tensor([-4.0, 1.0, 13.0, 1.0])
    #print(splat.is_valid_point(point))

    #point = torch.tensor([0.1884, 0.9588, 0.9180, 1.0])
    #print(splat.is_valid_point(point))
