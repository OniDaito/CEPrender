""" 
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/  
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/
Author : Benjamin Blundell - k1803390@kcl.ac.uk

util_math.py - Useful math functions mostly found in the splatting
pipeline and a few other places."""

import math
import torch

def angles_to_axis(x_rot, y_rot, z_rot):
  """ Convert angles to angle axis. Takes a while and isnt terribly 
  efficient but should be fine. Angles are in Radians."""
  # https://en.wikipedia.org/wiki/Rotation_matrix#Conversion_from_and_to_axis%E2%80%93angle

  assert (x_rot.device ==  y_rot.device == z_rot.device)

  x_rot = torch.tensor([x_rot],\
          dtype=torch.float32,requires_grad=True, device = x_rot.device)
  y_rot = torch.tensor([y_rot],\
          dtype=torch.float32, requires_grad=True, device = x_rot.device)
  z_rot = torch.tensor([z_rot],\
          dtype=torch.float32, requires_grad=True, device = x_rot.device)

  # 0,0,0 results in a zero length vector and that is no good, so
  # we add a small epsilon instead. Bit naughty
  if x_rot == 0 and y_rot == 0 and z_rot == 0:
    x_rot = x_rot + 1e-3
    y_rot = y_rot + 1e-3
    z_rot = z_rot + 1e-3

  mat = gen_rot(x_rot, y_rot, z_rot)
  (u, a) = mat_to_rod(mat)
  return u

def mat_to_rod(mat):
  u = torch.tensor([
      mat[2][1] - mat[1][2], 
      mat[0][2] - mat[2][0],
      mat[1][0] - mat[0][1]])
  t = mat[0][0] + mat[1][1] + mat[2][2]
  a = math.acos((t - 1.0) / 2.0)
  m = math.sqrt(u[0]**2 + u[1]**2 + u[2]**2)
  u = u / m * a 
  return (u, a)

def gen_perspective(fov, aspect, near, far, device="cpu"):
    """ Generate a perspective matrix. It's symetric and uses
    near, far, aspect ratio and field-of-view.
    We don't use the OpenGL type here, we go with DirectX.
    """
    D = 1.0 / math.tan(fov/2.0)
    A = aspect
    NF = near * far
    B = 2.0 * NF / (near - far)
   
    pm = torch.tensor([[D/A,0,0,0],\
      [0,D,0,0],\
      [0,0,far/(far-near), -NF/(far-near)],\
      [0,0,1,0]],\
       dtype=torch.float32, requires_grad=False, device = device)
    return pm

def gen_identity(device="cpu") :
    return torch.tensor([ [1.0, 0, 0, 0], [0, 1.0, 0, 0],\
        [0, 0, 1.0, 0], [0, 0, 0, 1.0] ],\
        dtype=torch.float32, requires_grad = False,\
        device = device)


def gen_scale(x, y, z, device="cpu") :
    """ Given three numbers, prodice a scale matrix. This cannot
    be differentiated. It's essentially a constant. """
    return torch.tensor([ [x, 0, 0, 0], [0, y, 0, 0],\
        [0, 0, z, 0], [0, 0, 0, 1.0] ],\
        dtype=torch.float32, requires_grad = False,\
        device = device)

def gen_ndc(size, device="cpu"):
    """ Generate a normalised-device-coordinates to screen matrix."""
    ds = 1.0
    sx = 0
    sy = 0
    ms = torch.tensor([\
        [size[0] / 2.0, 0, 0,size[0] / 2.0 + sx],\
        [0, -size[1] / 2.0, 0, size[1] / 2.0 + sy],\
        [0, 0, ds / 2.0, ds / 2.0],\
        [0, 0, 0, 1]], dtype=torch.float32, device = device,\
        requires_grad=False)

    return ms

def gen_trans(x, y, z):
    """ Generate a translation matrix in x,y,z. It's
    convoluted, just as gen_rot is as we want to keep
    the ability to use backward() and autograd."""
    assert (x.device ==  y.device == z.device)

    x_mask = torch.tensor([[0,0,0,1],\
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0]], dtype=torch.float32, device = x.device)

    y_mask = torch.tensor([[0,0,0,0],\
        [0,0,0,1],
        [0,0,0,0],
        [0,0,0,0]], dtype=torch.float32, device = x.device)

    z_mask = torch.tensor([[0,0,0,0],\
        [0,0,0,0],
        [0,0,0,1],
        [0,0,0,0]], dtype=torch.float32, device = x.device)

    base = torch.tensor([[1,0,0,0],\
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]], dtype=torch.float32, device = x.device)

    t_x = x.expand_as(x_mask) * x_mask
    t_y = y.expand_as(y_mask) * y_mask
    t_z = z.expand_as(z_mask) * z_mask

    tm = t_x + t_y + t_z + base

    return tm

def gen_trans_xy(x, y):
    """ Generate a translation matrix in x and y. It's
    convoluted, just as gen_rot is as we want to keep
    the ability to use backward() and autograd."""
    assert (x.device ==  y.device)

    x_mask = torch.tensor([[0,0,0,1],\
        [0,0,0,0],
        [0,0,0,0],
        [0,0,0,0]], dtype=torch.float32, device = x.device)

    y_mask = torch.tensor([[0,0,0,0],\
        [0,0,0,1],
        [0,0,0,0],
        [0,0,0,0]], dtype=torch.float32, device = x.device)

    base = torch.tensor([[1,0,0,0],\
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]], dtype=torch.float32, device = x.device)

    t_x = x.expand_as(x_mask) * x_mask
    t_y = y.expand_as(y_mask) * y_mask

    tm = t_x + t_y + base

    return tm

def gen_rot_rod(xr : torch.Tensor, yr : torch.Tensor,\
    zr : torch.Tensor ):
  """ Generate a rotation matrix but as a Rodrigues vector
  representation. It's a little better than 3 rotations as
  there are no singularities at the poles. 0,0,0 results in 
  badness so we add a small epsilon. xr and yr and zr
  are all tensors. """
  assert (xr.device ==  yr.device == zr.device)

  if xr == 0 and yr == 0 and zr == 0:
    xr = xr + 1e-3
    yr = yr + 1e-3
    zr = zr + 1e-3

  theta = torch.sqrt(torch.pow(xr, 2) + torch.pow(yr, 2) +\
      torch.pow(zr, 2))
  #x = torch.tensor([torch.div(xr, theta)], device=device)
  #y = torch.tensor([torch.div(yr, theta)], device=device)
  #z = torch.tensor([torch.div(zr, theta)], device=device)
  x = torch.div(xr, theta)
  y = torch.div(yr, theta)
  z = torch.div(zr, theta)

  t_cos = torch.cos(theta)
  t_sin = torch.sin(theta)
  m_cos = 1.0 - t_cos

  x0 = torch.add(torch.mul(torch.pow(x, 2), m_cos), t_cos)
  x1 = torch.sub(torch.mul(torch.mul(x, y), m_cos), torch.mul(z, t_sin))
  x2 = torch.add(torch.mul(torch.mul(x, z), m_cos), torch.mul(y, t_sin))
  
  y0 = torch.add(torch.mul(torch.mul(y, x), m_cos), torch.mul(z, t_sin))
  y1 = torch.add(torch.mul(torch.pow(y, 2), m_cos), t_cos)
  y2 = torch.sub(torch.mul(torch.mul(y, z), m_cos), torch.mul(x, t_sin))

  z0 = torch.sub(torch.mul(torch.mul(z, x), m_cos), torch.mul(y, t_sin))
  z1 = torch.add(torch.mul(torch.mul(z, y), m_cos), torch.mul(x, t_sin))
  z2 = torch.add(torch.mul(torch.pow(z, 2), m_cos), t_cos)
 
  # This method should work but doesn't. Ah well. Maybe gradients?
  #zero = torch.tensor([0.0], dtype=torch.float32, device = device)
  #one = torch.tensor([1.0], dtype=torch.float32, device = device)

  #xs = torch.stack((x0,x1,x2,zero),dim=1)
  #ys = torch.stack((y0,y1,y2,zero),dim=1)
  #zs = torch.stack((z0,z1,z2,zero),dim=1)
  #ns = torch.stack((zero,zero,zero,one),dim=1)

  #tmat = torch.stack((xs,ys,zs,ns))
  #rot_mat = tmat.squeeze() 
  #return rot_mat

  # In theory this line should work but it doesn't!
  # When tested in terms of the splat it's fine but when we train it 
  # all goes wrong :/
  #rot_mat = torch.tensor([[x0, x1, x2, 0], [y0, y1, y2, 0],\
  #    [z0, z1, z2, 0], [0,0,0,1]], dtype=torch.float32, device = device,
  #    requires_grad = True)
  #return rot_mat

  x0_mask = torch.tensor([[1,0,0,0],\
      [0,0,0,0],
      [0,0,0,0],
      [0,0,0,0]], dtype=torch.float32, device = xr.device)
  
  x1_mask = torch.tensor([[0,1,0,0],\
      [0,0,0,0],
      [0,0,0,0],
      [0,0,0,0]], dtype=torch.float32, device = xr.device)

  x2_mask = torch.tensor([[0,0,1,0],\
      [0,0,0,0],
      [0,0,0,0],
      [0,0,0,0]], dtype=torch.float32, device = xr.device)

  y0_mask = torch.tensor([[0,0,0,0],\
      [1,0,0,0],
      [0,0,0,0],
      [0,0,0,0]], dtype=torch.float32, device = xr.device)
  
  y1_mask = torch.tensor([[0,0,0,0],\
      [0,1,0,0],
      [0,0,0,0],
      [0,0,0,0]], dtype=torch.float32, device = xr.device)

  y2_mask = torch.tensor([[0,0,0,0],\
      [0,0,1,0],
      [0,0,0,0],
      [0,0,0,0]], dtype=torch.float32, device = xr.device)

  z0_mask = torch.tensor([[0,0,0,0],\
      [0,0,0,0],
      [1,0,0,0],
      [0,0,0,0]], dtype=torch.float32, device = xr.device)
  
  z1_mask = torch.tensor([[0,0,0,0],\
      [0,0,0,0],
      [0,1,0,0],
      [0,0,0,0]], dtype=torch.float32, device = xr.device)

  z2_mask = torch.tensor([[0,0,0,0],\
      [0,0,0,0],
      [0,0,1,0],
      [0,0,0,0]], dtype=torch.float32, device = xr.device)

  base = torch.tensor([[0,0,0,0],\
      [0,0,0,0],\
      [0,0,0,0],\
      [0,0,0,1]], dtype=torch.float32, device = xr.device)

  rot_x = x0.expand_as(x0_mask) * x0_mask +\
      x1.expand_as(x1_mask) * x1_mask +\
      x2.expand_as(x2_mask) * x2_mask

  rot_y = y0.expand_as(y0_mask) * y0_mask +\
      y1.expand_as(y1_mask) * y1_mask +\
      y2.expand_as(y2_mask) * y2_mask

  rot_z = z0.expand_as(z0_mask) * z0_mask +\
      z1.expand_as(z1_mask) * z1_mask +\
      z2.expand_as(z2_mask) * z2_mask

  tmat = torch.add(rot_x, rot_y)
  tmat2 = torch.add(tmat, rot_z)
  rot_mat = torch.add(tmat2, base)
  return rot_mat 

def normalize(tvec) :
  """ Normalise a vector that is in tensor format."""
  d = 0
  for i in range(tvec.shape[0]):
    d += tvec[i] * tvec[i]

  d = math.sqrt(d)
  ll = [] 
  for i in range(tvec.shape[0]):
    ll.append(float(tvec[i]) / d)
  return torch.tensor(ll, dtype=tvec.dtype)

def gen_rot(x_rot : torch.Tensor, y_rot : torch.Tensor,\
    z_rot : torch.Tensor) : 
  """ Make a rotation matrix from 3 tensors of dimension [1]
  representing the angle in radians around X, Y and Z axes in this
  order. It seems very verbose but this really does seem to work.
  """

  assert (x_rot.device ==  y_rot.device == z_rot.device)

  x_sin = torch.sin(x_rot)
  x_cos = torch.cos(x_rot)
  y_sin = torch.sin(y_rot)
  y_cos = torch.cos(y_rot)
  z_sin = torch.sin(z_rot)
  z_cos = torch.cos(z_rot)

  x_sin_mask = torch.tensor([[0,0,0,0],\
      [0,0,-1,0],
      [0,1,0,0],
      [0,0,0,0]], dtype=torch.float32, device = x_rot.device)

  x_cos_mask = torch.tensor([[0,0,0,0],\
      [0,1,0,0],
      [0,0,1,0],
      [0,0,0,0]], dtype=torch.float32, device = x_rot.device)

  y_sin_mask = torch.tensor([[0,0,1,0],\
      [0,0,0,0],
      [-1,0,0,0],
      [0,0,0,0]], dtype=torch.float32, device = x_rot.device)

  y_cos_mask = torch.tensor([[1,0,0,0],\
      [0,0,0,0],
      [0,0,1,0],
      [0,0,0,0]], dtype=torch.float32, device = x_rot.device)

  z_sin_mask = torch.tensor([[0,1,0,0],\
      [-1,0,0,0],
      [0,0,0,0],
      [0,0,0,0]], dtype=torch.float32, device = x_rot.device)

  z_cos_mask = torch.tensor([[1,0,0,0],\
      [0,1,0,0],
      [0,0,0,0],
      [0,0,0,0]], dtype=torch.float32, device = x_rot.device)

  base_x = torch.tensor([[1,0,0,0],\
      [0,0,0,0],\
      [0,0,0,0],\
      [0,0,0,1]], dtype=torch.float32, device = x_rot.device)

  base_y = torch.tensor([[0,0,0,0],\
      [0,1,0,0],\
      [0,0,0,0],\
      [0,0,0,1]], dtype=torch.float32, device = x_rot.device)

  base_z = torch.tensor([[0,0,0,0],\
      [0,0,0,0],\
      [0,0,1,0],\
      [0,0,0,1]], dtype=torch.float32, device = x_rot.device)

  rot_x = x_cos.expand_as(x_cos_mask) * x_cos_mask +\
      x_sin.expand_as(x_sin_mask) * x_sin_mask + base_x

  rot_y = y_cos.expand_as(y_cos_mask) * y_cos_mask +\
      y_sin.expand_as(y_sin_mask) * y_sin_mask + base_y

  rot_z = z_cos.expand_as(z_cos_mask) * z_cos_mask +\
      z_sin.expand_as(z_sin_mask) * z_sin_mask + base_z

  # why does this line screw up but two matmuls are fine? :s
  #rot_mat = rot_x * rot_y * rot_z
  tmat = torch.matmul(rot_x, rot_y)
  rot_mat = torch.matmul(tmat, rot_z)
  return rot_mat

if __name__ == "__main__":
  with torch.enable_grad():
    x0 = torch.tensor([1.0], dtype=torch.float32, requires_grad=True, device="cuda")
    y0 = torch.tensor([1.0], dtype=torch.float32, requires_grad=True, device="cuda")
    z0 = torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device="cuda")
    ts = torch.stack( (x0,y0,z0), dim=1 )
    print(ts)
    mat = gen_rot_rod(x0,y0,z0, device="cuda")
    print(mat)
    t2 = mat_to_rod(mat)
    print(t2)
    #fs = torch.sum(mat) 
    #print(fs)
    #fs.backward()
    #print(x0.grad, y0.grad, z0.grad)


