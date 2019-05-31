import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
    
PI      = 3.1415926535897932384626
eps     = 1e-5

def hamilton_product(qa, qb):
    """Multiply qa by qb.
    Args:
        qa: B X N X 4 quaternions
        qb: B X N X 4 quaternions
    Returns:
        q_mult: B X N X 4
    """
    qa_0 = qa[:, :, 0]
    qa_1 = qa[:, :, 1]
    qa_2 = qa[:, :, 2]
    qa_3 = qa[:, :, 3]
    
    qb_0 = qb[:, :, 0]
    qb_1 = qb[:, :, 1]
    qb_2 = qb[:, :, 2]
    qb_3 = qb[:, :, 3]
    
    # See https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    q_mult_0 = qa_0*qb_0 - qa_1*qb_1 - qa_2*qb_2 - qa_3*qb_3
    q_mult_1 = qa_0*qb_1 + qa_1*qb_0 + qa_2*qb_3 - qa_3*qb_2
    q_mult_2 = qa_0*qb_2 - qa_1*qb_3 + qa_2*qb_0 + qa_3*qb_1
    q_mult_3 = qa_0*qb_3 + qa_1*qb_2 - qa_2*qb_1 + qa_3*qb_0
    
    return torch.stack([q_mult_0, q_mult_1, q_mult_2, q_mult_3], dim=-1)

def quat_rotate(X, q):
    """Rotate points by quaternions.
    Args:
        X: B X N X 3 points
        q: B X 4 quaternions
    Returns:
        X_rot: B X N X 3 (rotated points)
    """
    # repeat q along 2nd dim
    # ones_x = X[[0], :, :][:, :, [0]]*0 + 1
    # q = torch.unsqueeze(q, 1)*ones_x

    q = torch.unsqueeze(q, 1).expand(q.size(0), X.size(1), q.size(1))

    q_conj = torch.cat([ q[:, :, [0]] , -1*q[:, :, 1:4] ], dim=-1)
    X = torch.cat([ X[:, :, [0]]*0, X ], dim=-1)
    
    X_rot = hamilton_product(q, hamilton_product(X, q_conj))
    return X_rot[:, :, 1:4]

def orthographic_proj_withz(X, cam, offset_z=0.):
    """
    X: B x N x 3
    cam: B x 7: [sc, tx, ty, quaternions]
    Orth preserving the z.
    """
    quat = cam[:, -4:]
    X_rot = quat_rotate(X, quat)
    scale = cam[:, 0].contiguous().view(-1, 1, 1)
    trans = cam[:, 1:3].contiguous().view(cam.size(0), 1, -1)
    proj = scale * X_rot
    proj_xy = proj[:, :, :2] + trans
    proj_z = proj[:, :, 2, None] + offset_z
    return torch.cat((proj_xy, proj_z), 2)

def look_at(vertices, eye, at=None, up=None):
    assert(vertices.dim() == 3)
    
    batch_size      = vertices.size(0)

    if at is None:
        at          = Variable(torch.FloatTensor([0, 0, 0]).cuda())
    elif isinstance(at, list):
        at          = Variable(torch.FloatTensor(at).cuda())

    if up is None:
        up          = Variable(torch.FloatTensor([0, 1, 0]).cuda())
    elif isinstance(up, list):
        up          = Variable(torch.FloatTensor(up).cuda())

    if isinstance(eye, list) or isinstance(eye, tuple):
        eye         = Variable(torch.FloatTensor(eye).cuda())
    if eye.dim() == 1:
        eye         = eye.unsqueeze(0).expand([batch_size, eye.size(0)])
    if at.dim() == 1:
        at          = at.unsqueeze(0).expand([batch_size, at.size(0)])
    if up.dim() == 1:
        up          = up.unsqueeze(0).expand([batch_size, up.size(0)])
        
            
    # Normalise up values. 
    up = up/torch.norm(up)

    z_axis = (at - eye)/(torch.norm(at - eye, 2, 1, keepdim=True) + eps)
    x_axis = torch.cross(up, z_axis)
    x_axis = x_axis/(torch.norm(x_axis, 2, 1, keepdim=True) + eps)
    y_axis = torch.cross(z_axis, x_axis)
    y_axis = y_axis/(torch.norm(y_axis, 2, 1, keepdim=True) + eps)

    r = torch.cat((x_axis.unsqueeze(1), y_axis.unsqueeze(1), z_axis.unsqueeze(1)), dim=1)
    if r.size(0) != vertices.size(0):
        r = r.view(vertices.size(0), 3, 3)

    if vertices.size() != eye.size():
        eye = eye.unsqueeze(1).expand(vertices.size())

    vertices = vertices - eye
    vertices = torch.bmm(vertices, r.permute(0, 2, 1))

    return vertices

def vertices_to_faces(vertices, faces):
    assert(vertices.dim() == 3)
    assert(faces.dim() == 3)
    assert(vertices.size(0)  == faces.size(0))
    assert(vertices.size(2) == 3)
    assert(faces.size(2) == 3)

    batch_size, n_vertices, _   = vertices.size()
    batch_size, n_faces, _      = faces.size()

    offset = Variable((torch.arange(0,batch_size).type('torch.LongTensor')*(n_vertices*3)).cuda())
    offset = offset.unsqueeze(1).unsqueeze(2).expand(batch_size, n_faces, 3)
    offset0 = offset.unsqueeze(2)
    offset1 = offset.unsqueeze(2) + 1
    offset2 = offset.unsqueeze(2) + 2
    coffset = torch.cat((offset0, offset1, offset2), dim=2).permute(0,1,3,2)
    cfaces = 3*(faces.unsqueeze(3).expand(batch_size, n_faces, 3, 3))

    return vertices.view(-1)[(cfaces + coffset).view(-1)].view(batch_size, n_faces, 3, 3)

def get_points_from_angles(distance, elevation, azimuth, degrees=True):
    if degrees:
        elevation   = elevation*PI/180.0
        azimuth     = azimuth*PI/180.0
    if isinstance(distance, float) or isinstance(distance, int):
        return Variable(torch.Tensor([
              distance*np.cos(elevation)*np.sin(azimuth), 
              distance*np.sin(elevation), 
              -1*distance*np.cos(elevation)*np.cos(azimuth)]).cuda())
    else:
        return torch.cat((
                (distance*torch.cos(elevation)*torch.sin(azimuth)).unsqueeze(-1),
                (distance*torch.sin(elevation)).unsqueeze(-1),
                (-1*distance*torch.cos(elevation)*torch.cos(azimuth)).unsqueeze(-1)), dim=-1)

   
def perspective(vertices, angle=30.0):
    assert(vertices.dim() == 3)
    if isinstance(angle, float) or isinstance(angle, int):
        angle   = Variable(torch.FloatTensor([angle]).cuda())

    angle       = angle*PI/180.0
    angle       = angle.expand(vertices.size(0))

    width       = torch.tan(angle)
    width       = width.unsqueeze(1).expand(vertices.size(0), vertices.size(1))
    z           = vertices[:, :, 2]
    x           = vertices[:, :, 0]/z/width
    y           = vertices[:, :, 1]/z/width
    vertices    = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)), dim=-1)
    return vertices

# Normalise the quaternion, add 1 to scale, etc. 
def fix_cam_params(params):
    scale = params[:,0:1]
    trans = params[:,1:3]
    quat = F.normalize(params[:,3:])
    new_scale = F.relu(scale + 1)  + 1e-12
    new_trans = trans
    new_quat = quat 
    
    return torch.cat((new_scale, new_trans, new_quat), dim=1)
