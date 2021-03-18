# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import copy
import math
from pytorch3d.structures import Meshes


SHAPENET_MIN_ZMIN = 0.67
SHAPENET_MAX_ZMAX = 0.92

SHAPENET_AVG_ZMIN = 0.77
SHAPENET_AVG_ZMAX = 0.90



def read_binvox_coords(f, integer_division=True, dtype=torch.float32):
    """
    Read a binvox file and return the indices of all nonzero voxels.

    This matches the behavior of binvox_rw.read_as_coord_array
    (https://github.com/dimatura/binvox-rw-py/blob/public/binvox_rw.py#L153)
    but this implementation uses torch rather than numpy, and is more efficient
    due to improved vectorization.

    I think that binvox_rw.read_as_coord_array actually has a bug; when converting
    linear indices into three-dimensional indices, they use floating-point
    division instead of integer division. We can reproduce their incorrect
    implementation by passing integer_division=False.

    Args:
      f (str): A file pointer to the binvox file to read
      integer_division (bool): If False, then match the buggy implementation from binvox_rw
      dtype: Datatype of the output tensor. Use float64 to match binvox_rw

    Returns:
      coords (tensor): A tensor of shape (N, 3) where N is the number of nonzero voxels,
           and coords[i] = (x, y, z) gives the index of the ith nonzero voxel. If the
           voxel grid has shape (V, V, V) then we have 0 <= x, y, z < V.
    """
    size, translation, scale = _read_binvox_header(f)
    storage = torch.ByteStorage.from_buffer(f.read())
    data = torch.tensor([], dtype=torch.uint8)
    data.set_(source=storage)
    vals, counts = data[::2], data[1::2]
    idxs = _compute_idxs_v2(vals, counts)
    if not integer_division:
        idxs = idxs.to(dtype)
    x_idxs = idxs // (size * size)
    zy_idxs = idxs % (size * size)
    z_idxs = zy_idxs // size
    y_idxs = zy_idxs % size
    coords = torch.stack([x_idxs, y_idxs, z_idxs], dim=1)
    return coords.to(dtype)


def _compute_idxs_v1(vals, counts):
    """ Naive version of index computation with loops """
    idxs = []
    cur = 0
    for i in range(vals.shape[0]):
        val, count = vals[i].item(), counts[i].item()
        if val == 1:
            idxs.append(torch.arange(cur, cur + count))
        cur += count
    idxs = torch.cat(idxs, dim=0)
    return idxs


def _compute_idxs_v2(vals, counts):
    """ Fast vectorized version of index computation """
    # Consider an example where:
    # vals   = [0, 1, 0, 1, 1]
    # counts = [2, 3, 3, 2, 1]
    #
    # These values of counts and vals mean that the dense binary grid is:
    # [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    #
    # So the nonzero indices we want to return are:
    # [2, 3, 4, 8, 9, 10]

    # After the cumsum we will have:
    # end_idxs = [2, 5, 8, 10, 11]
    end_idxs = counts.cumsum(dim=0)

    # After masking and computing start_idx we have:
    # end_idxs   = [5, 10, 11]
    # counts     = [3,  2,  1]
    # start_idxs = [2,  8, 10]
    mask = vals == 1
    end_idxs = end_idxs[mask]
    counts = counts[mask].to(end_idxs)
    start_idxs = end_idxs - counts

    # We initialize delta as:
    # [2, 1, 1, 1, 1, 1]
    delta = torch.ones(counts.sum().item(), dtype=torch.int64)
    delta[0] = start_idxs[0]

    # We compute pos = [3, 5], val = [3, 0]; then delta is
    # [2, 1, 1, 4, 1, 1]
    pos = counts.cumsum(dim=0)[:-1]
    val = start_idxs[1:] - end_idxs[:-1]
    delta[pos] += val

    # A final cumsum gives the idx we want: [2, 3, 4, 8, 9, 10]
    idxs = delta.cumsum(dim=0)
    return idxs


def _read_binvox_header(f):
    # First line of the header should be "#binvox 1"
    line = f.readline().strip()
    if line != b"#binvox 1":
        raise ValueError("Invalid header (line 1)")

    # Second line of the header should be "dim [int] [int] [int]"
    # and all three int should be the same
    line = f.readline().strip()
    if not line.startswith(b"dim "):
        raise ValueError("Invalid header (line 2)")
    dims = line.split(b" ")
    try:
        dims = [int(d) for d in dims[1:]]
    except ValueError:
        raise ValueError("Invalid header (line 2)")
    if len(dims) != 3 or dims[0] != dims[1] or dims[0] != dims[2]:
        raise ValueError("Invalid header (line 2)")
    size = dims[0]

    # Third line of the header should be "translate [float] [float] [float]"
    line = f.readline().strip()
    if not line.startswith(b"translate "):
        raise ValueError("Invalid header (line 3)")
    translation = line.split(b" ")
    if len(translation) != 4:
        raise ValueError("Invalid header (line 3)")
    try:
        translation = tuple(float(t) for t in translation[1:])
    except ValueError:
        raise ValueError("Invalid header (line 3)")

    # Fourth line of the header should be "scale [float]"
    line = f.readline().strip()
    if not line.startswith(b"scale "):
        raise ValueError("Invalid header (line 4)")
    line = line.split(b" ")
    if not len(line) == 2:
        raise ValueError("Invalid header (line 4)")
    scale = float(line[1])

    # Fifth line of the header should be "data"
    line = f.readline().strip()
    if not line == b"data":
        raise ValueError("Invalid header (line 5)")

    return size, translation, scale


def get_blender_intrinsic_matrix(N=None):
    """
    This is the (default) matrix that blender uses to map from camera coordinates
    to normalized device coordinates. We can extract it from Blender like this:

    import bpy
    camera = bpy.data.objects['Camera']
    render = bpy.context.scene.render
    K = camera.calc_matrix_camera(
         render.resolution_x,
         render.resolution_y,
         render.pixel_aspect_x,
         render.pixel_aspect_y)
    """
    K = [
        [2.1875, 0.0, 0.0, 0.0],
        [0.0, 2.1875, 0.0, 0.0],
        [0.0, 0.0, -1.002002, -0.2002002],
        [0.0, 0.0, -1.0, 0.0],
    ]
    K = torch.tensor(K)
    if N is not None:
        K = K.view(1, 4, 4).expand(N, 4, 4)
    return K


def blender_ndc_to_world(verts):
    """
    Inverse operation to projecting by the Blender intrinsic operation above.
    In other words, the following should hold:

    K = get_blender_intrinsic_matrix()
    verts == blender_ndc_to_world(project_verts(verts, K))
    """
    xx, yy, zz = verts.unbind(dim=1)
    a1, a2, a3 = 2.1875, 2.1875, -1.002002
    b1, b2 = -0.2002002, -1.0
    z = b1 / (b2 * zz - a3)
    y = (b2 / a2) * (z * yy)
    x = (b2 / a1) * (z * xx)
    out = torch.stack([x, y, z], dim=1)
    return out


def voxel_to_world(meshes):
    """
    When predicting voxels, we operate in a [-1, 1]^3 coordinate space where the
    intrinsic matrix has already been applied, the y-axis has been flipped to
    to align with the image plane, and the z-axis has been rescaled so the min/max
    z values in the dataset correspond to -1 / 1. This function undoes these
    transformations, and projects a Meshes from voxel-space into world space.

    TODO: This projection logic is tightly coupled to the MeshVox Dataset;
    they should maybe both be refactored?

    Input:
    - meshes: Meshes in voxel coordinate system

    Output:
    - meshes: Meshes in world coordinate system
    """
    verts = meshes.verts_packed()
    x, y, z = verts.unbind(dim=1)

    zmin, zmax = SHAPENET_MIN_ZMIN, SHAPENET_MAX_ZMAX
    m = 2.0 / (zmax - zmin)
    b = -2.0 * zmin / (zmax - zmin) - 1

    y = -y
    z = (z - b) / m
    verts = torch.stack([x, y, z], dim=1)
    verts = blender_ndc_to_world(verts)

    verts_list = list(verts.split(meshes.num_verts_per_mesh().tolist(), dim=0))
    faces_list = copy.deepcopy(meshes.faces_list())
    meshes_world = Meshes(verts=verts_list, faces=faces_list)

    return meshes_world


def compute_extrinsic_matrix(azimuth, elevation, distance):
    """
    Compute 4x4 extrinsic matrix that converts from homogenous world coordinates
    to homogenous camera coordinates. We assume that the camera is looking at the
    origin.

    Inputs:
    - azimuth: Rotation about the z-axis, in degrees
    - elevation: Rotation above the xy-plane, in degrees
    - distance: Distance from the origin

    Returns:
    - FloatTensor of shape (4, 4)
    """
    azimuth, elevation, distance = (float(azimuth), float(elevation), float(distance))
    az_rad = -math.pi * azimuth / 180.0
    el_rad = -math.pi * elevation / 180.0
    sa = math.sin(az_rad)
    ca = math.cos(az_rad)
    se = math.sin(el_rad)
    ce = math.cos(el_rad)
    R_world2obj = torch.tensor([[ca * ce, sa * ce, -se], [-sa, ca, 0], [ca * se, sa * se, ce]])
    R_obj2cam = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    R_world2cam = R_obj2cam.mm(R_world2obj)
    cam_location = torch.tensor([[distance, 0, 0]]).t()
    T_world2cam = -R_obj2cam.mm(cam_location)
    RT = torch.cat([R_world2cam, T_world2cam], dim=1)
    RT = torch.cat([RT, torch.tensor([[0.0, 0, 0, 1]])])

    # For some reason I cannot fathom, when Blender loads a .obj file it rotates
    # the model 90 degrees about the x axis. To compensate for this quirk we roll
    # that rotation into the extrinsic matrix here
    rot = torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    RT = RT.mm(rot.to(RT))

    return RT


def rotate_verts(RT, verts):
    """
    Inputs:
    - RT: (N, 4, 4) array of extrinsic matrices
    - verts: (N, V, 3) array of vertex positions
    """
    singleton = False
    if RT.dim() == 2:
        assert verts.dim() == 2
        RT, verts = RT[None], verts[None]
        singleton = True

    if isinstance(verts, list):
        verts_rot = []
        for i, v in enumerate(verts):
            verts_rot.append(rotate_verts(RT[i], v))
        return verts_rot

    R = RT[:, :3, :3]
    verts_rot = verts.bmm(R.transpose(1, 2))
    if singleton:
        verts_rot = verts_rot[0]
    return verts_rot


def project_verts(verts, P, eps=1e-1):
    """
    Project verticies using a 4x4 transformation matrix

    Inputs:
    - verts: FloatTensor of shape (N, V, 3) giving a batch of vertex positions.
    - P: FloatTensor of shape (N, 4, 4) giving projection matrices

    Outputs:
    - verts_out: FloatTensor of shape (N, V, 3) giving vertex positions (x, y, z)
        where verts_out[i] is the result of transforming verts[i] by P[i].
    """
    # Handle unbatched inputs
    singleton = False
    if verts.dim() == 2:
        assert P.dim() == 2
        singleton = True
        verts, P = verts[None], P[None]

    N, V = verts.shape[0], verts.shape[1]
    dtype, device = verts.dtype, verts.device

    # Add an extra row of ones to the world-space coordinates of verts before
    # multiplying by the projection matrix. We could avoid this allocation by
    # instead multiplying by a 4x3 submatrix of the projectio matrix, then
    # adding the remaining 4x1 vector. Not sure whether there will be much
    # performance difference between the two.
    ones = torch.ones(N, V, 1, dtype=dtype, device=device)
    verts_hom = torch.cat([verts, ones], dim=2)
    verts_cam_hom = torch.bmm(verts_hom, P.transpose(1, 2))

    # Avoid division by zero by clamping the absolute value
    w = verts_cam_hom[:, :, 3:]
    w_sign = w.sign()
    w_sign[w == 0] = 1
    w = w_sign * w.abs().clamp(min=eps)

    verts_proj = verts_cam_hom[:, :, :3] / w

    if singleton:
        return verts_proj[0]
    return verts_proj
