import math
from numbers import Number

import numpy as np

def vec(*iterable):
    """ shortcut to make numpy vector of any iterable(tuple...) or vector """
    return np.asarray(iterable if len(iterable) > 1 else iterable[0], 'f')


def normalized(vector):
    """ normalized version of any vector, with zero division check """
    norm = math.sqrt(sum(vector*vector))
    return vector / norm if norm > 0. else vector


def lerp(point_a, point_b, fraction):
    """ linear interpolation between two quantities with linear operators """
    return point_a + fraction * (point_b - point_a)


def identity():
    """ 4x4 identity matrix """
    return np.identity(4, 'f')


def ortho(left, right, bot, top, near, far):
    """ orthogonal projection matrix for OpenGL """
    dx, dy, dz = right - left, top - bot, far - near
    rx, ry, rz = -(right+left) / dx, -(top+bot) / dy, -(far+near) / dz
    return np.array([[2/dx, 0,    0,     rx],
                     [0,    2/dy, 0,     ry],
                     [0,    0,    -2/dz, rz],
                     [0,    0,    0,     1]], 'f')


def perspective(fovy, aspect, near, far):
    """ perspective projection matrix, from field of view and aspect ratio """
    _scale = 1.0/math.tan(math.radians(fovy)/2.0)
    sx, sy = _scale / aspect, _scale
    zz = (far + near) / (near - far)
    zw = 2 * far * near/(near - far)
    return np.array([[sx, 0,  0,  0],
                     [0,  sy, 0,  0],
                     [0,  0, zz, zw],
                     [0,  0, -1,  0]], 'f')


def frustum(xmin, xmax, ymin, ymax, zmin, zmax):
    """ frustum projection matrix for OpenGL, from min and max coordinates"""
    a = (xmax+xmin) / (xmax-xmin)
    b = (ymax+ymin) / (ymax-ymin)
    c = -(zmax+zmin) / (zmax-zmin)
    d = -2*zmax*zmin / (zmax-zmin)
    sx = 2*zmin / (xmax-xmin)
    sy = 2*zmin / (ymax-ymin)
    return np.array([[sx, 0,  a, 0],
                     [0, sy,  b, 0],
                     [0,  0,  c, d],
                     [0,  0, -1, 0]], 'f')


def translate(x=0.0, y=0.0, z=0.0):
    """ matrix to translate from coordinates (x,y,z) or a vector x"""
    matrix = np.identity(4, 'f')
    matrix[:3, 3] = vec(x, y, z) if isinstance(x, Number) else vec(x)
    return matrix


def scale(x, y=None, z=None):
    """scale matrix, with uniform (x alone) or per-dimension (x,y,z) factors"""
    x, y, z = (x, y, z) if isinstance(x, Number) else (x[0], x[1], x[2])
    y, z = (x, x) if y is None or z is None else (y, z)  # uniform scaling
    return np.diag((x, y, z, 1))


def sincos(degrees=0.0, radians=None):
    """ Rotation utility shortcut to compute sine and cosine of an angle. """
    radians = radians if radians else math.radians(degrees)
    return math.sin(radians), math.cos(radians)


def rotate(axis=(1., 0., 0.), angle=0.0, radians=None):
    """ 4x4 rotation matrix around 'axis' with 'angle' degrees or 'radians' """
    x, y, z = normalized(vec(axis))
    s, c = sincos(angle, radians)
    nc = 1 - c
    return np.array([[x*x*nc + c,   x*y*nc - z*s, x*z*nc + y*s, 0],
                     [y*x*nc + z*s, y*y*nc + c,   y*z*nc - x*s, 0],
                     [x*z*nc - y*s, y*z*nc + x*s, z*z*nc + c,   0],
                     [0,            0,            0,            1]], 'f')


def lookat(eye, target, up):
    """ Computes 4x4 view matrix from 3d point 'eye' to 'target',
        'up' 3d vector fixes orientation """
    view = normalized(vec(target)[:3] - vec(eye)[:3])
    up = normalized(vec(up)[:3])
    right = np.cross(view, up)
    up = np.cross(right, view)
    rotation = np.identity(4)
    rotation[:3, :3] = np.vstack([right, up, -view])
    return rotation @ translate(-eye)


# quaternion functions -------------------------------------------------------
def quaternion(x=vec(0., 0., 0.), y=0.0, z=0.0, w=1.0):
    """ Init quaternion, w=real and, x,y,z or vector x imaginary components """
    x, y, z = (x, y, z) if isinstance(x, Number) else (x[0], x[1], x[2])
    return np.array((w, x, y, z), 'f')


def quaternion_from_axis_angle(axis, degrees=0.0, radians=None):
    """ Compute quaternion from an axis vec and angle around this axis """
    sin, cos = sincos(radians=radians*0.5) if radians else sincos(degrees*0.5)
    return quaternion(normalized(vec(axis))*sin, w=cos)


def quaternion_from_euler(yaw=0.0, pitch=0.0, roll=0.0, radians=None):
    """ Compute quaternion from three euler angles in degrees or radians """
    siy, coy = sincos(yaw * 0.5, radians[0] * 0.5 if radians else None)
    sir, cor = sincos(roll * 0.5, radians[1] * 0.5 if radians else None)
    sip, cop = sincos(pitch * 0.5, radians[2] * 0.5 if radians else None)
    return quaternion(x=coy*sir*cop - siy*cor*sip, y=coy*cor*sip + siy*sir*cop,
                      z=siy*cor*cop - coy*sir*sip, w=coy*cor*cop + siy*sir*sip)


def quaternion_mul(q1, q2):
    """ Compute quaternion which composes rotations of two quaternions """
    return np.dot(np.array([[q1[0], -q1[1], -q1[2], -q1[3]],
                            [q1[1],  q1[0], -q1[3],  q1[2]],
                            [q1[2],  q1[3],  q1[0], -q1[1]],
                            [q1[3], -q1[2],  q1[1],  q1[0]]]), q2)


def quaternion_matrix(q):
    """ Create 4x4 rotation matrix from quaternion q """
    q = normalized(q)  # only unit quaternions are valid rotations.
    nxx, nyy, nzz = -q[1]*q[1], -q[2]*q[2], -q[3]*q[3]
    qwx, qwy, qwz = q[0]*q[1], q[0]*q[2], q[0]*q[3]
    qxy, qxz, qyz = q[1]*q[2], q[1]*q[3], q[2]*q[3]
    return np.array([[2*(nyy + nzz)+1, 2*(qxy - qwz),   2*(qxz + qwy),   0],
                     [2 * (qxy + qwz), 2 * (nxx + nzz) + 1, 2 * (qyz - qwx), 0],
                     [2 * (qxz - qwy), 2 * (qyz + qwx), 2 * (nxx + nyy) + 1, 0],
                     [0, 0, 0, 1]], 'f')


def quaternion_slerp(q0, q1, fraction):
    """ Spherical interpolation of two quaternions by 'fraction' """
    # only unit quaternions are valid rotations.
    q0, q1 = normalized(q0), normalized(q1)
    dot = np.dot(q0, q1)

    # if negative dot product, the quaternions have opposite handedness
    # and slerp won't take the shorter path. Fix by reversing one quaternion.
    q1, dot = (q1, dot) if dot > 0 else (-q1, -dot)

    theta_0 = math.acos(np.clip(dot, -1, 1))  # angle between input vectors
    theta = theta_0 * fraction                # angle between q0 and result
    q2 = normalized(q1 - q0*dot)              # {q0, q2} now orthonormal basis

    return q0*math.cos(theta) + q2*math.sin(theta)