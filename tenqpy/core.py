import quaternion 
import numpy as np

def real(q):
    """Returns the real part of a quaternion array

    Args:
        q (array): quaternion array

    Returns:
        array: real part of q
    """
    
    qf = quaternion.as_float_array(q)
    
    return qf[..., 0]

def imagi(q):
    """Returns the i-imaginary part of a quaternion array

    Args:
        q (array): quaternion array

    Returns:
        array: i-imaginary part of q
    """
    
    qf = quaternion.as_float_array(q)
    
    return qf[..., 1]

def imagj(q):
    """Returns the j-imaginary part of a quaternion array

    Args:
        q (array): quaternion array

    Returns:
        array: j-imaginary part of q
    """
    
    qf = quaternion.as_float_array(q)
    
    return qf[..., 2]

def imagk(q):
    """Returns the k-imaginary part of a quaternion array

    Args:
        q (array): quaternion array

    Returns:
        array: k-imaginary part of q
    """
    
    qf = quaternion.as_float_array(q)
    
    return qf[..., 3]

def imag(q):
    """Returns the imaginary part of a quaternion array as a pure quaternion array

    Args:
        q (array): quaternion array

    Returns:
        array: imaginary part of q (quaternion array)
    """
    
    qf = quaternion.as_float_array(q)
    qf0 = qf.copy()
    qf0[..., 0] = 0
    
    imq = quaternion.as_quat_array(qf0)
    return imq


def is_pure(q, **kwargs):
    """Check whether the array q is purely imaginary. 

    Args:
        q (array): quaternion array

    Returns:
        boolean: 
    """
    
    return np.allclose(q, imag(q), **kwargs)

def c2h(z, mu=quaternion.x):
    
    # check axis is a valid one
    if is_pure(mu) is False or np.isscalar(mu) is False:
        raise ValueError('axis mu should be a pure quaternion scalar')
    
    h = np.real(z) + mu*np.imag(z)
    
    return h
    
def dc(q):
    """Cayley-Dickson decomposition of a quaternion array q as
    
    .. math::
        q = z_1 + z_2 j, \quad z_1, z_2 \in \mathbb{C}_i

    Args:
        q (array): quaternion array

    Returns:
        z1, z2: complex numpy arrays in 1, i
    """
    qf = quaternion.as_float_array(q)
    
    z1 = qf[..., 0] + 1j*qf[..., 1]
    z2 = qf[..., 2] + 1j*qf[..., 3]
    
    return z1, z2

def cd(z1, z2):
    """Construct a quaternion array such that 
    
    .. math::
        q = z_1 + z_2 j, \quad z_1, z_2 \in \mathbb{C}_i

    Args:
        z1 (array): complex numpy array
        z2 (array): complex numpy array

    Returns:
        Array: quaternion array such that q
    """
    
    q1 = c2h(z1, mu=quaternion.x)
    q2 = c2h(z2, mu=quaternion.x)

    q = q1 + q2*quaternion.y   
    
    return q



    