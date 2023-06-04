import pytest

from tenqpy.core import *
import quaternion
import numpy as np

def test_real_imag():
    
    q = quaternion.as_quat_array(np.random.randn(2, 3, 4))
    
    req = real(q)
    imq = imag(q)
    
    assert np.allclose(q, req + imq)

def test_real_imag2():
    
    q = quaternion.as_quat_array(np.random.randn(2, 3, 4))

    req = real(q)
    imi = imagi(q)
    imj = imagj(q)
    imk = imagk(q)

    assert np.allclose(q, req + quaternion.x*imi + quaternion.y*imj + quaternion.z*imk)
    

def test_is_pure():
    
    q = quaternion.as_quat_array(np.random.randn(2, 3, 4))
    imq = imag(q)
    assert is_pure(q) is False
    assert is_pure(imq) is True

def test_cd():
    
    q = quaternion.as_quat_array(np.random.randn(2, 3, 4))
    
    z1, z2 = dc(q)
    p = cd(z1, z2)
    
    assert np.allclose(q, p)
