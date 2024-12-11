import jax
import jax.numpy as jnp

"""
JAX Compatible Bessel-J0 & Bessel-J1 functions.

This code is a translation of the CEPHES library to JAX, to match the scipy implementation to machine precision.

Thank you @benjaminpop (https://github.com/benjaminpope/jax-bssel)
"""

# Polynomial Coefficients for J0
PP0 = [7.96936729297347051624E-4,
        8.28352392107440799803E-2,
        1.23953371646414299388E0,
        5.44725003058768775090E0,
        8.74716500199817011941E0,
        5.30324038235394892183E0,
        9.99999999999999997821E-1]

PQ0 = [
        9.24408810558863637013E-4,
        8.56288474354474431428E-2,
        1.25352743901058953537E0,
        5.47097740330417105182E0,
        8.76190883237069594232E0,
        5.30605288235394617618E0,
        1.00000000000000000218E0]

QP0 = [-1.13663838898469149931E-2,
        -1.28252718670509318512E0,
        -1.95539544257735972385E1,
        -9.32060152123768231369E1,
        -1.77681167980488050595E2,
        -1.47077505154951170175E2,
        -5.14105326766599330220E1,
        -6.05014350600728481186E0]

QQ0 = [1.0,
        6.43178256118178023184E1,
        8.56430025976980587198E2,
        3.88240183605401609683E3,
        7.24046774195652478189E3,
        5.93072701187316984827E3,
        2.06209331660327847417E3,
        2.42005740240291393179E2]

YP0 = [1.55924367855235737965E4,
        -1.46639295903971606143E7,
        5.43526477051876500413E9,
        -9.82136065717911466409E11,
        8.75906394395366999549E13,
        -3.46628303384729719441E15,
        4.42733268572569800351E16,
        -1.84950800436986690637E16]

YQ0 = [1.04128353664259848412E3,
        6.26107330137134956842E5,
        2.68919633393814121987E8,
        8.64002487103935000337E10,
        2.02979612750105546709E13,
        3.17157752842975028269E15,
        2.50596256172653059228E17]

DR10 = 5.78318596294678452118E0
DR20 = 3.04712623436620863991E1

RP0 = [-4.79443220978201773821E9,
        1.95617491946556577543E12,
        -2.49248344360967716204E14,
        9.70862251047306323952E15]
RQ0 = [ 1.0,
        4.99563147152651017219E2,
        1.73785401676374683123E5,
        4.84409658339962045305E7,
        1.11855537045356834862E10,
        2.11277520115489217587E12,
        3.10518229857422583814E14,
        3.18121955943204943306E16,
        1.71086294081043136091E18]

PIO4 = .78539816339744830962 # pi/4
SQ2OPI = .79788456080286535588 # sqrt(2/pi)

# Polynomial Coefficients for J1
RP1 = [ -8.99971225705559398224E8, 
        4.52228297998194034323E11,
        -7.27494245221818276015E13, 
        3.68295732863852883286E15]

RQ1 = [1.0, 
       6.20836478118054335476E2, 
       2.56987256757748830383E5, 
       8.35146791431949253037E7, 
       2.21511595479792499675E10, 
       4.74914122079991414898E12, 
       7.84369607876235854894E14, 
       8.95222336184627338078E16, 
       5.32278620332680085395E18]

PP1 = [7.62125616208173112003E-4, 
       7.31397056940917570436E-2, 
       1.12719608129684925192E0, 
       5.11207951146807644818E0, 
       8.42404590141772420927E0, 
       5.21451598682361504063E0, 
       1.00000000000000000254E0]

PQ1 = [5.71323128072548699714E-4, 
       6.88455908754495404082E-2, 
       1.10514232634061696926E0, 
       5.07386386128601488557E0, 
       8.39985554327604159757E0, 
       5.20982848682361821619E0, 
       9.99999999999999997461E-1]

QP1 = [5.10862594750176621635E-2, 
       4.98213872951233449420E0, 
       7.58238284132545283818E1, 
       3.66779609360150777800E2, 
       7.10856304998926107277E2, 
       5.97489612400613639965E2, 
       2.11688757100572135698E2, 
       2.52070205858023719784E1]

QQ1  = [1.0, 
        7.42373277035675149943E1, 
        1.05644886038262816351E3, 
        4.98641058337653607651E3, 
        9.56231892404756170795E3, 
        7.99704160447350683650E3, 
        2.82619278517639096600E3, 
        3.36093607810698293419E2]

YP1 = [1.26320474790178026440E9,
       -6.47355876379160291031E11, 
       1.14509511541823727583E14,
       -8.12770255501325109621E15, 
       2.02439475713594898196E17,
       -7.78877196265950026825E17]

YQ1 = [5.94301592346128195359E2, 
       2.35564092943068577943E5, 
       7.34811944459721705660E7, 
       1.87601316108706159478E10, 
       3.88231277496238566008E12, 
       6.20557727146953693363E14, 
       6.87141087355300489866E16, 
       3.97270608116560655612E18]

Z1 = 1.46819706421238932572E1
Z2 = 4.92184563216946036703E1
THPIO4 = 2.35619449019234492885 # 3*pi/4

def __J0_small(x):
    '''
    Implementation of J0 for x < 5 
    '''
    z = x * x
    p = (z - jnp.array(DR10)) * (z - jnp.array(DR20))
    p = p * jnp.polyval(jnp.array(RP0),z)/jnp.polyval(jnp.array(RQ0), z)
    return jnp.where(x<1e-5,1-z/4.0,p)
    

def __J0_large(x):
    '''
    Implementation of J0 for x >= 5    
    '''
    w = 5.0/x
    q = 25.0/(x*x)
    p = jnp.polyval(jnp.array(PP0), q)/jnp.polyval(jnp.array(PQ0), q)
    q = jnp.polyval(jnp.array(QP0), q)/jnp.polyval(jnp.array(QQ0), q)
    xn = x - PIO4
    p = p * jnp.cos(xn) - w * q * jnp.sin(xn)
    return p * SQ2OPI / jnp.sqrt(x)

def J0(z):
    """
    Bessel function of the first kind of order zero and a real argument 
    - using the implementation from CEPHES, translated to Jax, to match scipy to machine precision.

    Reference:
    Cephes Mathematical Library.

    Args:
        z: The sampling point(s) at which the Bessel function of the first kind are
        computed.

    Returns:
        An array of shape `x.shape` containing the values of the Bessel function
    """
    z = jnp.asarray(z)
    z, = jax._src.numpy.util.promote_dtypes_inexact(z)
    z_dtype = jax.lax.dtype(z)

    if jax._src.dtypes.issubdtype(z_dtype, complex):
      raise ValueError("complex input not supported.")

    return jnp.where(jnp.abs(z) < 5.0, __J0_small(jnp.abs(z)), __J0_large(jnp.abs(z)))

def __J1_small(x):
    '''
    Implementation of J1 for x < 5 
    '''
    z = x * x
    w = jnp.polyval(jnp.array(RP1), z) / jnp.polyval(jnp.array(RQ1), z)
    w = w * x * (z - jnp.array(Z1)) * (z - jnp.array(Z2))
    return w

def __J1_large(x):  
    '''
    Implementation of J1 for x > 5 
    '''  
    w = 5.0 / x
    z = w * w
    p = jnp.polyval(jnp.array(PP1), z) / jnp.polyval(jnp.array(PQ1), z)
    q = jnp.polyval(jnp.array(QP1), z) / jnp.polyval(jnp.array(QQ1), z)
    xn = x - THPIO4
    p = p * jnp.cos(xn) - w * q * jnp.sin(xn)
    return p * SQ2OPI / jnp.sqrt(x)

def J1(z):
    """
    Bessel function of the first kind of order one and a real argument 
    - using the implementation from CEPHES, translated to Jax, to match scipy to machine precision.

    Reference:
    Cephes mathematical library.

    Args:
        x: The sampling point(s) at which the Bessel function of the first kind are
        computed.

    Returns:
        An array of shape `x.shape` containing the values of the Bessel function
    """

    z = jnp.asarray(z)
    z, = jax._src.numpy.util.promote_dtypes_inexact(z)
    z_dtype = jax.lax.dtype(z)

    if jax._src.dtypes.issubdtype(z_dtype, complex):
      raise ValueError("complex input not supported.")

    return jnp.sign(z) * jnp.where(jnp.abs(z) < 5.0, __J1_small(jnp.abs(z)), __J1_large(jnp.abs(z)))