import jax
import jax.numpy as jnp

# polynomial coefficients for J0

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

def j0_small(x):
    '''
    Implementation of J0 for x < 5 
    '''
    z = x * x
    # if x < 1.0e-5:
    #     return 1.0 - z/4.0

    p = (z - jnp.array(DR10)) * (z - jnp.array(DR20))
    p = p * jnp.polyval(jnp.array(RP0),z)/jnp.polyval(jnp.array(RQ0), z)
    return jnp.where(x<1e-5,1-z/4.0,p)
    

def j0_large(x):
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

def j0(z):
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

    return jnp.where(jnp.abs(z) < 5.0, j0_small(jnp.abs(z)),j0_large(jnp.abs(z)))