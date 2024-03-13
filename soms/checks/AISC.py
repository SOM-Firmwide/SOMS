import numpy as np

def SecondOrder(shapes : dict, forces : dict, Lb : float, E : float):
    """AISC Appendix 8 Approximate Second-Order Analysis

    Parameters
    ----------
    shapes : dict
        Dictionary with information about the cross section
    forces : dict
        Dictionary with the forces applied at the member
    Lb : float
        Member ubraced length
    E : float
        Young modulus

    Returns
    -------
    B1x, B1y : (np.array, np.array)
        Second order parameters
    """


    tb = 1
    Cm = 1
    alpha = 1
    Ix = shapes['Ix']
    Iy = shapes['Iy']

    Pu = abs(forces['P'].values.squeeze())
    Pu = np.reshape(Pu,(len(Ix),-1))

    # AISC Appendix 8
    Pe1x = np.pi**2 * 0.8 * tb * E * Ix / (Lb*12)**2
    Pe1y = np.pi**2 * 0.8 * tb * E * Iy / (Lb*12)**2
    B1x = Cm / (1 - alpha * Pu.T / Pe1x.values)
    B1y = Cm / (1 - alpha * Pu.T / Pe1y.values)

    return B1x, B1y


def Compression(shapes : dict, Lb : float, E : float, Fy : float):
    """AISC Chapter E Design of Members for Compression (E3)

    Parameters
    ----------
    shapes : dict
        Dictionary with information about the cross section
    forces : dict
        Dictionary with the forces applied at the member
    Lb : float
        Member ubraced length
    Fy : float
        Yield Strength

    Returns
    -------
    PhiPn
        Compression capacity
    """

    rx = shapes['rx']
    ry = shapes['ry']
    A = shapes['A']

    # AISC E3
    r = np.stack((rx, ry)).min(axis=0)
    Fe = np.pi**2 * E / (Lb * 12 / r)**2
    Fcr = np.where(Fy / Fe <= 2.25, (0.658**(Fy / Fe)) * Fy, 0.877 * Fe)

    PhiPn = 0.9 * Fcr * A
    PhiPn = PhiPn.values

    return PhiPn


def FlexureMajor(shapes : dict, Lb : float, E : float, Fy : float):
    """AISC Chapter F Design of Members for Flexure (F2)

    Parameters
    ----------
    shapes : dict
        Dictionary with information about the cross section
    Lb : float
        Member ubraced length
    E : float
        Young modulus
    Fy : float
        Yield Strength

    Returns
    -------
    PhiMnx
        Strong axis moment capacity
    """

    ho = shapes['ho']
    Iy = shapes['Iy']
    Cw = shapes['Cw']
    J = shapes['J']
    Sx = shapes['Sx']
    Zx = shapes['Zx']
    ry = shapes['ry']
    rts = shapes['rts']
    Cb = 1

    # AISC F2
    c = np.where(shapes['Type'] == "W", 1, ho / 2 *np.sqrt(Iy/Cw))

    # Yielding
    Mp = Fy * Zx

    #Lateral-Torsional Buckling
    Lp = 1.76 * ry * np.sqrt(E / Fy)
    Lr = 1.95 * rts * E / (0.7 * Fy) * np.sqrt(J * c / (Sx * ho) + np.sqrt((J * c / (Sx * ho))**2 + 6.76 * (0.7 * Fy / E)**2))

    Mn = np.where(Lb * 12 <= Lp, Mp, np.where(Lb*12 > Lr,
                                              np.stack((Cb * np.pi**2 * E / (Lb*12 / rts)**2 * np.sqrt(1 + 0.078 * (J * c) / (Sx * ho) * (Lb*12 / rts)**2)* Sx,  Mp)).min(axis=0),
                                              np.stack((Cb*(Mp - (Mp - 0.7 * Fy * Sx)*(Lb * 12 - Lp)/(Lr - Lp)),  Mp)).min(axis=0)))
    PhiMnx = 0.9 * Mn / 12 #kip-ft

    return PhiMnx


def FlexureMinor(shapes : dict, E : float, Fy : float):
    """AISC Chapter F Design of Members for Flexure (F6)

    Parameters
    ----------
    shapes : dict
        Dictionary with information about the cross section
    E : float
        Young modulus
    Fy : float
        Yield Strength

    Returns
    -------
    PhiMny
        Weak axis moment capacity
    """

    b_div_t = shapes['bf/2tf']
    Sy = shapes['Sy']
    Zy = shapes['Zy']

    # Width-to-thickness ratios: AISC TABLE B4.1b
    # Flanges of rolled I-shaped sections, channels, and tees
    Lambda_p_f = 0.38 * np.sqrt(E / Fy)
    Lambda_r_f = np.sqrt(E / Fy)
    Lambda_f = b_div_t

    # AISC F6
    #Yielding
    Mp = np.stack((Fy * Zy, 1.6 * Fy *Sy)).min(axis=0)

    # Flange Local Buckling
    Mn = np.where(Lambda_f <= Lambda_p_f, Mp, np.where(Lambda_f >= Lambda_r_f, 0.69 * E / b_div_t**2 * Sy,
                                                       Mp - (Mp - 0.7 * Fy * Sy) * (Lambda_f - Lambda_p_f) / (Lambda_r_f - Lambda_p_f)))
    PhiMny = 0.9 * Mn / 12 #kip-ft

    return  PhiMny


def CompressionDCR(forces : dict, PhiPn : np.array):
    """AISC Compression DCR

    Parameters
    ----------
    forces : dict
        Forces applied at element
    PhiPn : np.array
        Element capacity

    Returns
    -------
    DCR_P
        Demand Capacity Ration in compression
    """

    Pu = abs(forces['P'].values.squeeze())
    Pu = np.reshape(Pu,(len(PhiPn),-1))

    DCR_P = Pu.T / PhiPn

    return DCR_P


def FlexureMajorDCR(forces : dict, PhiMnx : np.array, B1x : np.array):
    """Flexure Major DCR using approximate second order analysis

    Parameters
    ----------
    forces : dict
        Forces applied at element
    PhiMnx : np.array
        Strong axis moment capacity
    B1x : np.array
        Second Order parameter

    Returns
    -------
    DCR_Mx
        Demannd Capacity Ratio (strong axis)
    """

    Mux = abs(forces['M3'].values.squeeze())
    Mux = np.reshape(Mux,(len(PhiMnx),-1))

    DCR_Mx = B1x * Mux.T / PhiMnx

    return DCR_Mx


def FlexureMinorDCR(forces : dict, PhiMny : np.array, B1y : np.array):
    """Flexure Minor DCR using approximate second order analysis

    Parameters
    ----------
    forces : dict
        Forces applied at element
    PhiMny : np.array
        Weak axis moment capacity
    B1y : np.array
        Second Order parameter

    Returns
    -------
    DCR_My
        Demand Capacity Ratio of moment (weak axis)
    """

    Muy = abs(forces['M2'].values.squeeze())
    Muy = np.reshape(Muy,(len(PhiMny),-1))

    DCR_My = B1y * Muy.T / PhiMny

    return  DCR_My


def InteractionDCR(DCR_P : np.array, DCR_Mx : np.array, DCR_My : np.array):
    """AISC Chapter H Design of Members for Combined Forces and Torsion (H1)

    Parameters
    ----------
    DCR_P : np.array
        DCR for axial load
    DCR_Mx : np.array
        DCR for moment at strong axis
    DCR_My : np.array
        DCR for moment at weak axis

    Returns
    -------
    DCR_Int
        Demand Capacity Ratio of interaction
    """

    #AISC H1
    DCR_Int = np.where(DCR_P >= 0.2, DCR_P + 8 / 9 * (DCR_Mx + DCR_My), DCR_P / 2 + (DCR_Mx + DCR_My))

    return DCR_Int