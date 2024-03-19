import numpy as np
from typing import Union

Array1D = Union[np.ndarray, list]


def P_delta(Pr: Array1D,
            I: Array1D,
            Lb: Array1D,
            E: float = 29000,
            tao_b: float = 1.0,
            Cm: Union[Array1D, float] = 1.0,
            alpha: float = 1.0) -> Array1D:
    """
    Calculate the B1 multiplier for :math:`P-\delta` effects using AISC
    Appendix 8 (Approximate Second-Order Analysis), with respect to the
    provided axis.

    Parameters
    ----------
    Pr : Array1D
        Required axial strength .
    I : Array1D
        Moment of intertia about the chosen axis (in^4 or mm^4).
    Lb : Array1D
        Member unbraced length (in. or mm).
    E : float, optional
        Modulus of elasticity of steel. 29000 ksi (200,000 MPa)
    tao_b : float, optional
        Stiffness reduction parameter, see Chapter C. The default is 1.0.
    Cm : Array1D, optional
        Coefficient accounting for nonuniform moment. The default is 1.0.
    alpha : float, optional
        alpha = 1.0 (LRFD), or 1.6 (ASD). The default is 1.0.

    Returns
    -------
    Array1D
        Multiplier for :math:`P-\delta` effects B1.

    """

    # TODO: tao_b is assumed to be 1, implement Chapter C3
    # Do all forces have the same sign? (we don't want to handle edge cases)
    assert np.all(Pr > 0) if Pr[0] > 0 else np.all(Pr < 0), \
        "Not all forces have the same sign, are some columns in tension?"
    Pr = np.abs(Pr)

    # Elastic critical buckling strength (AISC A-8-5)
    Pe1 = np.pi**2 * 0.8 * tao_b * E * I / (Lb)**2

    # AISC Appendix 8, (AISC A-8-3)
    B1 = Cm / (1 - alpha * Pr/Pe1)
    return B1


def E3_compression(A: Array1D,
                   rx: Array1D,
                   ry: Array1D,
                   Lb: Union[Array1D, float],
                   Fy: Union[Array1D, float],
                   E: float = 29000) -> Array1D:
    """
    AISC Chapter E Design of Members for Compression (E3)

    Parameters
    ----------
    A : Array1D
        Member areas.
    rx : Array1D
        Radius of gyration wrt strong axis.
    ry : Array1D
        Radius of gyration wrt weak axis.
    Lb : Union[Array1D, float]
        Member unbraced length, in.
    Fy : Union[Array1D, float]
        Yield strength, ksi.
    E : float, optional
        Young's modulus, ksi. The default is 29000

    Returns
    -------
    Array1D
        Compression capacity, :math:`\phi P_n`, (kip).

    """

    # Elastic buckling stress (AISC E3-4)
    r_min = np.minimum(rx, ry)
    Fe = np.pi**2 * E / (Lb / r_min)**2

    # Critical stress, (AISC E3-2, E3-3)
    Fcr = np.where(Fy / Fe <= 2.25,
                   0.658**(Fy / Fe) * Fy,  # Inelastic
                   0.877 * Fe)  # Elastic

    PhiPn = 0.9*Fcr*A
    return PhiPn


def F2_flexure_major(section: Union[Array1D, None],
                     ho: Union[Array1D, None],
                     J: Union[Array1D, None],
                     Sx: Union[Array1D, None],
                     Zx: Union[Array1D, None],
                     ry: Union[Array1D, None],
                     rts: Union[Array1D, None],
                     Fy: Union[Array1D, float],
                     Lb: Union[Array1D, float],
                     Iy: Union[Array1D, None],
                     Cw: Union[Array1D, None],
                     Cb: Union[Array1D, float] = 1.0,
                     E: float = 29000,
                     shapes: Union[dict, None] = None) -> Array1D:
    """
    AISC Chapter F Design of Members for Flexure (F2)

    Parameters
    ----------
    ho : Union[Array1D, None]
        Distance between flange centroids (in. or mm).
    J : Union[Array1D, None]
        Torsional constant (in^4 or mm^4).
    Sx : Union[Array1D, None]
        Elastic section modulus taken about the :math:`x`-axis,
        :math:`in^3 (mm^3)`.
    Zx : Union[Array1D, None]
        Plastic section modulus taken about the :math:`x`-axis,
        :math:`in^3 (mm^3)`.
    ry : Union[Array1D, None]
        Radius of gyration about the y-axis (in. or mm).
    rts : Union[Array1D, None]
        Effective radius of gyration (in. or mm).
    Fy : Union[Array1D, float]
        Specified minimum yield strength (ksi or MPa).
    Lb : Union[Array1D, float]
        Length between points that are either braced against lateral
        displacement fo compression flange or braced against twist of the cross
        section (in. or mm).
    Iy : Union[Array1D, None]
        Moment of intertia about the channel weak axis, for channels only
        (in^4 or mm^4).
    Cw : Union[Array1D, None]
        Warping constant (in^6 or mm^6).
    Cb : Union[Array1D, float], optional
        Lateral-torsional buckling modification factor for non-uniform moment
        diagrams. The default is 1.0.
    E : float, optional
        Modulus of elasticity of steel. 29000 ksi (200,000 MPa)
    shapes : Union[dict, None], optional
        Optionally pass in section properties in a dict or DataFrame.
        The default is None.

    Returns
    -------
    Array1D
        Strong axis moment capacity, :math:`\phi M_{n_x}`, (kip-in, N-mm)

    """

    # TODO consider removing option to pass dict, currently for convenience
    if shapes is not None:
        section = shapes['Type']
        ho = shapes['ho']
        J = shapes['J']
        Sx = shapes['Sx']
        Zx = shapes['Zx']
        ry = shapes['ry']
        rts = shapes['rts']
        # only used for channels, might throw error
        Iy = shapes['Iy']
        Cw = shapes['Cw']

    # AISC F2-8
    c = np.where(section == "W",  # TODO: infer section type from name?
                 1,
                 np.where(section == 'C',
                          ho / 2 * np.sqrt(Iy/Cw),
                          )
                 )

    # Plastic moment (AISC F2-1)
    Mp = Fy * Zx

    # Limiting laterally unbraced length for the limit state of yielding,
    # (AISC F2-5)
    Lp = 1.76 * ry * np.sqrt(E / Fy)

    # Limiting laterally unbraced length for the limit state of inelastic
    # lateral-torsional buckling, (AISC F2-6)
    Lr = 1.95 * rts * E / (0.7 * Fy) * np.sqrt(J * c / (Sx * ho) +
                                               np.sqrt((J * c / (Sx * ho))**2 +
                                                       6.76*(0.7 * Fy / E)**2)
                                               )

    # Elastic lateral-torsional buckling moment (AISC F2-3, F2-4)
    Mn_elastic = Cb*np.pi**2*E / (Lb/rts)**2 *\
        np.sqrt(1 + 0.078*(J*c)/(Sx*ho) * (Lb/rts)**2) * Sx

    # Inelastic lateral-torsional buckling moment (AISC F2-2)
    Mn_inelastic = Cb*(Mp - (Mp - 0.7*Fy*Sx) * (Lb - Lp)/(Lr - Lp))

    # Nominal flexural strength
    Mn = np.where(Lb <= Lp,
                  Mp,
                  np.where(Lb > Lr,
                           np.minimum(Mp, Mn_elastic),
                           np.minimum(Mp, Mn_inelastic)
                           )
                  )

    PhiMnx = 0.9 * Mn

    return PhiMnx


def F6_flexure_minor(Sy: Array1D,
                     Zy: Array1D,
                     lambda_f: Array1D,
                     Fy: Array1D,
                     E: float = 29000) -> Array1D:
    """
    AISC Chapter F Design of Members for Flexure (F6)

    Parameters
    ----------
    Sy : Array1D
        Elastic section modulus taken about the :math:`y`-axis,
        :math:`in^3 (mm^3)`.
    Zy : Array1D
        Plastic section modulus taken about the :math:`y`-axis,
        :math:`in^3 (mm^3)`.
    lambda_f : Array1D
        Slenderness parameter, equal to :math:`\frac{b_f}{2 t_f` for I-shapes,
                                                          see AISC F6-4.
    Fy : Array1D
        Specified minimum yield strength (ksi or MPa).
    E : float, optional
        Modulus of elasticity of steel. 29000 ksi (200,000 MPa)

    Returns
    -------
    Array1D
        Weak axis moment capacity, :math:`\phi M_{n_y}`, (kip-in, N-mm).

    """

    # Width-to-thickness ratios: AISC TABLE B4.1b
    # Flanges of rolled I-shaped sections, channels, and tees
    lambda_p_f = 0.38 * np.sqrt(E / Fy)  # Limit for compact flange
    lambda_r_f = np.sqrt(E / Fy)  # Limit for noncompact flange

    # Plastic moment (AISC F6-1)
    Mp = np.minimum(Fy*Zy, 1.6*Fy*Sy)

    # Flange Local Buckling
    # TODO: verify these inequalities
    is_compact = lambda_f <= lambda_p_f
    is_slender = lambda_f >= lambda_r_f

    # Elastic flange local buckling moment (AISC F6-3, F6-4)
    Mn_elastic = (0.69*E*Sy) / lambda_f**2

    # Nominal flexural strength
    # For non-compact flanges, interpolate between Mp and reduced moment
    Mn = np.where(is_compact,
                  Mp,
                  np.where(is_slender,
                           Mn_elastic,
                           Mp - (Mp - 0.7 * Fy * Sy) *
                           (lambda_f - lambda_p_f) / (lambda_r_f - lambda_p_f)
                           )
                  )

    PhiMny = 0.9*Mn
    return PhiMny


def H1_interaction(Pr: Array1D,
                   Pc: Array1D,
                   Mrx: Array1D,
                   Mcx: Array1D,
                   Mry: Array1D,
                   Mcy: Array1D) -> Array1D:
    """
    AISC Chapter H Design of Members for Combined Forces and Torsion (H1)

    Parameters
    ----------
    Pr : Array1D
        Required axial strength (kips, N).
    Pc : Array1D
        Available axial strength, see Chapter E, (kips, N).
    Mrx : Array1D
        Required strong axis flexural strength (kip-in, N-mm).
    Mcx : Array1D
        Available strong axis flexural strength, see Chapter F, (kip-in, N-mm).
    Mry : Array1D
        Required weak axis flexural strength (kip-in, N-mm).
    Mcy : Array1D
        Available weak axis flexural strength, see Chapter F, (kip-in, N-mm).

    Returns
    -------
    Array1D
        Interaction ratio.

    """

    # AISC H1
    DCR = np.where(Pr/Pc >= 0.2,
                   Pr/Pc + 8/9 * (Mrx/Mcx + Mry/Mcy),
                   Pr/Pc/2 + (Mrx/Mcx + Mry/Mcy)
                   )

    return DCR
