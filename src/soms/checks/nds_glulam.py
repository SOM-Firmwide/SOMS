# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:35:25 2024

@author: olek.niewiarowski
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
import time
import warnings
# TODO: discussion with SW 4/24
'''
- fire rating info should be rowwise props
- ability to quickly check sections for user loads
'''
# NDS 2.3.2: Load Duration Factor, $C_D$ (ASD ONLY)
C_D = pd.Series({'Permanent': 0.9,
                 'Ten years': 1.0,
                 'Two months': 1.15,
                 'Seven days': 1.25,
                 'Ten minutes': 1.6,
                 'Impact': 2.0}, name='CD')

# NDS N.3.3: Time Effect Factor, $\lambda$ (LRFD ONLY)
lmbda = pd.Series([0.6, 0.7, 0.8, 1.0, 1.25], name='lambda')

# %% NDS 5.1.4: Wet Service Factor, $C_M$
# Dry conditions defined by moisture content < 16%
# Section 5.1.4 see also (S5.1.5. 2005)
C_M = pd.DataFrame({'Wet': {'Fbx_pos': 0.8,
                            'Fby': 0.8,
                            'Ft': 0.8,
                            'Frt': 0.8,
                            'Fvx': 0.875,
                            'Fvy': 0.875,
                            'Fc_perp': 0.53,
                            'Fc': 0.73,
                            'E': 0.833,
                            'E_min': 0.833}})
C_M['Dry'] = 1.0

# %% NDS 2.3.3: Tempurature Factor, $C_t$
# Define the index for the DataFrame
index = ['Fbx_pos', 'Fby', 'Fvx', 'Fvy', 'Fc',
         'Fc_perp', 'Ft', 'Frt', 'E', 'E_min']

C_t_data = {
    ("Dry", "T<100F"): [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ("Dry", "100F<T<125F"): [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9],
    ("Dry", "125F<T<150F"): [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.9, 0.9, 0.9, 0.9],
    ("Wet", "T<100F"): [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ("Wet", "100F<T<125F"): [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.9, 0.9, 0.9, 0.9],
    ("Wet", "125F<T<150F"): [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9]
}

columns = pd.MultiIndex.from_tuples(list(C_t_data.keys()))
C_t = pd.DataFrame(C_t_data, index=index, columns=columns)
del index, C_t_data, columns

# %% Fire Design

# TODO verify, e.g Fv vals in spreadsheet but not NDS 16.2.2?
C_fire = pd.Series({'Fbx_posf': 2.85,
                    'Fbyf': 2.85,
                    'Ftf': 2.85,
                    'Fvyf': 2.75,
                    'Fvxf': 2.75,
                    'Frtf': 2.85,
                    'Fcf': 2.58,
                    'Fc_perpf': 2.58},
                   name='Design Stress to Member Strength Factor')
# %% NDS 2.3.5: Format Conversion Factor, $K_F$, (LRFD ONLY)

KF = pd.Series({'Fbx_pos': 2.54,
                'Fby': 2.54,
                'Ft': 2.70,
                'Frt': 2.88,
                'Fvx': 2.88,
                'Fvy': 2.88,
                'Fc_perp': 1.67,
                'Fc': 2.40,
                'E': 1.0,
                'E_min': 1.76}, name='KF')

# %% NDS 2.3.6: Resistance Factors, $\phi$, (LRFD ONLY)
phi = pd.Series({'Fbx_pos': 0.85,
                 'Fby': 0.85,
                 'Ft': 0.80,
                 'Frt': 0.75,
                 'Fvx': 0.75,
                 'Fvy': 0.75,
                 'Fc_perp': 0.90,
                 'Fc': 0.90,
                 'E': 1.0,
                 'E_min': 0.85}, name='phi')

# %% Adjustment Factors

# NDS 5.3.9: Stress Interaction Factor, $C_I$
# TODO: Implement CI
C_I = 1.0


def get_CI(*args):
    raise NotImplementedError("Stress interaction factor not yet implemented.")


def get_CC(t_lam: ArrayLike, R: ArrayLike) -> ArrayLike:
    r"""
    NDS 5.3.8: Curvature factor, :math:`C_C`.

    Parameters
    ----------
    t_lam : ArrayLike
        Lamination thickness (inches).
    R : ArrayLike
        Radius of curvature, (feet).

    Returns
    -------
    ArrayLike
        Curvature factor, :math:`C_C`.

    """
    R = np.absolute(R)
    return np.where(R > 0.0,
                    1.0 - 2000 * (t_lam / (R * 12)) ** 2,
                    1.0)


def get_Cvr(shape: str | ArrayLike) -> ArrayLike:
    r"""
    NDS 5.3.10: Shear Reduction Factor, :math:`C_{vr}`.

    Parameters
    ----------
    shape : str | ArrayLike
        String ("Rectangular" or "Nonprismatic").

    Returns
    -------
    ArrayLike
        Shear Reduction Factor, :math:`C_{vr}`.

    """
    assert np.all((shape == 'Rectangular') | (shape == "Nonprismatic")), \
        "Shape can only be Rectangular or Nonprismatic"
    return np.where(shape == 'Rectangular', 1.0, 0.72)


# @title NDS 3.3.3: Beam Stability Factor, $C_L$
# TODO: change Le units to inches.
def R_B(d: ArrayLike, b: ArrayLike,
        Le: ArrayLike,
        axis: ArrayLike,
        axis_orientation: ArrayLike) -> ArrayLike:
    r"""
    Slenderness ratio :math:`R_B` for bending members, with respect to the
     indicated axis (Eq. 3.3-5).

    Parameters
    ----------
    d : ArrayLike
        Depth (width) of bending member, inches.
    b : ArrayLike
        Breadth (thickness) of rectangular bending member, inches.
    Le : ArrayLike
        Effectuve span length of bending member, feet.
    axis : ArrayLike
        Specifed axis to calculate slenderness (x-x or y-y).
    axis_orientation : ArrayLike
        Axis orientation of member per Fig 5A (x-x or y-y).

    Returns
    -------
    ArrayLike
        Slenderness ratio for bending.

    """
    # TODO: verify this implementation
    R_B = np.where(axis == axis_orientation,
                   np.sqrt(Le*d*12/b**2),
                   np.sqrt(Le*b*12/d**2))

    if np.any(R_B > 50):
        exceeded_indices = np.where(R_B > 50)
        msg = (f"NDS 3.3.3.7: The slenderness ratio for bending members, "
               "R_B, shall not exceed 50. "
               f"Exceeded at indices: {exceeded_indices}")
        warnings.warn(msg, UserWarning)
    return R_B


def get_CL(d: ArrayLike, b: ArrayLike, Le: ArrayLike,
           axis: ArrayLike,
           axis_orientation: ArrayLike,
           method: str,
           Fb: ArrayLike,
           E_minp: ArrayLike,
           C_M: ArrayLike,
           C_t: ArrayLike,
           C_C: ArrayLike,
           C_I: ArrayLike,
           time_factor: ArrayLike) -> ArrayLike:
    r"""
    NDS 3.3.3: Beam Stability Factor, :math:`C_L`.

    Parameters
    ----------
    d : ArrayLike
        Depth (width) of bending member, inches.
    b : ArrayLike
        Breadth (thickness) of rectangular bending member, inches.
    Le : ArrayLike
        Effectuve span length of bending member, feet.
    axis : ArrayLike
        Specifed axis to calculate slenderness (x-x or y-y).
    axis_orientation : ArrayLike
        Axis orientation of member per Fig 5A (x-x or y-y).
    method : str
        ASD or LRFD.
    Fb : ArrayLike
        Reference bending design value with respect to "axis", psi.
    E_minp : ArrayLike
        Adjusted modulus of elasticity for stability calculations, psi.
    C_M : ArrayLike
        Wet service factor for Fb.
    C_t : ArrayLike
        Temperature factor for Fb.
    C_C : ArrayLike
        Curvature factor for structural glue laminated timber.
    C_I : ArrayLike
        Stress interaction factor for tapered glued laminated timber.
    time_factor : ArrayLike
        Either the load duration factor :math:`C_D` for ASD design or the time
         effect factor :math:`\lambda` for LRFD design.

    Returns
    -------
    ArrayLike
        Beam stability factor, :math:`C_L`.

    """
    RB = R_B(d, b, Le, axis, axis_orientation)
    FbE = 1.2 * E_minp / RB**2

    if method == 'ASD':
        F_bstar = Fb * C_M * C_t * C_C * C_I * time_factor
    elif method == 'LRFD':
        KF = 2.54
        phi = 0.85
        F_bstar = Fb * C_M * C_t * C_C * C_I * time_factor * KF * phi

    temp = FbE / F_bstar
    CL = (1 + temp)/1.9 - np.sqrt(((1 + temp)/1.9)**2 - (temp/0.95))
    return CL


def get_CL_fire(d: ArrayLike, b: ArrayLike, Le: ArrayLike,
                axis: ArrayLike,
                axis_orientation: ArrayLike,
                Fb: ArrayLike,
                E_minp: ArrayLike) -> ArrayLike:
    r"""
    NDS 3.3.3: Beam Stability Factor for fire design, :math:`C_L`.

    Parameters
    ----------
    d : ArrayLike
        Depth (width) of bending member, inches.
    b : ArrayLike
        Breadth (thickness) of rectangular bending member, inches.
    Le : ArrayLike
        Effectuve span length of bending member, feet.
    axis : ArrayLike
        Specifed axis to calculate slenderness (x-x or y-y).
    axis_orientation : ArrayLike
        Axis orientation of member per Fig 5A (x-x or y-y).
    Fb : ArrayLike
        Reference bending design value with respect to "axis", psi.
    E_minp : ArrayLike
        Adjusted modulus of elasticity for stability calculations, psi.

    Returns
    -------
    ArrayLike
        Beam stability factor for fire design, :math:`C_L`.

    """
    RB = R_B(d, b, Le, axis, axis_orientation)
    FbE = 1.2 * E_minp / RB**2

    F_bstar = 2.85 * Fb

    temp = FbE / F_bstar
    CL = (1 + temp)/1.9 - np.sqrt(((1 + temp)/1.9)**2 - (temp/0.95))
    return CL

# TODO: rename


def get_E_minp(method: str,
               E_min: ArrayLike,
               condition: ArrayLike,
               temperature: ArrayLike) -> ArrayLike:
    r"""
    Calculate the adjusted modulus of elasticity for beam and column stability
     calculations, :math:`E_{min'}`.

    Parameters
    ----------
    method : str
        ASD or LRFD.
    E_min : ArrayLike
        Reference modulus of elasticity for stability calculations, psi.
    condition : ArrayLike
        Service moisture condition, 'Wet' or 'Dry'.
    temperature : ArrayLike
        Temperature condition, one of 'T<100F', '100F<T<125F', '125F<T<150F')

    Returns
    -------
    ArrayLike
        Adjusted modulus of elasticity for beam and column stability
         calculations, :math:`E_{min'}`.

    """
    CM = pd.Series(condition).map(C_M.T["E_min"])
    Ct = (pd.MultiIndex.from_arrays([condition, temperature])
          .map(C_t.T['E_min'])
          )
    factor = CM * Ct
    if method == 'ASD':
        return E_min * factor
    elif method == 'LRFD':
        KF = 1.76
        phi = 0.85
        return E_min * factor * KF * phi
    else:
        raise ValueError(f"Invalid method '{method}' - must be ASD or LRFD.")


def get_CV(species: str,
           b: ArrayLike,
           d: ArrayLike,
           Lx: ArrayLike) -> ArrayLike:
    r"""
    NDS section 5.3.6: Volume Factor :math:`C_V`

    Parameters
    ----------
    species : str
        'Southern Pine' or other.
    b : ArrayLike
        Width (breadth) of rectangular bending member, inches.
        For multiple piece width layups, b = width of widest piece used in the
         layup. Thus, :math:`b \leq 10.75` in.
    d : ArrayLike
        Depth of bending member, inches.
    Lx : ArrayLike
        Length of bending member between points of zero moment, feet.

    Returns
    -------
    ArrayLike
        Volume factor, :math:`C_V`.

    """
    # x is 20 for Souther pine and 10 for all other species
    x = np.where(species == 'Southern Pine', 20, 10)
    b_ = np.minimum(b, 10.75)
    CV = np.minimum((21/Lx)**(1/x) * (12/d)**(1/x) * (5.125/b_)**(1/x), 1.0)
    return CV


def get_Cfu(axis_orientation: ArrayLike,
            b: ArrayLike,
            d: ArrayLike) -> ArrayLike:
    r"""
    NDS 5.3.7: Flat Use Factor, :math:`C_{fu}`

    Parameters
    ----------
    axis_orientation : ArrayLike
        Axis orientation of member per Fig 5A (x-x or y-y).
    b : ArrayLike
        Breadth (thickness) of rectangular bending member, inches.
    d : ArrayLike
        Depth (width) of bending member, inches.

    Returns
    -------
    ArrayLike
        Flat use factor, :math:`C_{fu}`.

    """
    # d_y, see Fig 5B.
    # Implementation: if the Cfu does not apply, we set d_y=12 such that Cfu=1.
    d_y = np.where((axis_orientation == 'x-x') & (b < 12),
                   b,
                   np.where((axis_orientation == 'y-y') & (d < 12),
                            d,
                            12)
                   )
    Cfu = (12/d_y)**(1/9)
    return Cfu


# NDS 3.7.1: Column Stability Factor, $C_p$
# TODO: add docs for cxn_fire flag, is fire ok?
def _col_slenderness(d: ArrayLike,
                     b: ArrayLike,
                     le: ArrayLike,
                     axis: ArrayLike,
                     axis_orientation: ArrayLike,
                     cxn_fire=False) -> ArrayLike:
    r"""
    Calculate the column slenderness ratio :math:`l_e / d` with respect to the
     indicated axis.

    Parameters
    ----------
    d : ArrayLike
        Depth (width) of bending member, inches.
    b : ArrayLike
        Breadth (thickness) of rectangular bending member, inches.
    le : ArrayLike
        Effectuve length of column member, feet.
    axis : ArrayLike
        Specifed axis to calculate slenderness (x-x or y-y).
    axis_orientation : ArrayLike
        Axis orientation of member per Fig 5A (x-x or y-y).

    Returns
    -------
    ArrayLike
        Slenderness ratio of compression member, :math:`l_e / d`.

    """
    limit = 50 if not cxn_fire else 75
    le_div_d = np.where(axis_orientation == axis, le*12/d, le*12/b)

    if np.any(le_div_d > limit):
        exceeded_indices = np.where(le_div_d > limit)
        msg = (f"NDS 3.7.1.4: The slenderness ratio for solid columns, le/d, "
               f"shall not exceed 50, except that during construction le/d "
               f"shall not exceed 75. "
               f"Exceeded {limit} at indices: {exceeded_indices}")
        warnings.warn(msg, UserWarning)
    return le_div_d


def get_Cp(method: str,
           d: ArrayLike,
           b: ArrayLike,
           axis_orientation: ArrayLike,
           lex: ArrayLike,
           ley: ArrayLike,
           Fc: ArrayLike,
           E_minp: ArrayLike,
           C_M: ArrayLike,
           C_t: ArrayLike,
           time_factor: ArrayLike) -> ArrayLike:
    r"""
    NDS 3.7.1: Column stability factor, :math:`C_p`.

    Parameters
    ----------
    method : str
        ASD or LRFD.
    d : ArrayLike
        Depth (width) of bending member, inches.
    b : ArrayLike
        Breadth (thickness) of rectangular bending member, inches.
    axis_orientation : ArrayLike
        Axis orientation of member per Fig 5A (x-x or y-y).
    lex : ArrayLike
        Effective length of compression member with respect to x-x axis, feet.
    ley : ArrayLike
        Effective length of compression member with respect to y-y axis, feet.
    Fc : ArrayLike
        Reference compression design value parallel to grain, psi.
    E_minp : ArrayLike
        Adjusted modulus of elasticity for stability calculations, psi.
    C_M : ArrayLike
        Wet service factor for Fc.
    C_t : ArrayLike
        Temperature factor for Fc.
    time_factor : ArrayLike
        Either the load duration factor :math:`C_D` for ASD design or the time
         effect factor :math:`\lambda` for LRFD design.

    Returns
    -------
    ArrayLike
        Column stability factor, :math:`C_p`.

    """
    # TODO: add other cases for sawn lumber and poles?
    c = 0.9  # glulam

    le_div_d = np.maximum(_col_slenderness(d, b, lex, 'x-x', axis_orientation),
                          _col_slenderness(d, b, ley, 'y-y', axis_orientation)
                          )
    if method == 'ASD':
        Fc_star = Fc * C_M * C_t * time_factor
    elif method == 'LRFD':
        KF = 2.40
        phi = 0.9
        Fc_star = Fc * C_M * C_t * time_factor * KF * phi

    FcE = 0.822 * E_minp / le_div_d**2
    return (1 + FcE/Fc_star)/(2*c) -\
        np.sqrt(((1 + FcE/Fc_star)/(2*c))**2 - (FcE/Fc_star/c))


def get_Cp_fire(d: ArrayLike,
                b: ArrayLike,
                axis_orientation: ArrayLike,
                lex: ArrayLike,
                ley: ArrayLike,
                Fc: ArrayLike,
                E_minp: ArrayLike) -> ArrayLike:
    r"""
    NDS 3.7.1: Column stability factor, :math:`C_p`.

    Parameters
    ----------
    d : ArrayLike
        Depth (width) of bending member, inches.
    b : ArrayLike
        Breadth (thickness) of rectangular bending member, inches.
    axis_orientation : ArrayLike
        Axis orientation of member per Fig 5A (x-x or y-y).
    lex : ArrayLike
        Effective length of compression member with respect to x-x axis, feet.
    ley : ArrayLike
        Effective length of compression member with respect to y-y axis, feet.
    Fc : ArrayLike
        Reference compression design value parallel to grain, psi.
    E_minp : ArrayLike
        Adjusted modulus of elasticity for stability calculations, psi.

    Returns
    -------
    ArrayLike
        Column stability factor, :math:`C_p`.

    """
    # TODO: add other cases for sawn lumber and poles?
    c = 0.9  # glulam

    le_div_d = np.maximum(_col_slenderness(d, b, lex, 'x-x', axis_orientation,
                                           cxn_fire=True),
                          _col_slenderness(d, b, ley, 'y-y',
                                           axis_orientation,
                                           cxn_fire=True)
                          )

    Fc_star = 2.58 * Fc

    FcE = 2.03 * 0.822 * E_minp / le_div_d**2
    return (1 + FcE/Fc_star)/(2*c) -\
        np.sqrt(((1 + FcE/Fc_star)/(2*c))**2 - (FcE/Fc_star/c))


def get_Cb(lb: ArrayLike) -> ArrayLike:
    r"""
    NDS 3.10.4 & 5.3.12: Section Bearing area factor, :math:`C_b`

    Parameters
    ----------
    lb : ArrayLike
        Bearing length measured parallel to grain, inches.

    Returns
    -------
    ArrayLike
        Section Bearing area factor, :math:`C_b`

    """
    return (lb + 0.375)/lb

# %% Core Functions

# TODO rename
# TODO: A_net # Net Section Area (NDS Section 3.1.2)


class NDSGluLamDesigner:

    def __init__(self, section_properties: pd.DataFrame,
                 method: str = None,
                 time_factor: float = None,
                 fire_design: bool = False) -> object:

        self.method = method
        self.fire_design = fire_design

        assert (method == 'LRFD') or (method == 'ASD'), \
            f"Unrecognized method {method}."
        _time_factor_name = 'lambda' if method == 'LRFD' else 'CD'

        if time_factor is None:
            # No time_factor provided, assume time factors in df
            # Create combinations of section/load combinations
            # Take the cartesian product of section and load combo/time factors
            _time_factor_dict = {'ASD': C_D, 'LRFD': lmbda}
            df_props = pd.merge(section_properties,
                                _time_factor_dict[method],
                                how='cross')
            print(
                f"Loaded {len(section_properties)} unique section properties."
            )
        else:
            # one time factor provided, usually for simple checks/tests
            df_props = section_properties.copy()
            df_props[_time_factor_name] = time_factor
            self.time_factor = time_factor

        table = get_factors(df_props, self.method,
                            fire_design=fire_design)
        table = _build_section_properties(table['b'], table['d'], df=table)
        self.table = apply_factors(
            table, self.method)

        print(
            f"""Initialized {method} property table of shape {
                self.table.shape}."""
        )

    def table_from_row(self, idx, fire=False):
        return table_from_df(self.table.iloc[idx], self.method, fire=fire)


def get_factors(df: pd.DataFrame,
                method: str,
                fire_design=False) -> pd.DataFrame:
    """
    Calculate adjustment factors for a DataFrame input of section properties.
    The section properties must include:
        - d, depth
        - b, width
        - axis_orientation, the lamination orientation per Fig 5A
        - ... To be completed

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame is assumed to contain all the necessary columns.
    method : str
        ASD or LRFD.

    Returns
    -------
    pd.DataFrame
        Returns the input DataFrame with appended columns.

    """
    assert (method == 'LRFD') or (method == 'ASD'), \
        f"Unrecognized method {method}."
    _time_factor_name = 'lambda' if method == 'LRFD' else 'CD'
    time_factor = df[_time_factor_name]

    def get_Le(coeff_lu, lu, d, coeff_d):
        return coeff_lu*lu + d*coeff_d/12

    d = df['d']
    b = df['b']
    axis_orientation = df['axis_orientation']

    df['CI'] = 1.0
    df['Cvr'] = get_Cvr(df['shape'])
    df['Frt'] = df['Cvr'] * df['Fvx'] / 3  # TODO: verify
    df['Cfu'] = get_Cfu(axis_orientation, b, d)

    df['Cb'] = get_Cb(df['lb'])

    df['CC'] = get_CC(df['t_lam'], df['R'])

    df['Lex'] = _Lex = get_Le(df['coeff_lu'],
                              df['lux'],
                              d,
                              df['coeff_d'])

    df['Ley'] = _Ley = get_Le(df['coeff_lu'],
                              df['luy'],
                              d,
                              df['coeff_d'])

    df['E_minp'] = E_minp = get_E_minp(method,
                                       df['E_min'],
                                       df['condition'],
                                       df['temperature'])

    multi_index = pd.MultiIndex.from_frame(df[['condition', 'temperature']])

    CM_Fbx_pos = df['condition'].map(C_M.T["Fbx_pos"])
    Ct_Fbx_pos = multi_index.map(C_t.T['Fbx_pos'])

    CM_Fby = df['condition'].map(C_M.T["Fby"])
    Ct_Fby = multi_index.map(C_t.T['Fby'])

    CM_Fc = df['condition'].map(C_M.T["Fby"])
    Ct_Fc = multi_index.map(C_t.T['Fby'])

    df['CL_x'] = get_CL(d, b, _Lex, 'x-x',
                        axis_orientation,
                        method,
                        df['Fbx_pos'],
                        E_minp,
                        CM_Fbx_pos,
                        Ct_Fbx_pos,
                        df['CC'],
                        C_I,
                        time_factor=time_factor)

    df['CL_y'] = get_CL(d, b, _Ley, 'y-y',
                        axis_orientation,
                        method,
                        df['Fby'],
                        E_minp,
                        CM_Fby,
                        Ct_Fby,
                        df['CC'],
                        C_I,
                        time_factor=time_factor)

    df['Cp'] = get_Cp(method, d, b, axis_orientation,
                      df['lex'], df['ley'],
                      df['Fc'], df['E_minp'],
                      CM_Fc,
                      Ct_Fc,
                      time_factor=time_factor
                      )

    df['CV'] = get_CV(df['Specie'], b, d, df['Lx'])

    # only the min applies to Fbx
    df['min(CL_x,CV)'] = np.minimum(df['CL_x'], df['CV'])

    # Fire
    if fire_design:
        d_fire = df['d_fire'] = df['d'] - df['a_char']*df['exp_d']
        b_fire = df['b_fire'] = df['b'] - df['a_char']*df['exp_b']

        for key in ['Fbx_pos', 'Fby', 'Ft', 'Fc', 'Fvx', 'Fvy', 'Frt', 'Fc_perp']:
            df[f"{key}f"] = df[key].copy()

        df['CL_xf'] = get_CL_fire(d_fire, b_fire, _Lex,
                                  'x-x', axis_orientation,
                                  df['Fbx_pos'], E_minp)

        df['CL_yf'] = get_CL_fire(d_fire, b_fire, _Ley,
                                  'y-y', axis_orientation,
                                  df['Fby'], E_minp)

        df['Cpf'] = get_Cp_fire(d_fire, b_fire, axis_orientation,
                                df['lex'], df['ley'],
                                df['Fc'], E_minp)
        df['min(CL_xf,CV)'] = np.minimum(df['CL_xf'], df['CV'])
    return df


def apply_factors(df: pd.DataFrame, method: str) -> pd.DataFrame:
    r"""
    Given a DataFrame containing section properties and adjustment factors,
     this function computes the adjusted design values :math:`F'`.

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.
    method : str
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # For ASD, factors common to all quantities are CM and Ct.
    # Remaining factors are:
    factor_mapping_ASD = {
        'Fbx_pos':  ['CD', 'min(CL_x,CV)', 'CC', 'CI'],
        'Fby':      ['CD', 'CL_y', 'Cfu', 'CI'],
        'Ft':       ['CD'],
        'Fvy':      ['CD', 'Cvr'],
        'Fvx':      ['CD', 'Cvr'],
        'Frt':      ['CD'],
        'Fc':       ['CD', 'Cp'],
        'Fc_perp':  ['Cb'],
        'E':        [],
        'E_min':    [],
        # Fire:
        'Fbx_posf': ['min(CL_xf,CV)'],
        'Fbyf':     ['CL_yf', 'Cfu'],
        'Ftf':      [],
        'Fvxf':     [],
        'Fvyf':     [],
        'Frtf':     [],
        'Fcf':      ['Cpf'],
        'Fc_perpf': [],
    }

    # For LRFD, factors common to all quantities are CM, Ct, KF, and phi.
    # Remaining factors are:
    factor_mapping_LRFD = {
        'Fbx_pos':  ['min(CL_x,CV)', 'CC', 'CI', 'lambda'],
        'Fby':      ['CL_y', 'Cfu', 'CI', 'lambda'],
        'Ft':       ['lambda'],
        'Fvy':      ['Cvr', 'lambda'],
        'Fvx':      ['Cvr', 'lambda'],
        'Frt':      ['lambda'],
        'Fc':       ['Cp', 'lambda'],
        'Fc_perp':  ['Cb'],
        'E':        [],
        'E_min':    []
    }

    copy = df.copy()

    # Multiply all quantities by CM and Ct (note this excludes fire quantities)
    copy[C_M.index] *= C_M[copy['condition']].T.values
    multi_index = pd.MultiIndex.from_frame(copy[['condition', 'temperature']])
    copy[C_t.index] *= C_t[multi_index].T.values

    if set(C_fire.index).issubset(copy.columns):
        copy[C_fire.keys()] *= C_fire

    if method == 'LRFD':
        copy[KF.keys()] *= KF
        copy[phi.keys()] *= phi

    factor_mappings = {'ASD': factor_mapping_ASD,
                       'LRFD': factor_mapping_LRFD}

    # Apply the remaining factors per the chosen mapping
    print(f"Applying {method} factors to reference design values...")

    # Determine which keys in factor_mappings are in the DataFrame
    factor_mapping = factor_mappings[method]
    existing_factors = {key: [f for f in factors if f in copy.columns]
                        for key, factors in factor_mapping.items() if key in copy.columns}

    # Create a DataFrame to hold the adjusted values
    adj_values = pd.DataFrame(index=copy.index)

    # Apply the remaining factors per the chosen mapping
    for quantity, factors in existing_factors.items():
        combined_factors = copy[factors].product(axis=1)
        adj_values[f'Adj_{quantity}'] = copy[quantity] * combined_factors

    # Concatenate the original DataFrame with the adjusted values
    return pd.concat([df, adj_values], axis=1)


def _build_section_properties(b, d, df=None, index=None):

    if df is None:
        df = pd.DataFrame(index=index, dtype='float64')

    df['A'] = A = b*d
    df['Ix'] = Ix = 1/12*b*d**3
    df['Iy'] = Iy = 1/12*d*b**3

    df['Sx'] = 1/6*b*d**2
    df['Sy'] = 1/6*d*b**2

    df['rx'] = np.sqrt(Ix/A)
    df['ry'] = np.sqrt(Iy/A)
    return df


def table_from_df(row: pd.Series, method: str,
                  fire=False) -> pd.DataFrame:
    r"""
    Display a row of a dataframe as NDS table 5.3.1.

    Parameters
    ----------
    row : pd.Series
        DESCRIPTION.
    method : str
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    assert (method == 'LRFD') or (method == 'ASD'), \
        f"Unrecognized method {method}."

    if method == 'ASD':
        if fire:
            return _fire_table(row)
        columns = ['CD', 'CM', 'Ct', 'CL', 'CV',
                   'Cfu', 'CC', 'CI', 'Cvr', 'CP', 'Cb']
        factor_mask = np.array([[1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
                                [1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                                [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                                [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])

    elif method == 'LRFD':
        columns = ['CM', 'Ct', 'CL', 'CV', 'Cfu', 'CC',
                   'CI', 'Cvr', 'CP', 'Cb', 'KF', 'phi', 'lambda']
        factor_mask = np.array([[1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1],
                                [1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1],
                                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                                [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
                                [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
                                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                                [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
                                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]])

    index = ['Fbx_pos',
             'Fby',
             'Ft',
             'Fvy',
             'Fvx',
             'Frt',
             'Fc',
             'Fc_perp',
             'E',
             'E_min']

    NDS = pd.DataFrame(index=index,
                       columns=columns,
                       data=factor_mask.astype(float))

    # Applies to all rows (CM, Ct)
    NDS['CM'] = C_M[row['condition']]
    NDS['Ct'] = C_t[row['condition']][row['temperature']]

    if method == 'LRFD':
        NDS['KF'] = KF
        NDS['phi'] = phi

    # Fill in remaining columns,
    NDS.at['Fbx_pos', 'CL'] = row['CL_x']
    NDS.at['Fby', 'CL'] = row['CL_y']
    NDS.at['Fbx_pos', 'CV'] = row['CV']
    NDS.at['Fby', 'Cfu'] = row['Cfu']
    NDS.at['Fbx_pos', 'CC'] = row['CC']

    NDS.at['Fc', 'CP'] = row['Cp']
    NDS.at['Fc_perp', 'Cb'] = row['Cb']

    # factors apply to non zero
    if method == 'ASD':
        NDS['CD'] = NDS['CD']*row['CD']
    elif method == 'LRFD':
        NDS['lambda'] = NDS['lambda']*row['lambda']

    NDS['CI'] = NDS['CI']*C_I
    NDS['Cvr'] = NDS['Cvr']*row['Cvr']

    NDS['factors'] = NDS.replace(0, 1).prod(axis=1)

    NDS.at['Fbx_pos', 'factors'] = row['Adj_Fbx_pos']/row['Fbx_pos']
    NDS['Ref'] = row[index]
    NDS['Adj'] = NDS['Ref']*NDS['factors']

    # previously calculated:
    adj_vals = (row[[f"Adj_{i}" for i in index]]
                .rename(dict([(f"Adj_{i}", i) for i in index])))
    np.testing.assert_array_almost_equal(NDS['Adj'], adj_vals)
    return NDS.replace(0, '-')


def _fire_table(row: pd.Series) -> pd.DataFrame:
    r"""
    Display a row of a dataframe as NDS table 5.3.1.

    Parameters
    ----------
    row : pd.Series
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    columns = ['C_fire', 'CV', 'Cfu', 'CL_fire', 'CP_fire']
    factor_mask = np.array([[0, 1, 0, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0]])

    index = ['Fbx_posf',
             'Fbyf',
             'Ftf',
             'Fvyf',
             'Fvxf',
             'Frtf',
             'Fcf',
             'Fc_perpf']

    NDS = pd.DataFrame(index=index,
                       columns=columns,
                       data=factor_mask.astype(float))

    NDS['C_fire'] = C_fire
    # Fill in columns
    NDS.at['Fbx_posf', 'CL_fire'] = row['CL_xf']
    NDS.at['Fbyf', 'CL_fire'] = row['CL_yf']
    NDS.at['Fbx_posf', 'CV'] = row['CV']
    NDS.at['Fbyf', 'Cfu'] = row['Cfu']

    NDS.at['Fcf', 'CP_fire'] = row['Cpf']

    NDS['factors'] = NDS.replace(0, 1).prod(axis=1)

    NDS.at['Fbx_posf', 'factors'] = row['Adj_Fbx_posf']/row['Fbx_pos']
    NDS['Ref'] = row[index]
    NDS['Adj'] = NDS['Ref']*NDS['factors']

    # previously calculated:
    adj_vals = (row[[f"Adj_{i}" for i in index]]
                .rename(dict([(f"Adj_{i}", i) for i in index])))
    np.testing.assert_array_almost_equal(NDS['Adj'], adj_vals)
    return NDS.replace(0, '-')

# %% Define NDS strength check functions


def nds_flexure(M: ArrayLike, S: ArrayLike, adj_Fb: ArrayLike) -> ArrayLike:
    r"""
    Flexure Check (NDS Section 3.3)

    Parameters
    ----------
    M : ArrayLike
        Bending moment, kip-ft.
    S : ArrayLike
        Section modulus, :math:`in^3`.
    adj_Fb : ArrayLike
        Adjusted bending design value, psi.

    Returns
    -------
    ArrayLike
        Flexural DCR.

    """
    fb = np.abs(M) * 1000 * 12 / S
    return fb/adj_Fb


def nds_compression(P: ArrayLike,
                    A_net: ArrayLike,
                    adj_Fc: ArrayLike) -> ArrayLike:
    r"""
    Compression Parallel to Grain (NDS Section 3.6)

    Parameters
    ----------
    P : ArrayLike
        Axial compressive load, kips.
    A_net : ArrayLike
        Net section of area, :math:`in^2`.
    adj_Fc : ArrayLike
        Adjusted compression design value, psi.

    Returns
    -------
    ArrayLike
        Compressive DCR parallel to grain.

    """
    compression = np.where(P < 0, -P, 0)
    fc = compression/A_net * 1000
    return fc/adj_Fc


def nds_tension(P: ArrayLike,
                A_net: ArrayLike,
                adj_Ft: ArrayLike) -> ArrayLike:
    r"""
    Tension Check (NDS Section 3.8)

    Parameters
    ----------
    P : ArrayLike
        Axial tensile load, kips.
    A_net : ArrayLike
        Net section of area, :math:`in^2`.
    adj_Ft : ArrayLike
        Adjusted tensile design value, psi.

    Returns
    -------
    ArrayLike
        Tensile DCR.

    """
    tension = np.where(P > 0, P, 0)
    ft = tension/A_net * 1000
    return ft/adj_Ft


def nds_shear(V2: ArrayLike,
              V3: ArrayLike,
              A: ArrayLike,
              axis_orientation: ArrayLike,
              adj_Fvx: ArrayLike,
              adj_Fvy: ArrayLike) -> ArrayLike:
    r"""
    Shear Check - Parallel to Grain (NDS Section 3.4).
     For rectangular sections only.

    Parameters
    ----------
    V2 : ArrayLike
        Shear (major), kips.
    V3 : ArrayLike
        Shear (minor), kips.
    A : ArrayLike
        Section area of rectangular section, :math:`in^2`.
    axis_orientation : ArrayLike
        Axis orientation of member per Fig 5A (x-x or y-y).
    adj_Fvx : ArrayLike
        Adjusted major shear design value, psi.
    adj_Fvy : ArrayLike
        Adjusted minor shear design value, psi.

    Returns
    -------
    ArrayLike
        Shear DCR.

    """
    fv = np.where(axis_orientation == 'x-x',
                  3*np.abs(V2)*1000/(2*A),
                  3*np.abs(V3)*1000/(2*A)
                  )
    Fv_p = np.where(axis_orientation == 'x-x', adj_Fvx, adj_Fvy)
    return fv/Fv_p


def nds_bending_axial_tension(P: ArrayLike,
                              M2: ArrayLike,
                              M3: ArrayLike,
                              b: ArrayLike,
                              d: ArrayLike,
                              A_net: ArrayLike,
                              adj_Ft: ArrayLike,
                              adj_Fbx_pos: ArrayLike,
                              adj_Fby: ArrayLike,
                              CL_x: ArrayLike,
                              CL_y: ArrayLike,
                              CV: ArrayLike) -> ArrayLike:
    r"""
    Bending + Axial Tension Check (NDS Section 3.9.1)

    Parameters
    ----------
    P : ArrayLike
        Axial compressive load, kips.
    M2 : ArrayLike
        Minor bending moment, kip-ft.
    M3 : ArrayLike
        Major bending moment, kip-ft.
    b : ArrayLike
        Breadth of rectangular bending member, inches.
    d : ArrayLike
        Depth of bending member, inches.
    A_net : ArrayLike
        Net section of area, :math:`in^2`.
    adj_Ft : ArrayLike
        Adjusted tensile design value, psi.
    adj_Fbx_pos : ArrayLike
        Adjusted major bending design value, psi.
    adj_Fby : ArrayLike
        Adjusted minor bending design value, psi..
    CL_x : ArrayLike
        Beam stability factor, major axis, :math:`C_{Lx}`.
    CL_y : ArrayLike
        Beam stability factor, minor axis, :math:`C_{Ly}`.
    CV : ArrayLike
        Volume factor, :math:`C_V`.

    Returns
    -------
    ArrayLike
        Interaction bending/tension DCR.

    """
    Fbx_star = adj_Fbx_pos/CL_x
    Fby_star = adj_Fby/CL_y
    Fbx_starstar = adj_Fbx_pos/CV

    # Combined Axial Tension and Bending Tension
    Sx = 1/6*b*d**2
    Sy = 1/6*d*b**2
    tension = np.where(P > 0, P, 0)
    ft = tension/A_net * 1000
    fbx = np.abs(M3) * 1000 * 12 / Sx
    fby = np.abs(M2) * 1000 * 12 / Sy
    SR1 = tension/adj_Ft + fbx/Fbx_star + fby/Fby_star
    SR2 = (fbx - ft)/Fbx_starstar + (fby - ft)/adj_Fby
    return np.maximum(SR1, SR2)


def nds_bending_axial_compression(P: ArrayLike,
                                  M2: ArrayLike,
                                  M3: ArrayLike,
                                  b: ArrayLike,
                                  d: ArrayLike,
                                  axis_orientation: ArrayLike,
                                  lex: ArrayLike,
                                  ley: ArrayLike,
                                  Lex: ArrayLike,
                                  Ley: ArrayLike,
                                  E_minp: ArrayLike,
                                  A_net: ArrayLike,
                                  adj_Fc: ArrayLike,
                                  adj_Fbx_pos: ArrayLike,
                                  adj_Fby: ArrayLike,
                                  fire: bool = False) -> ArrayLike:
    r"""
    Bending + Axial Compression or Biaxial Bending (NDS Section 3.9.2)

    Parameters
    ----------
    P : ArrayLike
        Axial compressive load, kips.
    M2 : ArrayLike
        Minor bending moment, kip-ft.
    M3 : ArrayLike
        Major bending moment, kip-ft.
    b : ArrayLike
        Breadth of rectangular bending member, inches.
    d : ArrayLike
        Depth of bending member, inches.
    axis_orientation : ArrayLike
        Axis orientation of member per Fig 5A (x-x or y-y).
    lex : ArrayLike
        Effective length of compression member with respect to x-x axis, feet.
    ley : ArrayLike
        Effective length of compression member with respect to y-y axis, feet.
    Lex : ArrayLike
        Effective length of bending member with respect to x-x axis, feet.
    Ley : ArrayLike
        Effective length of bending member with respect to y-y axis, feet.
    E_minp : ArrayLike
        Adjusted modulus of elasticity for stability calculations, psi.
    A_net : ArrayLike
        Net section of area, :math:`in^2`.
    adj_Fc : ArrayLike
        Adjusted compression design value, psi.
    adj_Fbx_pos : ArrayLike
        Adjusted major bending design value, psi.
    adj_Fby : ArrayLike
        Adjusted minor bending design value, psi.
    fire : bool, optional
        Flag for fire design. The default is False.

    Returns
    -------
    ArrayLike
        Interaction bending/compression DCR.

    """
    # TODO: rename E_minp, reorder args, rename lex/ley, Lex/Ley (confusing)

    Sx = 1/6*b*d**2
    Sy = 1/6*d*b**2

    dx = np.where(axis_orientation == 'x-x', d, b)
    dy = np.where(axis_orientation == 'y-y', d, b)
    FcEx = 0.822 * E_minp / (lex*12/dx)**2
    FcEy = 0.822 * E_minp / (ley*12/dy)**2

    RBx = R_B(d, b, Lex, 'x-x', axis_orientation)
    RBy = R_B(d, b, Ley, 'y-y', axis_orientation)

    FbEx = 1.2*E_minp / RBx**2
    if fire:
        FbEx *= 2.03
        FcEx *= 2.03
        FcEy *= 2.03
    # FbEy = 1.2*E_minp / RBy**2

    compression = np.where(P < 0, -P, 0)
    fc = compression/A_net * 1000
    fbx = np.abs(M3) * 1000 * 12 / Sx
    fby = np.abs(M2) * 1000 * 12 / Sy

    SR1 = (fc/adj_Fc)**2 + fbx/(adj_Fbx_pos*(1 - fc/FcEx)) +\
        fby/(adj_Fby*(1 - fc/FcEy - (fbx/FbEx)**2))

    SR2 = fc/FcEy + (fbx/FbEx)**2
    return np.maximum(SR1, SR2)


def nds_radial_stress(M, R, b, d, adj_Frt, shape, adj_Frc=None):
    # TODO: Need to relate bending moment direction to curvature to switch between radial tension and radial compression

    # (NDS Section 5.4.1)
    """The radial stress induced by a bending moment in a curved bending member
    of constant rectangular cross section"""

    fr = np.where(np.absolute(R) > 0,
                  3 * np.abs(M) * 1000 / (2*R*b*d),
                  0)
    return np.where(shape == 'Rectangular', fr/adj_Frt, 0)


# %% Collect Strength check functions
# TODO: improve unit handling, pass units list ['Kip', 'ft', 'F] or force_unit, length_unit
def get_DCRS(df, units='Kip-ft'):
    start_time = time.time()
    P = df['P']
    M2 = df['M2']
    M3 = df['M3']

    if units == 'Kip-in':
        M2 = M2/12
        M3 = M3/12

    b = df['b']
    d = df['d']

    df['DCR_M3'] = nds_flexure(M3, df['Sx'], df['Adj_Fbx_pos'])

    df['DCR_M2'] = nds_flexure(M2, df['Sy'], df['Adj_Fby'])

    df['DCR_C'] = nds_compression(P, df['A'], df['Adj_Fc'])

    df['DCR_T'] = nds_tension(P, df['A'], df['Adj_Ft'])

    df['DCR_V'] = nds_shear(df['V2'], df['V3'], df['A'],
                            df['axis_orientation'],
                            df['Adj_Fvx'], df['Adj_Fvy'])

    df['DCR_RM'] = nds_radial_stress(
        M3, df['R'], b, d, df['Adj_Frt'], df['shape'])

    df['DCR_TM'] = nds_bending_axial_tension(P, M2, M3,
                                             b, d, df['A'],
                                             df['Adj_Ft'], df['Adj_Fbx_pos'],
                                             df['Adj_Fby'],
                                             df['CL_x'],
                                             df['CL_y'],
                                             df['CV'])

    df['DCR_CM'] = nds_bending_axial_compression(P, M2, M3,
                                                 b, d,
                                                 df['axis_orientation'],
                                                 df['lex'], df['ley'],
                                                 df['Lex'], df['Ley'],
                                                 df['Adj_E_min'], df['A'],
                                                 df['Adj_Fc'],
                                                 df['Adj_Fbx_pos'],
                                                 df['Adj_Fby'])

    df['DCR_MAX'] = df[['DCR_M3', 'DCR_M2', 'DCR_C', 'DCR_T',
                        'DCR_V', 'DCR_RM', 'DCR_TM', 'DCR_CM']].max(axis=1)
    elapsed = time.time() - start_time
    print(f"Calculated DCRs for {len(df)} frames in {elapsed:.2f} seconds")
    return df


def get_DCRS_fire(df, units='Kip-ft'):
    start_time = time.time()
    P = df['P']
    M2 = df['M2']
    M3 = df['M3']

    if units == 'Kip-in':
        M2 = M2/12
        M3 = M3/12

    b = df['b_fire']
    d = df['d_fire']
    A = b*d

    Sxf = 1/6*b*d**2
    Syf = 1/6*d*b**2
    df['DCR_M3'] = nds_flexure(M3, Sxf, df['Adj_Fbx_posf'])

    df['DCR_M2'] = nds_flexure(M2, Syf, df['Adj_Fbyf'])

    df['DCR_C'] = nds_compression(P, A, df['Adj_Fcf'])

    df['DCR_T'] = nds_tension(P, A, df['Adj_Ftf'])

    df['DCR_V'] = nds_shear(df['V2'], df['V3'], A,
                            df['axis_orientation'],
                            df['Adj_Fvxf'], df['Adj_Fvyf'])

    df['DCR_RM'] = nds_radial_stress(
        M3, df['R'], b, d, df['Adj_Frtf'], df['shape'])

    df['DCR_TM'] = nds_bending_axial_tension(P, M2, M3,
                                             b, d, A,
                                             df['Adj_Ftf'], df['Adj_Fbx_posf'],
                                             df['Adj_Fbyf'],
                                             df['CL_xf'],
                                             df['CL_yf'],
                                             df['CV'])

    df['DCR_CM'] = nds_bending_axial_compression(P, M2, M3,
                                                 b, d,
                                                 df['axis_orientation'],
                                                 df['lex'], df['ley'],
                                                 df['Lex'], df['Ley'],
                                                 df['Adj_E_min'], A,
                                                 df['Adj_Fcf'],
                                                 df['Adj_Fbx_posf'],
                                                 df['Adj_Fbyf'],
                                                 fire=True)

    df['DCR_MAX'] = df[['DCR_M3', 'DCR_M2', 'DCR_C', 'DCR_T',
                        'DCR_V', 'DCR_RM', 'DCR_TM', 'DCR_CM']].max(axis=1)
    elapsed = time.time() - start_time
    print(f"Calculated DCRs for {len(df)} frames in {elapsed:.2f} seconds")
    return df
