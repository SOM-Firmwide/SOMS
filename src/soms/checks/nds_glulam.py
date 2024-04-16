# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:35:25 2024

@author: olek.niewiarowski
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


# NDS 2.3.2: Load Duration Factor, $C_D$ (ASD ONLY)
C_D = pd.Series({'Permanent': 0.9,
                 'Ten years': 1.0,
                 'Two months': 1.15,
                 'Seven days': 1.25,
                 'Ten minutes': 1.6,
                 'Impact': 2.0}, name='CD')

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
    assert np.all(R_B <= 50), \
        """NDS 3.3.3.7: The slenderness ratio for bending members,
         R_B shall not exceed 50"""
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


def get_E_minp(method: str,  # TODO raise error if method not recognized, typ.
               E_min: ArrayLike,
               condition: str,
               temperature: str) -> ArrayLike:
    r"""
    Calculate the adjusted modulus of elasticity for beam and column stability
     calculations, :math:`E_{min'}`.

    Parameters
    ----------
    method : str
        ASD or LRFD.
    E_min : ArrayLike
        Reference modulus of elasticity for stability calculations, psi.
    condition : str
        Service moisture condition, 'Wet' or 'Dry'.
    temperature : str
        Temperature condition, one of 'T<100F', '100F<T<125F', '125F<T<150F')

    Returns
    -------
    ArrayLike
        Adjusted modulus of elasticity for beam and column stability
         calculations, :math:`E_{min'}`.

    """
    # TODO: factor out use of tables?
    factor = C_M[condition]['E_min'] * C_t[condition][temperature]['E_min']
    if method == 'ASD':
        return E_min * factor
    elif method == 'LRFD':
        KF = 1.76
        phi = 0.85
        return E_min * factor * KF * phi


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

def _col_slenderness(d: ArrayLike,
                     b: ArrayLike,
                     le: ArrayLike,
                     axis: ArrayLike,
                     axis_orientation: ArrayLike) -> ArrayLike:
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
    le_div_d = np.where(axis_orientation == axis, le*12/d, le*12/b)
    assert np.all(le_div_d < 50), \
        """NDS 3.7.1.4: The slenderness ratio for solid columns, le/d, shall
         not exceed 50, except that during construction le/d shall not
         exceed 75."""
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
# TODO: make condition, temp, time, row-wise


class NDSGluLamDesigner:

    def __init__(self, section_properties: pd.DataFrame,
                 load_combos_w_time_factors: pd.DataFrame = None,
                 method: str = None, condition=None, temp=None, time=None) -> object:

        self.method = method
        self.condition = condition
        self.temp = temp
        self.time = time

        if method == 'LRFD':
            assert np.all(~np.isnan(load_combos_w_time_factors['lambda']))
            # Create combinations of section/load combinations
            # Take the cartesian product of section and load combos/time factors
            df_props = pd.merge(section_properties,
                                load_combos_w_time_factors,
                                how='cross')
            print(
                f"Loaded {len(section_properties)} unique section properties"
                f" and {len(load_combos_w_time_factors)} load combinations."
            )
        elif method == 'ASD':
            df_props = section_properties
            self.time = time

        table = get_factors(df_props, self.method,
                            self.condition, self.temp, time=time)
        table = _build_section_properties(table['b'], table['d'], df=table)
        self.table = apply_factors(
            table, self.method, self.condition, self.temp)

        print(
            f"""Initialized {method} property table of shape {
                self.table.shape}."""
        )

    def table_from_row(self, idx):
        return table_from_df(self.table.iloc[idx], self.method, self.condition, self.temp, time=self.time)


def get_factors(df: pd.DataFrame, method: str, condition, temp, time=None) -> pd.DataFrame:
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

    def get_Le(coeff_lu, lu, d, coeff_d):
        return coeff_lu*lu + d*coeff_d/12

    d = df['d']
    b = df['b']
    axis_orientation = df['axis_orientation']

    df['CI'] = 1.0
    df['Cvr'] = get_Cvr(df['shape'])
    df['Frt'] = df['Cvr'] * df['Fvx'] / 3
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

    df['E_minp'] = E_minp = get_E_minp(method, df['E_min'], condition, temp)

    if method == 'ASD':
        df['CD'] = time_factor = C_D[time]
    elif method == 'LRFD':
        time_factor = df['lambda']

    df['CL_x'] = get_CL(d, b, _Lex, 'x-x',
                        axis_orientation,
                        method,
                        df['Fbx_pos'],
                        E_minp,
                        C_M[condition]['Fbx_pos'],
                        C_t[condition][temp]['Fbx_pos'],
                        df['CC'],
                        C_I,
                        time_factor=time_factor)

    df['CL_y'] = get_CL(d, b, _Ley, 'y-y',
                        axis_orientation,
                        method,
                        df['Fby'],
                        E_minp,
                        C_M[condition]['Fby'],
                        C_t[condition][temp]['Fby'],
                        df['CC'],
                        C_I,
                        time_factor=time_factor)

    df['Cp'] = get_Cp(method, d, b, axis_orientation,
                      df['lex'], df['ley'],
                      df['Fc'], df['E_minp'],
                      C_M[condition]['Fc'],
                      C_t[condition][temp]['Fc'],
                      time_factor=time_factor
                      )

    df['CV'] = get_CV(df['Specie'], b, d, df['Lx'])

    return df


def apply_factors(df: pd.DataFrame, method: str, condition, temp) -> pd.DataFrame:
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
    factor_mapping_ASD = {'Fbx_pos': ['CD', 'CL_x', 'CV', 'CC', 'CI'],
                          'Fby': ['CD', 'CL_y', 'Cfu', 'CI'],
                          'Ft': ['CD'],
                          'Fvy': ['CD', 'Cvr'],
                          'Fvx': ['CD', 'Cvr'],
                          'Frt': ['CD'],
                          'Fc': ['CD', 'Cp'],
                          'Fc_perp': ['Cb'],
                          'E': [],
                          'E_min': []
                          }

    # For LRFD, factors common to all quantities are CM, Ct, KF, and phi.
    # Remaining factors are:
    factor_mapping_LRFD = {'Fbx_pos': ['CL_x', 'CV', 'CC', 'CI', 'lambda'],
                           'Fby': ['CL_y', 'Cfu', 'CI', 'lambda'],
                           'Ft': ['lambda'],
                           'Fvy': ['Cvr', 'lambda'],
                           'Fvx': ['Cvr', 'lambda'],
                           'Frt': ['lambda'],
                           'Fc': ['Cp', 'lambda'],
                           'Fc_perp': ['Cb'],
                           'E': [],
                           'E_min': []
                           }

    copy = df.copy()

    # Multiply all quantities by CM and Ct
    copy[C_M[condition].keys()] *= C_M[condition]
    copy[C_t[condition][temp].keys()] *= C_t[condition][temp]

    if method == 'LRFD':
        copy[KF.keys()] *= KF
        copy[phi.keys()] *= phi

    factor_mappings = {'ASD': factor_mapping_ASD,
                       'LRFD': factor_mapping_LRFD}

    # Apply the remaining factors per the
    adjusted_quanities = copy.apply(_apply_factors_complete,
                                    args=(factor_mappings[method],),
                                    axis=1)

    return pd.concat([df, adjusted_quanities], axis=1)


def _apply_factors_complete(row, factor_mapping):
    # Create a Series to hold the adjusted values
    adjusted_values = pd.Series(dtype='float64')

    # Iterate through each design quantity and its applicable factors
    for quantity, factors in factor_mapping.items():
        # Start with the reference value for the quantity
        # All quantities are modified by CM and Ct
        adjusted_value = row[quantity]

        # Special handling for Fbx_pos and Fby: use the smaller of CL and CV
        if quantity in ['Fbx_pos']:
            cl_value = row['CL_x']
            cv_value = row['CV']
            # Use the smaller of CL and CV
            adjusted_value *= min(cl_value, cv_value)
            # Apply other factors excluding CL and CV
            for factor in factors:
                if factor not in ['CL_x', 'CV']:
                    adjusted_value *= row[factor]
        else:
            # Apply each relevant factor by multiplying
            for factor in factors:
                adjusted_value *= row[factor]

        # Store the adjusted value in the Series
        adjusted_values[f'Adj_{quantity}'] = adjusted_value

    return adjusted_values


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


# @title Adjustment Factors Table from DataFrame Row

# Set up Adjustment Factors Table

def table_from_df(row: pd.Series, method: str, condition, temp, time=None) -> pd.DataFrame:
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

    if method == 'ASD':
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
    NDS['CM'] = C_M[condition]
    NDS['Ct'] = C_t[condition][temp]

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
        NDS['CD'] = NDS['CD']*C_D[time]
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
                                  adj_Fby: ArrayLike) -> ArrayLike:
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
def get_DCRS(df, units='Kip-ft'):
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
    return df


# %%
