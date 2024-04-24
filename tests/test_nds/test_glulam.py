# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 00:36:14 2024

@author: olek.niewiarowski
"""

import pandas as pd
import numpy as np
import os
from soms.checks import NDSGluLamDesigner


def test_ASD_factors():
    condition = 'Dry'
    temp = '100F<T<125F'
    test_file = 'v7.7-Deck-outercolumnline-ENVELOPE_test.xlsx'

    test_input = pd.read_excel(os.path.join(os.getcwd(), test_file),
                               skiprows=[0, 2],
                               sheet_name='Section Properties')

    actual = pd.read_excel(test_file,
                           skiprows=None,
                           sheet_name='Table').set_index('Index')

    ASD = NDSGluLamDesigner(test_input, method='ASD',
                            condition=condition, temp=temp, time_factor=0.9)
    table = ASD.table_from_row(0).drop(columns='factors')

    # Clean the dfs by removing "-" so we can compare them numerically
    actual_clean = clean_convert(actual)
    table_clean = clean_convert(table)

    find_mismatches(table_clean, actual_clean)
    return table


def test_ASD_fire():
    condition = 'Dry'
    temp = '100F<T<125F'

    a_char = 1.8
    exposed_sides_b = 2
    exposed_sides_d = 2

    test_file = 'v7.7-Deck-outercolumnline-ENVELOPE_test.xlsx'
    test_input = pd.read_excel(os.path.join(os.getcwd(), test_file),
                               skiprows=[0, 2],
                               sheet_name='Section Properties')

    test_input['b_fire'] = test_input['b'] - a_char*exposed_sides_b
    test_input['d_fire'] = test_input['d'] - a_char*exposed_sides_d

    actual = pd.read_excel(test_file,
                           skiprows=None,
                           sheet_name='Fire').set_index('Index')

    ASD = NDSGluLamDesigner(test_input, method='ASD',
                            condition=condition,
                            temp=temp,
                            time_factor=0.9,
                            fire_design=True)
    table = ASD.table_from_row(0, fire=True).drop(columns='factors')

    # Clean the dfs by removing "-" so we can compare them numerically
    actual_clean = clean_convert(actual)
    table_clean = clean_convert(table)

    find_mismatches(table_clean, actual_clean)
    return table


def clean_convert(df):
    # Convert all entries to floats where possible, replacing '-' with NaN
    return df.replace('-', np.nan).apply(pd.to_numeric, errors='coerce')


def find_mismatches(df_actual, df_desired, atol=1e-8, rtol=1e-5):

    # Ensure the same structure for comparison (same columns and index)
    if (not df_actual.columns.equals(df_desired.columns)) or \
            (not df_actual.index.equals(df_desired.index)):
        raise ValueError("DataFrames have different structure")

    # Handling numerical data with tolerance using DataFrame to keep the structure
    mismatches = pd.DataFrame(~np.isclose(df_actual.to_numpy(),
                                          df_desired.to_numpy(),
                                          atol=atol,
                                          rtol=rtol,
                                          equal_nan=True),
                              index=df_actual.index,
                              columns=df_actual.columns)

    # Filter out mismatches
    differences = mismatches.where(mismatches).stack().index.tolist()

    mismatch_details = []
    if differences:
        for index, column in differences:
            detail = (f"Mismatch in row {index}, column {column}:"
                      f" actual={df_actual.at[index, column]},"
                      f" desired={df_desired.at[index, column]}")
            mismatch_details.append(detail)
            print(detail)

    # Assert no mismatches for pytest
    assert not differences, "DataFrames mismatch found:\n" + \
        "\n".join(mismatch_details)


if __name__ == "__main__":

    test_ASD_factors()
    test_ASD_fire()
