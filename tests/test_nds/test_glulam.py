# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 00:36:14 2024

@author: olek.niewiarowski
"""

import pandas as pd
import numpy as np
import os
from soms.checks import NDSGluLamDesigner, get_DCRS_fire, get_DCRS


def test_ASD_factors():
    test_file = 'v7.7-Deck-outercolumnline-ENVELOPE_test.xlsx'
    directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(directory, test_file)

    test_input = pd.read_excel(file_path,
                               skiprows=[0, 2],
                               sheet_name='Section Properties')

    reference = pd.read_excel(file_path,
                              skiprows=None,
                              sheet_name='Table').set_index('Index')

    ASD = NDSGluLamDesigner(test_input, method='ASD', time_factor=0.9)
    actual = ASD.table_from_row(0).drop(columns='factors')

    # Clean the dfs by removing "-" so we can compare them numerically
    reference_clean = clean_convert(reference)
    actual_clean = clean_convert(actual)

    find_mismatches(actual_clean, reference_clean)

    # Test DCRs
    dcr_reference = pd.read_excel(file_path,
                                  skiprows=None,
                                  sheet_name='DCR')
    df = ASD.table.copy()
    df['P'] = dcr_reference['P']
    df['M3'] = dcr_reference['M3']
    df['M2'] = dcr_reference['M2']
    df['V2'] = dcr_reference['V2']
    df['V3'] = dcr_reference['V3']
    dcr = get_DCRS(df)

    dcr = dcr[dcr_reference.columns]
    find_mismatches(dcr, dcr_reference)
    return None


def test_ASD_fire():

    test_file = 'v7.7-Deck-outercolumnline-ENVELOPE_test.xlsx'
    directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(directory, test_file)
    test_input = pd.read_excel(file_path,
                               skiprows=[0, 2],
                               sheet_name='Section Properties')

    reference = pd.read_excel(file_path,
                              skiprows=None,
                              sheet_name='Fire').set_index('Index')

    ASD = NDSGluLamDesigner(test_input, method='ASD', time_factor=0.9,
                            fire_design=True)
    actual = ASD.table_from_row(0, fire=True).drop(columns='factors')

    # Clean the dfs by removing "-" so we can compare them numerically
    reference_clean = clean_convert(reference)
    actual_clean = clean_convert(actual)

    find_mismatches(actual_clean, reference_clean)

    # Test DCRs
    dcr_reference = pd.read_excel(file_path,
                                  skiprows=None,
                                  sheet_name='DCR_Fire')
    df = ASD.table.copy()
    df['P'] = dcr_reference['P']
    df['M3'] = dcr_reference['M3']
    df['M2'] = dcr_reference['M2']
    df['V2'] = dcr_reference['V2']
    df['V3'] = dcr_reference['V3']
    dcr = get_DCRS_fire(df)

    dcr = dcr[dcr_reference.columns]
    find_mismatches(dcr, dcr_reference)
    return None


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
