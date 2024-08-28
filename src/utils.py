import numpy as np
import pandas as pd
import polars as pl
import zipfile
import re
import json
import pathlib
from typing import Any, Dict, List

#==================================================
# Process data in zip files, map values, and create a JSON file for each year's data
def extract_and_process_zipfiles(
    zip_filepaths: Dict[int, str],
    NIBRS_mapper_directory: pathlib.Path
) -> None:
    """
    Extracts and processes zip files containing data and codebooks, mapping variable descriptions and values
    to create a cleaned JSON file for each year's data.

    Parameters:
        zip_filepaths (dict[int, str]): A dictionary where keys are year and values are paths to zip files.
        NIBRS_mapper_directory (pathlib.Path): The directory where the JSON output files will be saved.

    Returns:
        None
    """

    for current_year in zip_filepaths:
        current_year_zip_filepath = pathlib.Path(zip_filepaths[current_year])

        # Construct the JSON output file path
        json_filepath = pathlib.Path(
            NIBRS_mapper_directory,
            f"{current_year} - {current_year_zip_filepath.name.replace('.zip', '.json')}"
        )

        codebook_filepath = None
        with zipfile.ZipFile(current_year_zip_filepath) as czip:
            # Find the codebook file within the zip file
            for filepath in czip.namelist():
                if filepath.endswith('PI_Codebook.xlsx'):
                    codebook_filepath = filepath
                    break

            if codebook_filepath:
                codebook_excel_object = pd.ExcelFile(czip.open(codebook_filepath))
                sheet_names = codebook_excel_object.sheet_names

        if codebook_filepath is None:
            raise FileNotFoundError(f" - PI_Codebook.xlsx file not found in {current_year_zip_filepath}")
        else:
            initial_data_mapper = {}

            # Iterate through each sheet in the codebook
            for sheet_index, current_sheet in enumerate(sheet_names, 1):
                if current_sheet not in ['Common Values']:
                    DS_number = 'DS' + str(sheet_index).zfill(4)
                    sheet_name_with_number = f'{current_sheet} - {DS_number}'
                else:
                    sheet_name_with_number = current_sheet

                print(f'   - Starting {sheet_name_with_number}')

                # Parse the sheet data
                codebook_sheet_data = codebook_excel_object.parse(sheet_name=current_sheet)
                codebook_sheet_data.columns = [col.strip() for col in codebook_sheet_data.columns]
                codebook_sheet_data = codebook_sheet_data.ffill()

                # Define column names based on the sheet type
                if current_sheet in ['Batch Header Extract']:
                    column_labels = 'Variable Name'
                    column_description = 'Variable Label'
                    value = 'Value'
                    value_description = 'Value Labels'
                    variable_type = 'Variable Type'
                elif current_sheet in ['Common Values']:
                    column_labels = 'Value Reference'
                    column_description = 'Value Reference Description'
                    value = 'Value'
                    value_description = 'Value Label'
                    variable_type = None
                else:
                    column_labels = 'Variable Name'
                    column_description = 'Variable Label'
                    value = 'Value'
                    value_description = 'Value Label'
                    variable_type = 'Variable Type'

                column_label_groups = codebook_sheet_data.groupby(column_labels)
                sheet_mapper = {}

                # Map the variable descriptions and values
                for column_label_group, column_label_group_data in column_label_groups:
                    column_description_value = column_label_group_data[column_description].iloc[0]
                    value_mapper = column_label_group_data[[value, value_description]].set_index(value)[value_description].to_dict()

                    if current_sheet in ['Common Values']:
                        column_label_group = str(int(column_label_group))
                        variable_type_value = None
                    else:
                        variable_type_value = column_label_group_data[variable_type].iloc[0]

                    sheet_mapper[column_label_group] = {
                        'description': column_description_value,
                        'variable_type': variable_type_value,
                        'value_mapper': value_mapper,
                    }

                initial_data_mapper[sheet_name_with_number] = sheet_mapper

            clean_data_mapper = {}

            # Clean the data mapper by resolving common values references
            for sheet_name, initial_sheet_data in initial_data_mapper.items():
                if sheet_name not in ['Common Values']:
                    clean_sheet_data = {}
                    for variable, variable_data in initial_sheet_data.items():
                        variable_description = variable_data.get('description')
                        variable_type = variable_data.get('variable_type')
                        value_mapper = variable_data.get('value_mapper')

                        for value, description in value_mapper.items():
                            if 'Common Values' in description:
                                common_values_search = re.search(r'value reference (\d+)', description.lower())
                                common_values_int = common_values_search.groups()[0]
                                common_values_record = initial_data_mapper['Common Values'].get(common_values_int)
                                clean_sheet_data[variable] = {
                                    'description': variable_description,
                                    'variable_type': variable_type,
                                    'value_mapper': common_values_record.get('value_mapper'),
                                }
                            else:
                                clean_sheet_data[variable] = {
                                    'description': variable_description,
                                    'variable_type': variable_type,
                                    'value_mapper': value_mapper,
                                }

                    clean_data_mapper[sheet_name] = clean_sheet_data

            # Save the cleaned data mapper to a JSON file
            with open(json_filepath, 'w') as fout:
                json.dump(clean_data_mapper, fout)

#==================================================
# Process data in zip files, map values, and create Parquet files for each descriptor
def process_descriptor_files(filepath_mapper: Dict[str, Dict[str, str]]) -> None:
    """
    Processes data in zip files, maps values, and creates Parquet files for each descriptor.

    Parameters:
        filepath_mapper (Dict[str, Dict[str, str]]): A dictionary where keys are descriptors and values are dictionaries
            containing 'zip_filepath', 'column_mapper_filepath', and 'year'.

    Returns:
        None
    """
    
    for descriptor in filepath_mapper:
        descriptor_folder = pathlib.Path(descriptor)
        descriptor_folder.mkdir(exist_ok=True)
        
        descriptor_info = filepath_mapper.get(descriptor)
        
        zip_filepath = descriptor_info.get('zip_filepath')
        column_mapper_filepath = descriptor_info.get('column_mapper_filepath')
        year = descriptor_info.get('year')
        
        with open(column_mapper_filepath) as fin:
            column_mapper = json.load(fin)
        
        for column_mapper_key, column_mapper_key_info in column_mapper.items():
            clean_data_filepath = pathlib.Path(
                descriptor_folder,
                f"{column_mapper_key.lower().replace(' - ', ' ').replace(' ', '_')}.parquet"
            )

            if not clean_data_filepath.exists():
                print(f' - Starting {column_mapper_key}')
                DS_number = column_mapper_key.split('-')[-1]
                DS_number = DS_number.strip()
                
                with zipfile.ZipFile(zip_filepath) as czip:
                    for filepath in czip.namelist():
                        if (DS_number in filepath) and filepath.lower().endswith('data.tsv'):
                            current_data = pl.read_csv(
                                czip.read(filepath), 
                                separator='\t',
                                null_values=[' '],
                                infer_schema_length=0,
                            )
                            break

                # Strip ' (See Codebook)' from column headers
                new_columns = [col.replace(' (See Codebook)', '') for col in current_data.columns]
                current_data = current_data.rename(dict(zip(current_data.columns, new_columns)))
                current_data_original_columns = set(current_data.columns)

                for original_column_name in column_mapper_key_info:
                    if original_column_name in current_data_original_columns:
                        description = column_mapper_key_info.get(original_column_name).get('description')
                        new_column_name = f'{original_column_name}-{description}'
                        variable_type = column_mapper_key_info.get(original_column_name).get('variable_type')
                        value_mapper = column_mapper_key_info.get(original_column_name).get('value_mapper')
                        
                        current_data = current_data.with_columns(
                            pl.col(original_column_name).replace(value_mapper).alias(new_column_name)
                        ).drop(
                            original_column_name
                        )
                    
                        if variable_type == 'Numeric':
                            try:
                                current_data = current_data.with_columns(
                                    pl.col(new_column_name).cast(pl.Int32)
                                )
                            except:
                                try:
                                    current_data = current_data.with_columns(
                                        pl.col(new_column_name).cast(pl.Float32)
                                    )
                                except:
                                    pass
                        elif variable_type == 'Date':
                            try:
                                current_data = current_data.with_columns(
                                    pl.col(new_column_name).str.to_datetime(format='%d-%b-%Y')
                                )
                            except:
                                pass

                current_data.write_parquet(clean_data_filepath)
                del current_data

#==================================================
# Search for values and labels based on references in the sheet 'Common Values'
def search_common_value(
    common_val_sheet: pd.DataFrame,
    reference_col: str = 'Value Reference',
    value_col: str = 'Value',
    value_label_col: str = 'Value Label',
    reference: int = 1
) -> Dict[int, str]:
    """
    Maps values in the common values sheet to their corresponding labels based on the reference.

    Parameters:
        common_val_sheet (pd.DataFrame): DataFrame containing the common values and their labels.
        reference_col (str): The column name for references.
        value_col (str): The column name for values.
        value_label_col (str): The column name for value labels.
        reference (int): The reference number to filter by.

    Returns:
        Dict[int, str]: A dictionary mapping values to their corresponding labels.
    """

    df_filter = common_val_sheet[reference_col] == reference
    value_mapper = common_val_sheet.loc[df_filter, [value_col, value_label_col]]
    value_mapper = value_mapper.set_index(value_col)

    return value_mapper[value_label_col].to_dict()

#==================================================
# Map value using the value_mapper and common_values_sheet
def map_value(
    x: Any,
    value_mapper: Dict[Any, str],
    common_values_sheet: pd.DataFrame,
    common_value_cache: Dict[int, Dict[int, str]],
    ref_regex: re.Pattern,
    reference_col: str = 'Value Reference',
    value_col: str = 'Value',
    value_label_col: str = 'Value Label'
) -> Any:
    """
    Maps a given value using the provided value mapper and common values sheet.
    
    Parameters:
        x (Any): The value to be mapped.
        value_mapper (Dict[Any, str]): Dictionary mapping original values to their corresponding labels.
        common_values_sheet (pd.DataFrame): DataFrame containing the common values and their labels.
        common_value_cache (Dict[int, Dict[int, str]]): Cache to store common value mappings.
        ref_regex (re.Pattern): Compiled regex pattern to find 'Reference' followed by an integer.
        reference_col (str): The column name for references.
        value_col (str): The column name for values.
        value_label_col (str): The column name for value labels.
    
    Returns:
        Any: The mapped value if found, otherwise the original value.
    """

    if x in value_mapper:
        return value_mapper[x]
    
    for lbl in value_mapper.values():
        if 'Common Values' in lbl:
            match = ref_regex.search(lbl)
            if match:
                ref = int(match.group(1))
                if ref in common_value_cache:
                    common_mapper = common_value_cache[ref]
                else:
                    common_mapper = search_common_value(
                        common_values_sheet, 
                        reference_col=reference_col, 
                        value_col=value_col, 
                        value_label_col=value_label_col, 
                        reference=ref
                    )
                    common_value_cache[ref] = common_mapper
                return common_mapper.get(x, x)
    
    return x

#==================================================
# Map labels to values
def map_doc_to_code(
    path: str,
    sheet_and_sample: Dict[str, pd.DataFrame],
    valid_sheets: List[str],
    common_values_sheet_name: str = 'Common Values',
    extract: bool = False,
    zipfile_path: str = None,
    reference_col: str = 'Value Reference',
    value_col: str = 'Value',
    value_label_col: str = 'Value Label'
) -> Dict[str, pd.DataFrame]:
    """
    Maps coded values in the sample DataFrames to their corresponding labels based on the codebook.

    Parameters:
        path (str): Path to the Excel file or zip file containing the codebook.
        sheet_and_sample (Dict[str, pd.DataFrame]): Dictionary with sheet names as keys and sample DataFrames as values.
        valid_sheets (List[str]): List of valid sheet names to check against.
        common_values_sheet_name (str): Name of the sheet containing common values. Default is 'Common Values'.
        extract (bool): Flag indicating whether to extract the Excel file from a zip archive. Default is False.
        zipfile_path (str, optional): Path to the zip file if extraction is needed. Must be provided if extract is True.
        reference_col (str): The column name for references.
        value_col (str): The column name for values.
        value_label_col (str): The column name for value labels.

    Returns:
        Dict[str, pd.DataFrame]: The dictionary of sample DataFrames with coded values mapped to their corresponding labels.
    """

    # Check if all sheet names are valid
    for sheet in sheet_and_sample:
        if sheet not in valid_sheets:
            raise ValueError(f"Sheet must be one of {valid_sheets}.")

    # Extract the file if it is in a zip archive
    if extract:
        if zipfile_path is None:
            raise ValueError("zipfile_path must be provided if extract is True.")
        
        with zipfile.ZipFile(zipfile_path, 'r') as z:
            with z.open(path) as file:
                codebook = pd.read_excel(file, sheet_name=None)
    else:
        # Load the codebook directly from the provided path
        codebook = pd.read_excel(path, sheet_name=None)

    mapped_samples = {}

    # Load and forward fill the common values sheet
    common_values_sheet = codebook[common_values_sheet_name].ffill()

    # Cache to store common value mappings
    common_value_cache = {}

    # Compiled regex to find 'Reference' followed by an integer
    ref_regex = re.compile(r'Reference (\d+)')

    for sheet_name, sample in sheet_and_sample.items():
        if sheet_name not in codebook:
            raise ValueError(f"Sheet '{sheet_name}' does not exist in the codebook.")

        # Forward fill missing values in the sheet codebook
        sheet_codebook = codebook[sheet_name].ffill()

        # Get unique variable names from the codebook
        variable_names = sheet_codebook['Variable Name'].unique()

        # Find columns in the sample that match variable names
        code_to_map = sample.columns.intersection(variable_names)

        for code in code_to_map:
            if code in sample.columns:
                df_filter = sheet_codebook['Variable Name'] == code
                
                # Use appropriate value label column based on sheet type
                label = value_label_col if sheet_name != 'Batch Header Extract' else 'Value Labels'

                # Create a value mapper dictionary
                value_mapper = sheet_codebook.loc[df_filter, [value_col, label]]
                value_mapper = value_mapper.set_index(value_col)
                value_mapper = value_mapper[label].to_dict()

                # Apply the value mapping function to the sample column
                sample[code] = sample[code].apply(
                    lambda x: map_value(
                        x=x,
                        value_mapper=value_mapper,
                        common_values_sheet=common_values_sheet,
                        common_value_cache=common_value_cache,
                        ref_regex=ref_regex,
                        reference_col=reference_col,
                        value_col=value_col,
                        value_label_col=value_label_col
                    )
                )
            
        mapped_samples[sheet_name] = sample

    return mapped_samples

#==================================================
# Rename column headers in samples based on variable labels
def rename_headers(
    path: str,
    sheet_and_sample: Dict[str, pd.DataFrame],
    valid_sheets: List[str],
    extract: bool = False,
    zipfile_path: str = None
) -> Dict[str, pd.DataFrame]:
    """
    Rename column headers in samples based on variable labels.
    
    Parameters:
        path (str): The path to the codebook file.
        sheet_and_sample (Dict[str, pd.DataFrame]): Dictionary with sheet names as keys and sample DataFrames as values.
        valid_sheets (List[str]): List of valid sheet names to check against.
        extract (bool): If True, extract the codebook from a zip archive.
        zipfile_path (str): The path to the zip file, if extract is True.
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with updated sample DataFrames.
    """

    # Check if all sheet names are valid
    for sheet in sheet_and_sample:
        if sheet not in valid_sheets:
            raise ValueError(f"Sheet must be one of {valid_sheets}.")

    # Extract the file if it is in a zip archive
    if extract:
        if zipfile_path is None:
            raise ValueError("zipfile_path must be provided if extract is True.")
        
        with zipfile.ZipFile(zipfile_path, 'r') as z:
            with z.open(path) as file:
                codebook = pd.read_excel(file, sheet_name=None)
    else:
        # Load the codebook directly from the provided path
        codebook = pd.read_excel(path, sheet_name=None)

    for sheet_name, sample in sheet_and_sample.items():
        if sheet_name not in codebook:
            raise ValueError(f"Sheet '{sheet_name}' does not exist in the codebook.")

        # Forward fill missing values in the sheet codebook
        sheet_codebook = codebook[sheet_name].ffill()

        # Create a mapping from variable names to variable labels
        variable_label_map = dict(zip(sheet_codebook['Variable Name'], sheet_codebook['Variable Label']))
        new_columns = {}

        for column in sample.columns:
            if column in variable_label_map:
                # Clean and format the variable label
                variable_label = variable_label_map[column].replace("(See Codebook)", "").strip()
                variable_label = "_".join(variable_label.split()).upper()

                # Create new column name based on variable label
                new_columns[column] = re.sub(r'\d+', '', column) + "_" + variable_label if re.match(r'[A-Za-z]+\d+', column) else variable_label
            else:
                new_columns[column] = column

        # Rename columns in the sample DataFrame
        sample.rename(columns=new_columns, inplace=True)

    return sheet_and_sample

#==================================================
# Print unique values after mapping
def print_unique_values(map_dict: Dict[str, pd.DataFrame]):
    """
    Prints unique values for each column in each sheet after mapping.
    
    Parameters:
    map_dict (Dict[str, pd.DataFrame]): Dictionary of mapped DataFrames with sheet names as keys.
    """
    
    for sheet, sample in map_dict.items():
        print(f"{sheet}'s columns:")
        for col in sample.columns:
            print(f'\t{col}: {sample[col].unique()}')

#==================================================