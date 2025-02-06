import pandas as pd

def check_equipment_input(line_types_file):
    """
    Raises an error if the csv File contain less columns than the expected,
    or if the data have incorrect format

    Parameters:
        line_types_file (list): csv file.

    Raises:
        ValueError: Describing the error.
    """
    line_types_file.seek(0)
    Line_types = pd.read_csv(line_types_file)
    if Line_types.shape[1]!=6:
        line_types_file.seek(0)
        Line_types = pd.read_csv(line_types_file, sep=';')


    expected_columns_names = ['Name','r_ohm_per_km','x_ohm_per_km','max_i_ka','cost_per_km_€','type']

    missing_columns_Names = [col_name for col_name in expected_columns_names if col_name not in Line_types.columns]
    if len(missing_columns_Names)>=1:
        missing_data=''
        for col_name in missing_columns_Names:
            missing_data = missing_data + ', '+col_name
        raise ValueError("Missing Categories from .csv file:"+missing_data)
    col_classes = {'Name':str,'r_ohm_per_km':float,'x_ohm_per_km':float,'max_i_ka':float,'cost_per_km_€':float,'type':str}
    print(Line_types)
    for col in Line_types.columns:
        print(not(Line_types.loc[:,col].apply(lambda x: isinstance(x, col_classes[col])).all()))
        if Line_types.loc[:,col].isna().any():
            raise ValueError("Category:"+col+' contains missing values')
        if not(Line_types.loc[:,col].apply(lambda x: isinstance(x, col_classes[col])).all()):
            raise ValueError("Column "+col+" should contain "+str(col_classes[col])+" values")
    # Check if the column contains only 'OH' or 'UG'
    if not(Line_types['type'].isin(['OH', 'UG']).all()):
        raise ValueError("Column Type should contain only 'OH' or 'UG' as values")
    return Line_types

def check_equipment_file(filename):
    try:
         Line_types=check_equipment_input(filename)
         return '.csv file has the correct format',Line_types
    except ValueError as e:
        return e, None
