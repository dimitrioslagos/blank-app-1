import pandas as pd


def check_P_input(load_names, active_power_file):
    """
    Raises an error if the excel File contain less columns than the loads defined in pandapower,
    or if a name of a load is not mentioned

    Parameters:
        load_names (list): List of load names.

    Raises:
        ValueError: Describing the error.
    """
    active_power_file.seek(0)
    P_curve = pd.read_csv(active_power_file, index_col=0)
    print(P_curve.shape)
    if P_curve.shape[1]==0:
        active_power_file.seek(0)
        P_curve = pd.read_csv(active_power_file, index_col=0, sep=';')
        print(P_curve.shape)
    if P_curve.shape[0]!=8760:
        raise ValueError("Power Curves File should containt 8760 values per substation")

    missing_Load_Curves = [load for load in load_names if load not in P_curve.columns]
    if len(missing_Load_Curves)>=1:
        missing_substations=''
        for sub in missing_Load_Curves:
            missing_substations = missing_substations + ', '+sub
        raise ValueError("Missing Substations from .csv file:"+missing_substations)

    for col in P_curve.columns:
        if P_curve.loc[:,col].isna().any():
            raise ValueError("Substations:"+col+' contains missing values')
        if not(P_curve.loc[:,col].apply(lambda x: isinstance(x, float)).all()):
            raise ValueError("Substations:"+col+' contains values that are not numeric (float)')
    return P_curve


def check_cosphi_input(load_names, cosphi_file):
    """
    Raises an error if the excel File contain less columns than the loads defined in pandapower,
    or if a name of a load is not mentioned

    Parameters:
        load_names (list): List of load names.

    Raises:
        ValueError: Describing the error.
    """
    cosphi_file.seek(0)
    cosphi = pd.read_csv(cosphi_file, index_col=0)
    if cosphi.shape[1]==0:
        cosphi_file.seek(0)
        cosphi = pd.read_csv(cosphi_file, index_col=0, sep=';')
    if cosphi.shape[0]!=len(load_names):
        raise ValueError("Cosphi File should as many lines as the loads defined")

    missing_Load_Curves = [load for load in load_names if load not in cosphi.index]
    if len(missing_Load_Curves)>=1:
        missing_substations=''
        for sub in missing_Load_Curves:
            missing_substations = missing_substations + ', '+sub
        raise ValueError("Missing Substations from .csv file:"+missing_substations)

    for col in cosphi.index:
        if cosphi.loc[col].isna().any():
            raise ValueError("Substations:"+col+' contains missing values')
        if not(cosphi.loc[col].apply(lambda x: isinstance(x, float)).all()):
            raise ValueError("Substations:"+col+' contains values that are not numeric (float)')
    return cosphi


def check_P_file(load,filename):
    try:
         P_curve=check_P_input(load, filename)
         return '.csv file has the correct format',P_curve
    except ValueError as e:
        return e, None

def check_cosphi_file(load,filename):
    try:
         cosphi=check_cosphi_input(load, filename)
         return '.csv file has the correct format',cosphi
    except ValueError as e:
        return e, None
