import pandapower as pp
import pandas as pd
import pandapower.topology as top
import geopandas as gpd
import folium
import os
import base64
import io





def check_ext_grid_input(Grid: pd.DataFrame, Bus: pd.DataFrame):
    """
    Raises an error if the DataFrame's Substation shape is greater than 1  or zero
    or do not contains the column 'bus'
    or the name of the bus is not in the busses sheet
    or returns True.

    Parameters:
        df (pd.DataFrame): The DataFrame to validate.

    Raises:
        ValueError: If the DataFrame's shape is invalid (greater than 2 or zero in any dimension).
    """
    rows, _ = Grid.shape
    if rows == 0:
        raise ValueError("No Substation has been defined in 'Grid' sheet of the xlsx file")
    if rows >= 2 :
        raise ValueError("More than one Substation has been defined in 'Substation' sheet of the xlsx file")
    if rows == 1:
        if 'bus' not in Grid.columns:
            raise ValueError("No columns 'bus' is defined in 'Substation' sheet of the xlsx file")
        else:
            if Grid.loc[0,'bus'] not in Bus.name.to_list():
                raise ValueError("No Bus has been defined in 'Busses' sheet with name:"+Grid.loc[0,'bus'])


def check_busses_input(Bus: pd.DataFrame):
    """
    Raises an error if the DataFrame's Buses do not have the correct shape, or wrong columns, or variable type

    Parameters:
        Bus (pd.DataFrame): The DataFrame to validate.

    Raises:
        ValueError: If the DataFrame's shape is invalid (greater than 2 or zero in any dimension).
    """
    cols_names = ['name','vn_kv','substation','latitude','longitude']
    rows, cols = Bus.shape
    classes = {'name':str, 'vn_kv':float, 'substation':bool,'latitude':float,'longitude':float}
    classes_str = {'name': 'str', 'vn_kv': 'float', 'substation': 'bool', 'latitude': 'float', 'longitude': 'float'}
    if cols != 5:
        raise ValueError("'Busses' sheet should have 5 columns: [name,vn_kv,substation,latitude,longitude]")
    else:
        for name_col in cols_names:
            if name_col not in Bus.columns:
                raise ValueError("'Busses' sheet do not have column:"+name_col)
            if not(Bus[name_col].apply(lambda x: isinstance(x, classes[name_col])).all()):
                raise ValueError("Column "+name_col+" should contain "+str(classes_str[name_col])+" values")

def check_lines_input(Line: pd.DataFrame, Bus: pd.DataFrame):
    """
    Raises an error if the DataFrame's Buses do not have the correct shape, or wrong columns, or variable type

    Parameters:
        Bus (pd.DataFrame): The DataFrame to validate.

    Raises:
        ValueError: If the DataFrame's  is invalid.
    """
    cols_names = ['name','from_bus','to_bus','length_km','r_ohm_per_km',
                  'x_ohm_per_km','c_nf_per_km','g_us_per_km','max_i_ka']


    rows, cols = Line.shape
    classes = {'name':str, 'from_bus':str,'to_bus':str,'length_km':float,
               'r_ohm_per_km':float,'x_ohm_per_km':float, 'c_nf_per_km':float,'g_us_per_km':float,
               'max_i_ka':float}
    classes_str = {'name':'str', 'from_bus':'str','to_bus':'str','length_km':'float',
               'r_ohm_per_km':'float','x_ohm_per_km':'float', 'c_nf_per_km':'float','g_us_per_km':'float',
               'max_i_ka':'float'}
    #check, shape, format and data types
    if cols != 9:
        raise ValueError("'Lines' sheet should have 9 columns: [name, from_bus, to_bus, length_km, r_ohm_per_km,x_ohm_per_km,,c_nf_per_km,g_us_per_km,max_i_ka]")
    else:
        for name_col in cols_names:
            if name_col not in Line.columns:
                raise ValueError("'Lines' sheet do not have column:"+name_col)
            if not(Line[name_col].apply(lambda x: isinstance(x, classes[name_col])).all()):
                raise ValueError("Column "+name_col+" should contain "+str(classes_str[name_col])+" values")
    # check if buses have been declared
    for _,line in Line.iterrows():
        if line['from_bus'] not in Bus.name.to_list():
            raise ValueError("In line:"+line['name']+" 'from_bus' terminal bus:"+line['from_bus'] +" is not in Busses sheet")
        if line['to_bus'] not in Bus.name.to_list():
            raise ValueError("In line:"+line['name']+" 'to_bus' terminal bus:"+ line['to_bus']+" is not in Busses sheet")


def check_sgen_input(Sgen: pd.DataFrame, Bus: pd.DataFrame):
    """
    Raises an error if the DataFrame's Buses do not have the correct shape, or wrong columns, or variable type

    Parameters:
        Sgen (pd.DataFrame): The DataFrame to validate.

    Raises:
        ValueError: If the DataFrame's shape is invalid.
    """
    cols_names = ['bus','name','p_max_mw']
    rows, cols = Sgen.shape
    classes = {'bus':str, 'name':str, 'p_max_mw':float}
    if cols != 3:
        raise ValueError("'Generators' sheet should have 3 columns: [bus,name,p_max_mw]")
    else:
        for name_col in cols_names:
            if name_col not in Sgen.columns:
                raise ValueError("'Generators' sheet do not have column:"+name_col)
            if not(Sgen[name_col].apply(lambda x: isinstance(x, classes[name_col])).all()):
                raise ValueError("Column "+name_col+" should contain "+str(classes[name_col])+" values")
    # check if buses have been declared
    for _,sg in Sgen.iterrows():
        if sg['bus'] not in Bus.name.to_list():
            raise ValueError("In Generator:"+sg['name']+" 'bus':"+sg['bus'] +" is not in Busses sheet")

def check_excel_data_input(Busses,GRID,SGEN,Lines):
    try:
        check_busses_input(Busses)
        flag_bus = True
    except ValueError as e:
        return e
        flag_bus = False

    if flag_bus:
        try:
            check_lines_input(Lines,Busses)
            flag_lines = True
        except ValueError as e:
            return e
            flag_lines = False
        try:
            check_ext_grid_input(GRID,Busses)
            flag_grid = True
        except ValueError as e:
            return e
            flag_grid = False
        try:
            check_sgen_input(SGEN,Busses)
            flag_sgen = True
        except ValueError as e:
            return e
            flag_sgen = False
        if (flag_bus)&(flag_lines)&(flag_grid)&(flag_sgen):
            return 'success'


def generate_pandapower_net(Busses,GRID,SGEN,Lines):
    net = pp.create_empty_network(sn_mva=1)
    ##Create Ext_Grid & 1st bus
    pp.create_bus(net, name=GRID.loc[0, 'bus'],
                  vn_kv=Busses.loc[GRID.loc[0, 'bus'] == Busses.name].vn_kv.values[0],
                  geodata=(Busses.loc[GRID.loc[0, 'bus'] == Busses.name, 'latitude'].values[0],
                           Busses.loc[GRID.loc[0, 'bus'] == Busses.name, 'longitude'].values[0]))
    pp.create_ext_grid(net, name=GRID.loc[0, 'bus'], bus=0, vm_pu=1)
    ##Create lines & rest of busses. Generation from extgrid and downwards is essential to later stage
    connected_lines = Lines.index[Lines.from_bus == net.bus.loc[0, 'name']].to_list() + \
                      Lines.index[Lines.to_bus == net.bus.loc[0, 'name']].to_list()
    passed_lines = []
    passed_buses = [net.bus.loc[0, 'name']]
    while (len(passed_buses) != Busses.shape[0]) & (len(passed_lines) != Lines.shape[0]):
        for new_line in connected_lines:
            if Lines.loc[new_line, 'from_bus'] not in net.bus.name.to_list():
                pp.create_bus(net, name=Lines.loc[new_line, 'from_bus'],
                              vn_kv=Busses.loc[Lines.loc[new_line, 'from_bus'] == Busses.name].vn_kv.values[0],
                              geodata=(Busses.loc[Lines.loc[new_line, 'from_bus'] == Busses.name, 'latitude'].values[0],
                                       Busses.loc[Lines.loc[new_line, 'from_bus'] == Busses.name, 'longitude'].values[
                                           0]))
            if Lines.loc[new_line, 'to_bus'] not in net.bus.name.to_list():
                pp.create_bus(net, name=Lines.loc[new_line, 'to_bus'],
                              vn_kv=Busses.loc[Lines.loc[new_line, 'to_bus'] == Busses.name].vn_kv.values[0],
                              geodata=(Busses.loc[Lines.loc[new_line, 'to_bus'] == Busses.name, 'latitude'].values[0],
                                       Busses.loc[Lines.loc[new_line, 'to_bus'] == Busses.name, 'longitude'].values[0]))
            pp.create_line_from_parameters(net,
                                           from_bus=
                                           net.bus.index[Lines.loc[new_line, 'from_bus'] == net.bus.name].values[0],
                                           to_bus=net.bus.index[Lines.loc[new_line, 'to_bus'] == net.bus.name].values[
                                               0],
                                           r_ohm_per_km=Lines.loc[new_line, 'r_ohm_per_km'],
                                           x_ohm_per_km=Lines.loc[new_line, 'x_ohm_per_km'],
                                           max_i_ka=Lines.loc[new_line, 'max_i_ka'],
                                           g_us_per_km=Lines.loc[new_line, 'g_us_per_km'],
                                           c_nf_per_km=Lines.loc[new_line, 'c_nf_per_km'],
                                           length_km=Lines.loc[new_line, 'length_km'],
                                           name=Lines.loc[new_line, 'name'])
        passed_lines = passed_lines + connected_lines
        # not passed busses
        pending_buses = [bus for bus in net.bus.name if bus not in passed_buses]
        if len(pending_buses) >= 1:
            connected_lines = Lines.index[Lines.from_bus == pending_buses[0]].to_list() + \
                              Lines.index[Lines.to_bus == pending_buses[0]].to_list()
            connected_lines = [line for line in connected_lines if line not in passed_lines]
            passed_buses.append(pending_buses[0])
    ###Create Loads
    for id, bus in Busses.iterrows():
        if (bus['substation']) & (bus['name'] not in net.ext_grid.name.to_list()):
            pp.create_load(net, p_mw=0, q_mvar=0, name=bus['name'],
                           bus=net.bus.index[net.bus.name == bus['name']].values[0])
    ###Create SGEN
    for id, gen in SGEN.iterrows():
        pp.create_sgen(net, p_mw=0, q_mvar=0, max_p_mw=gen['p_max_mw'],
                       name=gen['name'], bus=net.bus.index[net.bus.name == gen['bus']].values[0])
    return net

def check_file_structure(filename):
    try:
        filename.seek(0)
        Busses = pd.read_excel(filename, sheet_name='Busses')
        bus_sheet_flag = True
    except ValueError as e:
        return e
        bus_sheet_flag=False

    try:
        filename.seek(0)
        GRID = pd.read_excel(filename, sheet_name='Substation')
        substation_sheet_flag = True
    except ValueError as e:
        return e
        substation_sheet_flag = False

    try:
        filename.seek(0)
        SGEN = pd.read_excel(filename, sheet_name='Generators')
        SGEN_sheet_flag = True
    except ValueError as e:
        return e
        SGEN_sheet_flag = False

    try:
        filename.seek(0)
        Lines = pd.read_excel(filename, sheet_name='Lines')
        Lines_sheet_flag = True
    except ValueError as e:
        return e
        Lines_sheet_flag = False

    if (Lines_sheet_flag)&(SGEN_sheet_flag)&(substation_sheet_flag)&(bus_sheet_flag):
        return 'success'


def main_code_planning_settings_topology(filename):
    msg = check_file_structure(filename)
    if msg == 'success':
        print(msg)
        Busses = pd.read_excel(filename, sheet_name='Busses')
        GRID = pd.read_excel(filename, sheet_name='Substation')
        SGEN = pd.read_excel(filename, sheet_name='Generators')
        Lines = pd.read_excel(filename, sheet_name='Lines')
        msg = check_excel_data_input(Busses, GRID, SGEN, Lines)
        if msg == 'success':
            net = generate_pandapower_net(Busses, GRID, SGEN, Lines)
            return net,'Network Topology File is correct'
        else:
            return None, msg
    else:
        return None, msg




#net, msg = main_code_planning_settings_topology('Spain_Data2.xlsx')
#print('a')
#pp.to_json(net, "net.json")

def check_if_loops_exists(netx):
    top.determine_stubs(netx, roots = [0])
    return netx


def generate_diagram(netx):

        ##### Create a GeoDataFrame for buses
        gdf_buses = gpd.GeoDataFrame(netx.bus_geodata.index,
                                         geometry=gpd.points_from_xy(netx.bus_geodata.x, netx.bus_geodata.y),
                                         crs="EPSG:4326")

        # Initialize a folium map centered around the network's mean coordinates
        m = folium.Map(location=[netx.bus_geodata.x.mean(), netx.bus_geodata.y.mean()], zoom_start=17)
        for _, row in gdf_buses.iterrows():
            name = netx.bus.loc[row[0], 'name']
            name = name
            popup = folium.Popup(f'<b style="font-size:16px;">{name}</b>', max_width=200)
            if row[0] in netx.ext_grid.bus.to_list():
                icon = folium.CustomIcon(
                    icon_image=r"icons\TF_main.png",
                    icon_size=(50, 50),
                )
                folium.Marker(location=[row.geometry.x, row.geometry.y],
                                      popup=popup, icon=icon).add_to(m)
            else:
                icon = folium.CustomIcon(
                    icon_image=r"icons\TF_Grey.png",
                    icon_size=(20, 20),
                )
                folium.Marker(location=[row.geometry.x, row.geometry.y],
                              popup=popup, icon=icon).add_to(m)
            ##Add Lines
            for it, row in netx.line.iterrows():
                if row.in_service:
                    line_coordinates = [
                        [netx.bus_geodata.loc[row['from_bus'], 'x'], netx.bus_geodata.loc[row['from_bus'], 'y']],
                        [netx.bus_geodata.loc[row['to_bus'], 'x'], netx.bus_geodata.loc[row['to_bus'], 'y']]]

                    name = row['name']
                    popup = folium.Popup(f'<b style="font-size:16px;">{name}</b>', max_width=200)
                    color = 'blue'
                    # Create a PolyLine object with the specified geometry
                    folium.PolyLine(
                        locations=line_coordinates,  # Pass the list of coordinates
                        color=color,  # Line color
                        weight=5,  # Line thickness
                        opacity=0.7,  # Line transparency
                        popup=popup
                    ).add_to(m)
                else:
                    line_coordinates = [
                        [netx.bus_geodata.loc[row['from_bus'], 'x'], netx.bus_geodata.loc[row['from_bus'], 'y']],
                        [netx.bus_geodata.loc[row['to_bus'], 'x'], netx.bus_geodata.loc[row['to_bus'], 'y']]]

                    name = row['name'] + ' is considered out of service'
                    popup = folium.Popup(f'<b style="font-size:16px;">{name}</b>', max_width=200)
                    # Create a PolyLine object with the specified geometry
                    folium.PolyLine(
                        locations=line_coordinates,  # Pass the list of coordinates
                        color='grey',  # Line color
                        weight=5,  # Line thickness
                        opacity=0.7,  # Line transparency
                        dash_array='5, 10',
                        popup=popup
                    ).add_to(m)

            directory = "Maps/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            m.save("Maps/network_map.html")
        return 0
