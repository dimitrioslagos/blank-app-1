import pandas as pd
import pandapower as pp
import ast
import folium
import requests
import configparser
from pf_toolbox import run_pfs
import geopandas as gpd
import webbrowser



def read_config(filename='settings.cfg'):
    config = configparser.ConfigParser()
    config.read(filename)
    settings = {}
    for section in config.sections():
        for option in config.options(section):
            settings[option] = config.get(section, option)
    return settings


def get_pv_power_curves(settings_file_name, geodata_file):
    settings = read_config(filename=settings_file_name)
    locations_user = ast.literal_eval(settings['pv_locations'])
    powers_user = ast.literal_eval(settings['pv_powers'])
    geodata = pd.read_csv(geodata_file, delimiter=';')
    Power_curve = pd.DataFrame(columns=locations_user)

    for loc in locations_user:
        id = geodata.ID == locations_user[0]
        if id.sum() == 0:
            print(loc + " is not valid secondary substation name. PV not added")
            continue
        # Define the parameters
        latitude = geodata.loc[id, 'LAT'].values[0]
        longitude = geodata.loc[id, 'LON'].values[0]
        startyear = 2019
        endyear = 2019
        optimalinclination = 1
        outputformat = 'json'
        pvtechchoice = 'crystSi'
        peakpower = powers_user[locations_user.index(loc)]
        loss = 5
        pvcalculation = 1

        # Construct the API request URL
        url = f"https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat={latitude}&lon={longitude}&startyear={startyear}&pvcalculation={pvcalculation}&endyear={endyear}&optimalinclination={optimalinclination}&outputformat={outputformat}&pvtechchoice={pvtechchoice}&peakpower={peakpower}&loss={loss}"

        # Make the API request
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            # Extract and print the hourly PV production data
            hourly_data = data['outputs']['hourly']
            Power_curve[loc] = pd.DataFrame(hourly_data)['P'] / 1e6  # W to MW
    return Power_curve


def get_user_pv(settings_file_name):
    settings = read_config(filename=settings_file_name)
    locations_user = ast.literal_eval(settings['pv_locations'])
    years_user = ast.literal_eval(settings['pv_installation_year'])
    PV = pd.DataFrame(columns=locations_user)
    PV.loc[0, :] = years_user
    return PV


def generate_pp_net(xlsx_filename, settings_file):
    data_lines_mv = pd.read_excel(xlsx_filename, sheet_name='Lines')
    buses = pd.read_excel(xlsx_filename, sheet_name='Busses', index_col=0)
    grid = pd.read_excel(xlsx_filename, sheet_name='Substation')
    sgen = pd.read_excel(xlsx_filename, sheet_name='Generators')
    net = pp.create_empty_network()

    # Bus Data##
    net.bus.vn_kv = buses.vn_kv
    net.bus.in_service = True
    net.bus.name = buses.CT
    net.bus.index = buses.index

    net.bus_geodata.x = buses.Lat
    net.bus_geodata.y = buses.Lon
    net.bus_geodata.index = buses.index

    # Substation data
    for i in range(grid.shape[0]):
        pp.create_ext_grid(net=net, bus=buses.index[buses.name == grid['bus_name'].values[i]][0], vm_pu=1,
                           name=buses.loc[grid['bus_name'].values[i] == buses['name'], 'CT'].values[0])
    # Line Data
    net.line.name = data_lines_mv.name
    net.line.from_bus = data_lines_mv.from_bus
    net.line.to_bus = data_lines_mv.to_bus
    net.line.length_km = data_lines_mv.length_km
    net.line.r_ohm_per_km = data_lines_mv.r_ohm_per_km
    net.line.x_ohm_per_km = data_lines_mv.x_ohm_per_km
    net.line.c_nf_per_km = data_lines_mv.c_nf_per_km
    net.line.g_us_per_km = data_lines_mv.g_us_per_km
    net.line.max_i_ka = data_lines_mv.max_i_ka
    net.line.in_service = True
    net.line.type = data_lines_mv.type
    net.line.parallel = data_lines_mv.parallel

    # Identify the bus connected to the external grid
    ext_grid_bus = net.ext_grid.bus.iloc[0]
    # Create a new index for the buses
    old_bus_indices = net.bus.index.tolist()
    new_bus_indices = {old: new for new, old in
                       enumerate(sorted(old_bus_indices, key=lambda x: x == ext_grid_bus, reverse=True))}
    # Reindex the buses
    net.bus.index = [new_bus_indices[idx] for idx in net.bus.index]
    net.bus_geodata.index = [new_bus_indices[idx] for idx in net.bus_geodata.index]
    net.bus = net.bus.sort_index(ascending=True)
    # Update from_bus and to_bus in the lines DataFrame
    net.line['from_bus'] = net.line['from_bus'].map(new_bus_indices)
    net.line['to_bus'] = net.line['to_bus'].map(new_bus_indices)
    net.ext_grid.bus = 0
    # Load Data
    for i in net.bus.index:
        if (sum(net.ext_grid['bus'] == i) >= 1) | (net.bus['name'].str.contains('Fict')[i]):
            continue
        else:
            pp.create_load(net, i, p_mw=0.1, q_mvar=0.05, name=net.bus.loc[i, 'name'])

    ##Add RES, either by topology file or user defined

    if sgen.shape[0] >= 1:
        for i in range(sgen.shape[0]):
            pp.create_sgen(net=net, bus=net.bus.index[net.bus.name == sgen.loc[i, 'Substation']][0], p_mw=0)

    ##
    PVs = get_user_pv(settings_file)
    settings = read_config(filename=settings_file)
    T = ast.literal_eval(settings['horizon'])
    networks = {i: [] for i in range(T)}
    for i in range(T):
        if (PVs <= i).sum().sum() >= 1:
            net_copy = net.deepcopy()
            for loc in PVs.columns:
                if PVs.loc[0, loc] <= i:
                    pp.create_sgen(net=net_copy, bus=net.bus.index[net.bus.name == loc][0], p_mw=0)
            networks[i] = net_copy
        else:
            networks[i] = net
    return networks
    ###Add generators####


def plot_network_with_lf_res(netx, year_results):
    year = len(netx) - 1
    ##### Create a GeoDataFrame for buses
    gdf_buses = gpd.GeoDataFrame(netx[0].bus_geodata.index,
                                 geometry=gpd.points_from_xy(netx[0].bus_geodata.x, netx[0].bus_geodata.y),
                                 crs="EPSG:4326")

    # Initialize a folium map centered around the network's mean coordinates
    m = folium.Map(location=[netx[0].bus_geodata.x.mean(), netx[0].bus_geodata.y.mean()], zoom_start=17)
    for _, row in gdf_buses.iterrows():
        name = networks[0].bus.loc[row[0], 'name']
        maxV = year_results[year]['v'][:, row[0]].max().round(3)
        minV = year_results[year]['v'][:, row[0]].min().round(3)
        avegV = year_results[year]['v'][:, row[0]].mean().round(3)
        name = name + '<br>' + 'Average Voltage:' + str(avegV) + '<br>' + 'Minimum Voltage:' + str(minV) + \
               '<br>' + 'Maximum Voltage:' + str(maxV)
        popup = folium.Popup(f'<b style="font-size:16px;">{name}</b>', max_width=200)
        if (avegV <= 1.04) & (avegV >= 0.96):
            folium.Marker(location=[row.geometry.x, row.geometry.y],
                          popup=popup, icon=folium.Icon(color='green')).add_to(m)
        if (avegV <= 1.08) & (avegV >= 1.04):
            folium.Marker(location=[row.geometry.x, row.geometry.y], radius=5,
                          popup=popup, icon=folium.Icon(color='orange')).add_to(m)
        if (avegV >= 1.08):
            folium.Marker(location=[row.geometry.x, row.geometry.y], radius=5,
                          popup=popup, icon=folium.Icon(color='red')).add_to(m)
        if (avegV <= 0.96) & (avegV >= 0.92):
            folium.Marker(location=[row.geometry.x, row.geometry.y], radius=5,
                          popup=popup, icon=folium.Icon(color='lightblue')).add_to(m)
        if (avegV < 0.92):
            folium.Marker(location=[row.geometry.x, row.geometry.y], radius=5,
                          popup=popup, icon=folium.Icon(color='blue')).add_to(m)
    ##Add Lines
    netx[len(netx) - 1].line.in_service = True
    netx[len(netx) - 1].line.loc[year_results[len(netx) - 1]['outaged_line'], 'in_service'] = False
    for it, row in netx[len(netx) - 1].line.iterrows():
        if row.in_service:
            line_coordinates = [
                [netx[0].bus_geodata.loc[row['from_bus'], 'x'], netx[0].bus_geodata.loc[row['from_bus'], 'y']],
                [netx[0].bus_geodata.loc[row['to_bus'], 'x'], netx[0].bus_geodata.loc[row['to_bus'], 'y']]]
            if it < year_results[len(netx) - 1]['outaged_line']:
                maxL = year_results[year]['loading'][:, it].max().round(1)
                minL = year_results[year]['loading'][:, it].min().round(1)
                avegL = year_results[year]['loading'][:, it].mean().round(1)
            else:
                maxL = year_results[year]['loading'][:, it - 1].max().round(1)
                minL = year_results[year]['loading'][:, it - 1].min().round(1)
                avegL = year_results[year]['loading'][:, it - 1].mean().round(1)
            name = row['name'] + '<br>' + 'Average Loading:' + str(avegL) + '<br>' + 'Minimum Loading:' + str(minL) + \
                   '<br>' + 'Maximum Loading:' + str(maxL)
            popup = folium.Popup(f'<b style="font-size:16px;">{name}</b>', max_width=200)
            # coloring
            if maxL <= 25:
                color = 'blue'
            if (maxL > 25) & (maxL <= 50):
                color = 'darkblue'
            if (maxL > 50) & (maxL <= 75):
                color = 'yellow'
            if (maxL > 75) & (maxL <= 100):
                color = 'red'
            if maxL > 100:
                color = 'darkred'
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
                [netx[0].bus_geodata.loc[row['from_bus'], 'x'], netx[0].bus_geodata.loc[row['from_bus'], 'y']],
                [netx[0].bus_geodata.loc[row['to_bus'], 'x'], netx[0].bus_geodata.loc[row['to_bus'], 'y']]]

            name = row['name'] + ' is considered out of service'
            popup = folium.Popup(f'<b style="font-size:16px;">{name}</b>', max_width=200)
            # Create a PolyLine object with the specified geometry
            folium.PolyLine(
                locations=line_coordinates,  # Pass the list of coordinates
                color='blue',  # Line color
                weight=5,  # Line thickness
                opacity=0.7,  # Line transparency
                dash_array='5, 10',
                popup=popup
            ).add_to(m)
    m.save("pandapower_network_map.html")
    webbrowser.open("pandapower_network_map.html")
    return 0


##Generate Networks
#networks = generate_pp_net(xlsx_filename='Spain_Data.xlsx', settings_file='settings_spain.cfg')

# settings = read_config(filename='settings_spain.cfg')
# Horizon = ast.literal_eval(settings['horizon'])
# ##Generate yearly curves
# # PV
# PVs = get_pv_power_curves(settings_file_name='settings_spain.cfg', geodata_file='topology_substation.csv')
# # Load
# P = pd.read_csv('P.csv', index_col=0)
# cosphi = pd.read_csv('coshpi.csv', index_col=0)['0']
# P.index = range(8760)
#
# load_factor = ast.literal_eval(settings['load_groth_rate'])
#
# year_results = run_pfs(networks=networks, T=Horizon, cosphi=cosphi, Pl=P, Ppv=PVs)
# plot_network_with_lf_res(networks, year_results)






