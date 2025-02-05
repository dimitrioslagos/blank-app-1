import pandas as pd
from pandapower.plotting.plotly import simple_plotly
import os
import pandapower as pp
import pandapower.converter as pc
import folium
import geopandas as gpd
import webbrowser


from pandapower.converter import from_cim
import numpy as np
import pandapower.plotting.plotly as plotly
from pandapower.plotting.generic_geodata import create_generic_coordinates
import pyproj

def plot_in_map(netx):
    ##### Create a GeoDataFrame for buses
    # Assuming netx.bus_geodata contains your 'x' and 'y' columns
    min_lon, max_lon = 21.0, 24.5  # in degrees
    min_lat, max_lat = 38.0, 42.5  # in degrees

    # EPSG:2100 bounds (example values; replace with actual bounds of your region)
    min_x, max_x = 0, 297 # in meters
    min_y, max_y = 0, 210  # in meters

    # Calculate scaling factors and offsets
    scale_x = 1#(max_lon - min_lon) / (max_x - min_x)
    offset_x = min_lon - (min_x * scale_x)

    scale_y = 1#(max_lat - min_lat) / (max_y - min_y)
    offset_y = min_lat - (min_y * scale_y)

    # Function to transform EPSG:2100 (x, y) to WGS-84 (lon, lat)
    def transform_to_wgs84(x, y):
        lon = x * scale_x + offset_x
        lat = y * scale_y + offset_y
        return lon, lat



    # Transform to WGS-84
    lon, lat = transform_to_wgs84(netx.bus_geodata['x'].values, netx.bus_geodata['y'].values)

    # Transform coordinates from EPSG:2100 to EPSG:4326
    proj_transformer = pyproj.Transformer.from_crs("EPSG:2100", "EPSG:4326", always_xy=True)

    netx.bus_geodata['lon'] = lon
    netx.bus_geodata['lat'] = lat

    # Apply transformation to all points in the bus_geodata
    # Create a GeoDataFrame in EPSG:4326 for Folium
    gdf_buses = gpd.GeoDataFrame(
        netx.bus_geodata.index,
        geometry=gpd.points_from_xy(netx.bus_geodata['lon'], netx.bus_geodata['lat']),
        crs="EPSG:4326"
    )

    # Initialize a folium map centered around the network's mean coordinates
    m = folium.Map(
        location=[netx.bus_geodata['lat'].mean(), netx.bus_geodata['lon'].mean()],
        zoom_start=17
    )
    # Initialize a folium map centered around the network's mean coordinates
    m.save("pandapower_network_map.html")
    webbrowser.open("pandapower_network_map.html")
    return 0


def generate_plot(netx):
    lc = plotly.create_line_trace(netx, infofunc='length='+netx.line.length_km.round(3).astype('str'),use_line_geodata=False)
    bc1 = plotly.create_bus_trace(netx, netx.bus.index, size=10, color="blue", infofunc=netx.bus.name,trace_name='bus')
    bc2 = plotly.create_bus_trace(netx, netx.ext_grid.bus.values, size=10, color="yellow", infofunc=netx.bus.name,trace_name='ext_grid')
    bc3 = plotly.create_bus_trace(netx, netx.load.bus.unique(), size=10, color="green", infofunc=netx.bus.name,trace_name='load')
    plotly.draw_traces(bc1+lc+bc2+bc3, figsize=1, aspectratio=(8, 6))
    return 0


def get_terminals(netx):
    terminals = []
    for bus in netx.bus.index:
        connected_elements = (netx.line.from_bus == bus).sum() + (netx.line.to_bus == bus).sum() \
                             + (netx.switch.bus == bus).sum() + (netx.switch.element == bus).sum() \
                             + (netx.load.bus == bus).sum() + (netx.ext_grid.bus == bus).sum()
        if connected_elements == 1:
            terminals.append(bus)
    return terminals

def remove_empty_terminals(netx,terminals):
    for term in terminals:
        elements_types = ['line','switch','load']
        for element in elements_types:
            if element is 'line':
                el = netx.line.index[(netx.line.from_bus == term) | (netx.line.to_bus == term)].to_list()
                netx.line.drop(index=el,inplace=True)
            elif element is 'switch':
                el = netx.switch.index[(netx.switch.bus == term) | (netx.switch.element == term)].to_list()
                netx.switch.drop(index=el,inplace=True)
            else:
                el = netx.load.index[(netx.load.bus == term)].to_list()
                netx.load.drop(index=el,inplace=True)
    netx.bus.drop(index=terminals,inplace=True)
    return netx



net = pp.from_json("TEST_v3.json")
#plot_in_map(net)

hv_bus = net.bus[net.bus.vn_kv==150]
connected_lines = []
connected_trafo_lv_bus = []
for slack in hv_bus.index:
    connected_lines = connected_lines + net.line[net.line.from_bus == slack].index.tolist() + \
                      net.line[net.line.to_bus == slack].index.tolist()
    connected_trafo_lv_bus= connected_trafo_lv_bus+ net.trafo[net.trafo.hv_bus == slack].lv_bus.tolist()

slack_name = 'GND_33997471'
slack = net.bus.loc[net.bus.name == slack_name].index[0]

net2 = pp.create_empty_network()
connected_lines = net.line[net.line.from_bus == slack].index.tolist() + net.line[net.line.to_bus == slack].index.tolist()


passed_bus = []
pp.create_bus(net2, vn_kv=net.bus.loc[slack, 'vn_kv'], name=net.bus.loc[slack, 'name'])


bus_lv_load = net.bus.loc[net.load.bus].index[(net.bus.loc[net.load.bus].vn_kv-0.4).abs()<=1e-3]
for bus_l in bus_lv_load:
    if np.abs(net.bus.loc[bus_l,'vn_kv']-0.4)<=1e-3:
        if not (net.switch.loc[net.switch.bus==bus_l].empty):
            bul1 = net.switch.loc[net.switch.bus==bus_l,'element'].values[0]
            if not(net.trafo.loc[net.trafo.lv_bus==bul1].empty):
                for _, trafo in net.trafo.loc[net.trafo.lv_bus==bul1].iterrows():
                    pp.create_load(net,bus=trafo.hv_bus,p_mw=0,q_mvar=0, name=net.load.loc[net.load.bus==bus_l,'name'].values[0],sn_mva=trafo.sn_mva)
                    net.load.drop(index=net.load.index[net.load.bus==bus_l], inplace=True)
print(net.bus.shape)
print(net.load.shape)

bus_id = slack
connected_lines = net.line[net.line.from_bus == slack].index.tolist() \
                  + net.line[net.line.to_bus == slack].index.tolist()
connected_switches = [] if len(connected_lines)>=1 \
    else net.switch[net.switch.element == bus_id].index.to_list() \
         + net.switch[net.switch.bus == bus_id].index.to_list()

while len(connected_lines+connected_switches)>=1:
    for line in connected_lines:
        if not(net.line.loc[line,'name'] in net2.line.name.to_list()):
            if not(net.bus.loc[net.line.loc[line,'to_bus'],'name'] in net2.bus.name.to_list()):
                pp.create_bus(net2, vn_kv=net.bus.loc[net.line.loc[line,'to_bus'], 'vn_kv'],
                              name=net.bus.loc[net.line.loc[line,'to_bus'], 'name'])
                bus_t = net.line.loc[line, 'to_bus'] if bus_id!=net.line.loc[line, 'to_bus'] else []
                bus_f = bus_id
            if not (net.bus.loc[net.line.loc[line, 'from_bus'], 'name'] in net2.bus.name.to_list()):
                pp.create_bus(net2, vn_kv=net.bus.loc[net.line.loc[line, 'from_bus'], 'vn_kv'],
                              name=net.bus.loc[net.line.loc[line, 'from_bus'], 'name'])
                bus_f = net.line.loc[line, 'from_bus'] if bus_id != net.line.loc[line, 'from_bus'] else []
                bus_t = bus_id
            flag_loop = (net.bus.loc[net.line.loc[line, 'from_bus'], 'name'] in net2.bus.name.to_list()) & (net.bus.loc[net.line.loc[line,'to_bus'],'name'] in net2.bus.name.to_list())
            if flag_loop:
                bus_f = net.line.loc[line,'from_bus']
                bus_t = net.line.loc[line, 'to_bus']
            ###Create Lines
            pp.create_line_from_parameters(net2,from_bus=net2.bus[net2.bus.name==net.bus.loc[bus_f, 'name']].index[0],
                                               to_bus=net2.bus[net2.bus.name==net.bus.loc[bus_t, 'name']].index[0],
                                               length_km=net.line.loc[line,'length_km'],
                               name=net.line.loc[line,'name'],r_ohm_per_km=net.line.loc[line,'r_ohm_per_km'],
                               x_ohm_per_km=net.line.loc[line,'x_ohm_per_km'],max_i_ka=net.line.loc[line,'max_i_ka'],
                                               c_nf_per_km=net.line.loc[line,'c_nf_per_km'])

    for sw in connected_switches:
        if not (net.switch.loc[sw, 'name'] in net2.switch.name.to_list()):
            if not (net.bus.loc[net.switch.loc[sw, 'bus'], 'name'] in net2.bus.name.to_list()):
                pp.create_bus(net2, vn_kv=net.bus.loc[net.switch.loc[sw, 'bus'], 'vn_kv'],
                                  name=net.bus.loc[net.switch.loc[sw, 'bus'], 'name'])
                bus_n = net.switch.loc[sw, 'bus'] if bus_id != net.switch.loc[sw, 'bus'] else []
                element_n = bus_id
            if not (net.bus.loc[net.switch.loc[sw, 'element'], 'name'] in net2.bus.name.to_list()):
                pp.create_bus(net2, vn_kv=net.bus.loc[net.switch.loc[sw, 'element'], 'vn_kv'],
                              name=net.bus.loc[net.switch.loc[sw, 'element'], 'name'])
                element_n = net.switch.loc[sw, 'element'] if bus_id != net.switch.loc[sw, 'element'] else []
                bus_n = bus_id
                ###Create Lines
            flag_loop = (net.bus.loc[net.switch.loc[sw, 'bus'], 'name'] in net2.bus.name.to_list()) & (net.bus.loc[net.switch.loc[sw,'element'],'name'] in net2.bus.name.to_list())
            if flag_loop:
                element_n = net.switch.loc[sw, 'element']
                bus_n = net.switch.loc[sw, 'bus']
            pp.create_switch(net2,bus=net2.bus[net2.bus.name == net.bus.loc[bus_n, 'name']].index[0],
                                               element=net2.bus[net2.bus.name == net.bus.loc[element_n, 'name']].index[0],
                                               name=net.switch.loc[sw, 'name'],
                                               et=net.switch.loc[sw, 'et'])
    passed_bus.append(net2.bus.index[net2.bus.name==net.bus.loc[bus_id, 'name']][0])
    if len([k for k in net2.bus.index.to_list() if k not in passed_bus])>=1:
        b = [k for k in net2.bus.index.to_list() if k not in passed_bus][0]
        bus_id = net.bus.loc[net.bus.name==net2.bus.loc[b,'name']].index[0]
        connected_lines = net.line[net.line.from_bus == bus_id].index.tolist() \
                          + net.line[net.line.to_bus == bus_id].index.tolist()
        connected_switches = net.switch[net.switch.element == bus_id].index.to_list() \
                 + net.switch[net.switch.bus == bus_id].index.to_list()
    else:
        connected_lines = []
        connected_switches = []
for load in net.load.index:
    pp.create_load(net2,bus=net2.bus.index[net2.bus.name==net.bus.loc[net.load.loc[load,'bus'],'name']].values[0],
                   name=net.load.loc[load,'name'],
                   sn_mva= net.load.loc[load,'sn_mva'],
                   p_mw=0,
                   q_mvar=0)
pp.create_ext_grid(net2,bus=0,vm_pu=1)
create_generic_coordinates(net2, respect_switches=True, overwrite=True)
print(net2.bus.shape)
print(net2.load.shape)
generate_plot(net2)

net22 = net2.deepcopy()
terminals = get_terminals(net22)
while len(terminals)>=1:
    print(terminals)
    net22 = remove_empty_terminals(net22, terminals)
    terminals = get_terminals(net22)
print(net22.bus.shape)
print(net22.load.shape)
generate_plot(net22)


net3 = net22.deepcopy()
for id,sw in net3.switch.iterrows():
    bus_to_delete = sw.element
    bus_new_val = sw.bus
    net3.line.loc[net3.line.from_bus==bus_to_delete,'from_bus']=bus_new_val
    net3.line.loc[net3.line.to_bus == bus_to_delete, 'to_bus'] = bus_new_val
    net3.switch.loc[net3.switch.bus == bus_to_delete, 'bus'] = bus_new_val
    net3.switch.loc[net3.switch.element == bus_to_delete, 'element'] = bus_new_val
    net3.load.loc[net3.load.bus == bus_to_delete,'bus'] = bus_new_val
    if (net3.bus.index==bus_to_delete).sum()==1:
        net3.bus.drop(index=[bus_to_delete],inplace=True)
    net3.switch.drop(index=[id],inplace=True)

print(net3.bus.shape)
print(net3.load.shape)
generate_plot(net3)

net4 = net3.deepcopy()
#net4.line.loc[net3.line.from_bus>net3.line.to_bus,'from_bus']=net3.line.loc[net3.line.from_bus>net3.line.to_bus,'to_bus']
#net4.line.loc[net3.line.from_bus>net3.line.to_bus,'to_bus']=net3.line.loc[net3.line.from_bus>net3.line.to_bus,'from_bus']
for id in net4.line.index:
    end_bus = net4.line.loc[id,'to_bus']
    start_bus = net4.line.loc[id,'from_bus']

    length = net4.line.loc[id,'length_km']
    line_previous = (net4.line.index[(net4.line.to_bus == start_bus)
                                     |(net4.line.loc[net4.line.index!=id,'from_bus'] == start_bus)])
    line_after = net4.line.index[(net4.line.from_bus == end_bus)
                                 |(net4.line.loc[net4.line.index!=id,'to_bus'] == end_bus)]
    if line_previous.shape[0] == 1:
        keep_line_a = net4.load.index[(net4.load.bus==start_bus)].shape[0]>=1
    else:
        keep_line_a = True
    if line_after.shape[0] == 1:
        keep_line_b = net4.load.index[(net4.load.bus==end_bus)].shape[0]>=1
    else:
        keep_line_b = True
    keep_line = ((keep_line_a) & (keep_line_b)) | ((line_previous.shape[0]!=1)&(line_after.shape[0]!=1))
    if keep_line:
        continue
    else:
        if length<=0.001:
            if (line_previous.shape[0] == 1)& (not(keep_line_a)):
                if net4.line.loc[line_previous, 'from_bus'].values[0] == start_bus:
                    net4.line.loc[line_previous, 'from_bus'] = end_bus
                else:
                    net4.line.loc[line_previous, 'to_bus'] = end_bus
                net4.load[net4.load.bus==start_bus].bus=end_bus
                net4.bus.drop(index=[start_bus], inplace=True)
                net4.line.drop(index=[id], inplace=True)
            else:
                if net4.line.loc[line_after, 'from_bus'].values[0] == end_bus:
                    net4.line.loc[line_after, 'from_bus'] = start_bus
                else:
                    net4.line.loc[line_after, 'to_bus'] = start_bus
                net4.load[net4.load.bus==end_bus].bus=start_bus
                net4.bus.drop(index=[end_bus], inplace=True)
                net4.line.drop(index=[id], inplace=True)
        else:
            if (line_previous.shape[0] == 1)& (not(keep_line_a)):
                if np.abs(net4.line.loc[line_previous, 'max_i_ka'].values[0] - net4.line.loc[id, 'max_i_ka']) <= 1e-4:
                    if net4.line.loc[line_previous, 'from_bus'].values[0] == start_bus:
                        net4.line.loc[line_previous, 'from_bus'] = end_bus
                    else:
                        net4.line.loc[line_previous, 'to_bus'] = end_bus
                    net4.line.loc[line_previous, 'length_km'] = net4.line.loc[line_previous, 'length_km'] + length
                    net4.load[net4.load.bus==start_bus].bus=end_bus
                    net4.bus.drop(index=[start_bus],inplace=True)
                    net4.line.drop(index=[id], inplace=True)
            else:
                if np.abs(net4.line.loc[line_after, 'max_i_ka'].values[0] - net4.line.loc[id, 'max_i_ka']) <= 1e-4:
                    if net4.line.loc[line_after, 'from_bus'].values[0] == end_bus:
                        net4.line.loc[line_after, 'from_bus'] = start_bus
                    else:
                        net4.line.loc[line_after, 'to_bus'] = start_bus
                    net4.line.loc[line_after, 'length_km'] = net4.line.loc[line_after, 'length_km']+length
                    net4.bus.drop(index=[end_bus],inplace=True)
                    net4.line.drop(index=[id], inplace=True)
                    net4.load[net4.load.bus==end_bus].bus=start_bus

print(net4.bus.shape)
generate_plot(net4)

net5 = net4.deepcopy()


##Merge parallel lines
bus_pair = pd.DataFrame(columns=['bus_pair'])
bus_pair['bus_pair'] = net5.line.apply(lambda x: tuple(sorted([x['from_bus'], x['to_bus']])), axis=1)
bus_pair.index= net5.line.index
# Group by bus pairs and count
parallel_lines = bus_pair.groupby('bus_pair').size()
parallel_lines_buses = parallel_lines[(parallel_lines > 1)]
changed_buses = {}
for busa, busb in parallel_lines_buses.index.values:
    if busa in changed_buses.keys():
        busa = changed_buses[busa]
    if busb in changed_buses.keys():
        busb = changed_buses[busb]
    lines = (net5.line.index[(net5.line.to_bus == busa)&(net5.line.from_bus == busb)]).to_list() \
            +(net5.line.index[(net5.line.from_bus == busa)&(net5.line.to_bus == busb)]).to_list()
    print(busa, busb)
    line_previous = (net5.line.index[(net5.line.to_bus == busa)
                                     |(net5.line.from_bus == busa)])
    line_previous = line_previous.to_list()

    for id in lines:
        if id in line_previous:
            line_previous.remove(id)
    line_after = net5.line.index[(net5.line.from_bus == busb)
                                 |(net5.line.to_bus == busb)]
    line_after = line_after.to_list()
    for id in lines:
        if id in line_after:
            line_after.remove(id)
    remove_line = ((len(line_previous)==2) | (len(line_after)==2))
    if remove_line:
        previous_pairs = net5.line.loc[line_previous].apply(lambda x: tuple(sorted([x['from_bus'], x['to_bus']])),
                                                            axis=1)
        if (len(line_previous)==2)&(previous_pairs.value_counts()[0]==2):
            line_previous_buses = bus_pair.loc[line_previous].values[0]
            common_bus = list(set(line_previous_buses).intersection(set((busa, busb))))
            if len(common_bus)==1:
                common_bus = list(set(line_previous_buses).intersection(set((busa, busb))))[0]
                if (net5.load.bus==common_bus).sum()==0:
                    lines_for_remove = []
                    for i in range(2):
                        if np.abs(net5.line.loc[line_previous[i],'max_i_ka'] - net5.line.loc[lines[i],'max_i_ka'])<=1e-4:
                            lines_for_remove.append(lines[i])
                            net5.line.loc[line_previous[i], 'length_km'] = net5.line.loc[line_previous[i], 'length_km'] \
                                                                               + net5.line.loc[lines[i], 'length_km']
                            if net5.line.loc[line_previous[i], 'from_bus'] == common_bus:
                                    net5.line.loc[line_previous[i], 'from_bus'] = busb if busa == common_bus else busa
                                    changed_buses[common_bus] = net5.line.loc[line_previous[i], 'from_bus']
                            if net5.line.loc[line_previous[i], 'to_bus'] == common_bus:
                                    net5.line.loc[line_previous[i], 'to_bus'] = busb if busa == common_bus else busa
                                    changed_buses[common_bus] = net5.line.loc[line_previous[i], 'to_bus']
                    net5.line.drop(index=lines_for_remove,inplace=True)
                    net5.bus.drop(index=[common_bus], inplace=True)
        else:
            after_pairs = net5.line.loc[line_after].apply(lambda x: tuple(sorted([x['from_bus'], x['to_bus']])), axis=1)
            if after_pairs.value_counts()[0]==2:
                line_after_buses = after_pairs.values[0]
                common_bus = list(set(line_after_buses).intersection(set((busa, busb))))
                print(common_bus)
                if len(common_bus)==1:
                    common_bus = list(set(line_after_buses).intersection(set((busa, busb))))[0]
                    if (net5.load.bus==common_bus).sum()==0:
                        lines_for_remove = []
                        for i in range(2):
                            if np.abs(net5.line.loc[line_after[i],'max_i_ka'] - net5.line.loc[lines[i],'max_i_ka'])<=1e-4:
                                lines_for_remove.append(lines[i])
                                net5.line.loc[line_after[i], 'length_km'] = net5.line.loc[line_after[i], 'length_km'] \
                                                                               + net5.line.loc[lines[i], 'length_km']
                                if net5.line.loc[line_after[i], 'from_bus'] == common_bus:
                                    net5.line.loc[line_after[i], 'from_bus'] = busb if busa == common_bus else busa
                                    changed_buses[common_bus] = net5.line.loc[line_after[i], 'from_bus']
                                if net5.line.loc[line_after[i], 'to_bus'] == common_bus:
                                    net5.line.loc[line_after[i], 'to_bus'] = busb if busa == common_bus else busa
                                    changed_buses[common_bus] = net5.line.loc[line_after[i], 'to_bus']
                        net5.line.drop(index=lines_for_remove,inplace=True)
                        net5.bus.drop(index=[common_bus], inplace=True)











print(net5.bus.shape)
generate_plot(net5)
net5.line.r_ohm_per_km = 0.001
net5.line.loc[net5.line.x_ohm_per_km==0,'x_ohm_per_km'] = 0.001
net5.load.p_mw = 10/net.load.shape[0]
net5.load.q_mvar = 0.5*10/net.load.shape[0]
pp.runpp(net5)

mpc = pc.to_mpc(net5,"feeder.mat",init='flat')


