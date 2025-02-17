import pandapower as pp
import pandapower.plotting.plotly as pplotly
import numpy as np
import gurobipy as gp
import configparser
import pandas as pd
from gurobipy import GRB
import requests
import ast
import os
import pandas as pd
import plotly.io as pio
import plotly.express as px
from pf_toolbox import generate_list_of_networks, read_config, get_pv_power_curves, get_geodata, run_pfs
from tslearn.clustering import TimeSeriesKMeans
import numpy as np
import ast
import streamlit as st

def generate_curves(Ppv, Pl):
    if Ppv.empty:
        Ptot = Pl.sum(axis=1)
    else:
        Ptot = Pl.sum(axis=1)-Ppv.sum(axis=1)
    Ptot = Ptot.values.reshape(365,24)
    # Step 2: Sum the values for each specified range
    Ptot_sums = pd.DataFrame({
        '0-5': Ptot[:, 0:6].sum(axis=1),
        '6-11': Ptot[:, 6:12].sum(axis=1),
        '12-17': Ptot[:, 12:18].sum(axis=1),
        '18-23': Ptot[:, 18:24].sum(axis=1)
    })
    return Ptot_sums


def get_factors_for_losses(Ppv,Pl,labels):
    cf = np.unique(labels).shape[0]*[0]
    if Ppv.empty:
        Ptot = Pl.sum(axis=1)
    else:
        Ptot = Pl.sum(axis=1) - Ppv.sum(axis=1)
    Ptot = Ptot.values.reshape(365,24)
    for i in np.unique(labels):
        cf[i]=(Ptot[labels==i, :].mean(axis=0)/Ptot[labels==i, :].max(axis=0)).mean()
    return cf


def get_factors_for_thresholds(labels,year_results):
    cf = {i:[] for i in np.unique(labels)}
    Loading_max = (year_results['loading'].max(axis=1)).reshape(365,24)
    for i in np.unique(labels):
        if (Loading_max[labels == i] >= 90).sum() >= 1:
            ids = (Loading_max[labels == i].max(axis=1) >= 90)
            cf[i]=ids.sum() / (labels == i).sum()
        else:
            cf[i]=1
    return cf

def get_curves(Ppv,Pl,labels):
    curves = {i:{'load': {j:[] for j in Pl.columns}, 'PV':{jp:[] for jp in Ppv.columns}} for i in np.unique(labels)}
    if Ppv.empty:
        Ptot = Pl.sum(axis=1)
    else:
        Ptot = Pl.sum(axis=1) - Ppv.sum(axis=1)
    Ptot = Ptot.values.reshape(365, 24)
    for label in labels:
        id_day = (Ptot[labels==label,:]).argmax(axis=0)
        id_day = np.where(labels == label)[0][id_day]
        loads = pd.DataFrame(index = range(24), columns=Pl.columns)
        pvs = pd.DataFrame(index=range(24), columns=Ppv.columns)
        for h in range(24):
            for j in Pl.columns:
                loads.loc[h,j] = Pl.loc[id_day[h]*24+h,j]
            for k in Ppv.columns:
                pvs.loc[h,k] =  Ppv.loc[id_day[h]*24+h,k]
        curves[label]['load']={j: loads.loc[:,j] for j in Pl.columns}
        curves[label]['PV'] = {j: pvs.loc[:, j] for j in Ppv.columns}

    return curves

def get_scenarios(Ppv, Pl,T,lf,year_results):
    opt_input_data = {i:{'c_loss':[],'curves':[],'action_c':[],'days_count':[]} for i in range(T)}
    for t in range(T):
        print('  Computing scenarios for year ' + str(t + 1))
        Ptot_1 = generate_curves(Ppv,Pl*(1+lf)**t)
        km = TimeSeriesKMeans(n_clusters=3, metric="dtw")
        labels = km.fit_predict(Ptot_1)
        opt_input_data[t]['days_count'] = {i:(labels==i).sum() for i in np.unique(labels)}
        opt_input_data[t]['c_loss']=get_factors_for_losses(Ppv,Pl*(1+lf)**t,labels)
        opt_input_data[t]['curves']=get_curves(Ppv, Pl*(1+lf)**t,labels)
        opt_input_data[t]['action_c']=get_factors_for_thresholds(labels,year_results[t])
    return opt_input_data


class Solution:
    def __init__(self, l, lmax,  p, q, v, ls, p_cur, q_pv, flex_d, flex_u, upgrades, p_loss, costs, pb, pb_max):
        self.L = l
        self.Lmax = lmax
        self.P = p
        self.Q = q
        self.V = v
        self.Ls = ls
        self.P_cur = p_cur
        self.RES_q = q_pv
        self.Flex_d = flex_d
        self.Flex_u = flex_u
        self.Upgrades = upgrades
        self.P_loss = p_loss
        self.Costs = costs
        self.Pb = pb
        self.Pb_max = pb_max

def check_upgrades_max_RES(upgrades, types, lines_names,length):
    # Iterate through the array to check for upgrades
    types_n = upgrades.shape[0]  # Number of types
    lines_n = upgrades.shape[1]
    if upgrades.sum().sum().sum()>=1:
        Upgrades = pd.DataFrame(index=range(1,int(upgrades.sum().sum().sum())+1))
        k=1
        for upgrade_type in range(types_n):
            for line in range(lines_n):
                if upgrades[ upgrade_type, line] == 1:
                    Upgrades.loc[k, 'new Type'] = types.index[upgrade_type]
                    Upgrades.loc[k, 'Replace Line'] = lines_names[line]
                    Upgrades.loc[k, 'Cost(€)'] = length[line]*types.loc[types.index[upgrade_type],'cost_per_km_€']
                    k = k + 1
    else:
        Upgrades = pd.DataFrame()
    return Upgrades

def check_upgrades(upgrades, types, lines_names):
    # Iterate through the array to check for upgrades
    years = upgrades.shape[0]  # Number of years
    types_n = upgrades.shape[1]  # Number of types
    lines_n = upgrades.shape[2]
    print(upgrades)# Number of lines
    if upgrades.sum().sum().sum()>=1:
        Upgrades = pd.DataFrame(index=range(1,int(upgrades.sum().sum().sum())+1))
        k=1
        for year in range(years):
            for upgrade_type in range(types_n):
                for line in range(lines_n):
                    if upgrades[year, upgrade_type, line] == 1:
                        Upgrades.loc[k,'year']=year+1
                        Upgrades.loc[k, 'new Type'] = types.index[upgrade_type]
                        Upgrades.loc[k, 'Replace Line'] = lines_names[line]
                        k=k+1
    else:
        Upgrades = pd.DataFrame()
    return Upgrades



def create_cost_analysis_graph(sol,settings,name):
    interest = ast.literal_eval(settings['interest_rate'])/100
    inflation = ast.literal_eval(settings['inflation_rate'])/100


    df = pd.DataFrame(sol.costs, columns=['Energy Cost of Power Losses',
                                          'Load Shedding (involuntary)',
                                          'RES curtail (involuntary)',
                                          'Flexibility Purchase',
                                          'Grid Upgrades',
                                          'BESS Operation',
                                          'BESS investment'])

    Costy = df.sum(axis=1)
    NPV = 0
    for i in Costy.index:
        NPV = NPV + Costy[i]*(pow((1 + inflation) / (1 + interest), i))



    df.drop(columns=df.columns[df.sum(axis=0) <= 10], inplace=True)
    x = pd.concat([pd.DataFrame(range(1,1+df.shape[0]), columns=['Year'])] * (df.columns.shape[0]), ignore_index=True)
    # Melt the DataFrame for Plotly Express (stacked bar chart)
    df_melted = df.melt(value_vars=df.columns,
                        var_name="Cost Type", value_name="Cost")
    # Define custom colors for each "Cost Type"
    color_discrete_map = {
        'Energy Cost of Power Losses': '#1f77b4',
        'Load Shedding (involuntary)': '#ff7f0e',
        'RES curtail (involuntary)': '#2ca02c',
        'Flexibility Purchase': '#d62728',
        'Grid Upgrades': '#9467bd',
        'BESS Operation': '#8c564b',
        'BESS investment': '#e377c2'
    }
    # Generate an interactive bar plot using Plotly Express
    fig = px.bar(df_melted, x=x['Year'], y="Cost", color="Cost Type",
                 title="Total NPV =" + str(NPV),
                 labels={"Cost": "Cost (€)", "Year": "Year"},
                 barmode='stack',color_discrete_map=color_discrete_map)
    fig.update_traces(hoverlabel=dict(
        font=dict(
            family="Arial",  # You can change to any font family
            size=24,  # Change to your desired font size
            color="black"  # Change to your desired font color
        )
    ))

    # Show plot in Jupyter Notebook (if running in a Jupyter environment)
    # Export the plot to an HTML file
    pio.write_html(fig, file=name+"_cost_analysis.html", auto_open=False)
    return 0

def order_lines(netx):
    Lines_DF  = netx[0].line[netx[0].line.in_service].copy()
    bus_s = [netx[0].ext_grid.bus.values[0]]
    passed_lines = []
    while len(bus_s)>=1:
        Lines_at_s=list(pp.get_connected_elements(netx[0], 'line', bus_s[0]))
        Lines_at_s = [line for line in Lines_at_s if
                          line not in netx[0].line.index[netx[0].line.in_service == False].to_list()]
        Lines_at_s = [line for line in Lines_at_s if
                          line not in passed_lines]
        passed_lines = passed_lines + Lines_at_s
        for line in Lines_at_s:
            if Lines_DF.loc[line,'from_bus']!=bus_s[0]:
                Lines_DF.loc[line, 'to_bus'] = Lines_DF.loc[line, 'from_bus']
                Lines_DF.loc[line, 'from_bus'] = bus_s[0]
            bus_s.append(Lines_DF.loc[line, 'to_bus'])
        bus_s = bus_s[1:]
    return Lines_DF

def run_opt_BESS(line_types, netx, cluster_data,settings,cosphi, lf_results,w_loss):
    ###Generate tanphi for loads
    tan_phi = pd.Series(index = netx[0].load.name.to_list())
    for j in tan_phi.index:
        tan_phi.loc[j] = np.tan(np.arccos(cosphi.loc[j].values[0]))
    ####
    years = range(ast.literal_eval(settings['horizon']))
    types = line_types.shape[0]
    days = range(len(cluster_data[0]['curves'].keys()))
    hours = range(24)
    vb = netx[0].bus['vn_kv'].unique()[0]
    sb = 1
    zb = vb * vb / sb
    ib = sb / (vb * (3 ** 0.5))
    #Pull static data from settings
    interest = ast.literal_eval(settings['interest_rate'])/100
    year_investments = ast.literal_eval(settings['year_of_investments'])-1
    inflation = ast.literal_eval(settings['inflation_rate'])/100
    cost_energy = ast.literal_eval(settings['energy_cost'])
    flexibility_cost = ast.literal_eval(settings['flexibility_cost'])
    load_shedding_cost = ast.literal_eval(settings['load_shedding_cost'])
    res_curtailement_cost = ast.literal_eval(settings['res_curtailment_cost'])
    pbat_min = ast.literal_eval(settings['minimum_battery_storage_size_mw'])
    pbat_max = ast.literal_eval(settings['maximum_battery_storage_size_mw'])
    bat_cost_p = ast.literal_eval(settings['battery_system_cost_power_mw'])
    bat_cost_e = ast.literal_eval(settings['battery_system_cost_energy_mwh'])
    possible_bes_locations_names = ast.literal_eval(settings['candidate_storage_bus'])
    nch = ast.literal_eval(settings['charge_discharge_efficiency'])
    nd = nch
    soc_min = ast.literal_eval(settings['soc_min_percentage'])
    soc_max = ast.literal_eval(settings['soc_max_percentage'])
    soc_init = ast.literal_eval(settings['soc_init'])
    qf = np.tan(np.arccos(ast.literal_eval(settings['cosphi_b'])))
    res_flex_max = ast.literal_eval(settings['res_flexibility_max'])/100
    load_flex_max = ast.literal_eval(settings['flexibility_max_l'])/100
    pf_pv = ast.literal_eval(settings['res_power_factor_limit'])
    ######


    Lines_DF  = order_lines(netx)
    for line in Lines_DF.index:
        netx[0].line.loc[line,'from_bus']=Lines_DF.loc[line,'from_bus']
        netx[0].line.loc[line, 'to_bus'] = Lines_DF.loc[line, 'to_bus']


    ###Possible upgrades based on load flow results
    upgrade_lines=[]
    id_oos = netx[0].line.index[netx[0].line.in_service==False]
    for year in years:
        critical_lines =  np.where(np.any(lf_results[year]['loading'] > 95, axis=0))[0].tolist()
        if len(critical_lines)>=1:
            for i in critical_lines:
                if i>=id_oos:
                    name = netx[0].line.loc[i+1,'name']
                else:
                    name = netx[0].line.loc[i,'name']
                print(name)
                upgrade_lines.append([netx[0].line[netx[0].line.name==name].index[0],year])
    print(upgrade_lines)
    if upgrade_lines:
        possible_upgrades = np.array(upgrade_lines)
    else:
        possible_upgrades = np.array([100,100]).reshape(1,2)
    print(possible_upgrades)

    model = gp.Model("new")
    lines_id = Lines_DF.index.to_list()
    # Create decision variables
    i_sq = model.addVars(years, days, hours, lines_id, lb=0.0, ub=float('inf'), vtype=GRB.CONTINUOUS, name='CurrentSq')
    p_line = model.addVars(years, days, hours, lines_id, lb=-50, ub=50, vtype=GRB.CONTINUOUS, name='ActivePower')
    q_line = model.addVars(years, days, hours, lines_id, lb=-50, ub=50, vtype=GRB.CONTINUOUS, name='ReactivePower')
    v_sq = model.addVars(years, days, hours, netx[0].bus.index, lb=0.0, ub=float('inf'),
                         vtype=GRB.CONTINUOUS, name='VoltageSq')
    load_shed = model.addVars(years, days, hours, netx[0].bus.index, lb=0.0, ub=float('inf'), vtype=GRB.CONTINUOUS,
                              name='LoadShed')
    curtailed_res_power = model.addVars(years, days, hours, netx[years[-1]].sgen.index, lb=0.0, ub=float('inf'),
                                        vtype=GRB.CONTINUOUS, name='RESCur')
    res_q = model.addVars(years, days, hours, netx[years[-1]].sgen.index, lb=-50, ub=50,
                                        vtype=GRB.CONTINUOUS, name='RES_Q')

    down_flexibility = model.addVars(years, days, hours, netx[years[-1]].sgen.index, lb=0.0, ub=float('inf'),
                                     vtype=GRB.CONTINUOUS, name='FlexDo')
    up_flexibility = model.addVars(years, days, hours, netx[0].bus.index, lb=0.0, ub=float('inf'),
                                   vtype=GRB.CONTINUOUS, name='FlexUp')
    aux_up = model.addVars(years, lines_id, line_types.index, vtype=GRB.BINARY, lb=0, ub=1, name='newline')
    in_up = model.addVars(years, lines_id, line_types.index, vtype=GRB.BINARY, lb=0, ub=1, name='upgrade')

    aux_Ploss = model.addVars(years, days, hours, lines_id, lb=0.0, ub=float('inf'), vtype=GRB.CONTINUOUS, name='ploss')
    aux_Qloss = model.addVars(years, days, hours, lines_id, lb=0.0, ub=float('inf'), vtype=GRB.CONTINUOUS, name='qloss')
    aux_Vdrop = model.addVars(years, days, hours, lines_id, lb=-50, ub=50, vtype=GRB.CONTINUOUS, name='vdrop')


    possible_bes_locations=[]
    for name in possible_bes_locations_names:
        possible_bes_locations.append(netx[0].bus.index[netx[0].bus.name==name][0])
    if len(possible_bes_locations)>0:
        bes_place = model.addVars(years, possible_bes_locations, lb=0, ub=1, vtype=GRB.BINARY, name='BES_loc')


        bes_size_energy = model.addVars(years, possible_bes_locations, lb=0, ub=pbat_max, vtype=GRB.CONTINUOUS,
                                        name='BES_size_energy')
        bes_size_power = model.addVars(years, possible_bes_locations, lb=0, ub=4*pbat_max, vtype=GRB.CONTINUOUS,
                                       name='BES_size_power')
        bes_power_ch = model.addVars(years, days, hours,possible_bes_locations, lb=0, ub=pbat_max, vtype=GRB.CONTINUOUS,
                                     name ='Pch')
        bes_power_dch = model.addVars(years, days, hours,possible_bes_locations, lb=0, ub=pbat_max, vtype=GRB.CONTINUOUS,
                                      name ='Pdc')
        bes_energy = model.addVars(years, days, hours, possible_bes_locations, lb=0, ub=100, vtype=GRB.CONTINUOUS,
                                   name='energy')
        bes_q = model.addVars(years, days, hours, possible_bes_locations, lb=-qf*pbat_max, ub=qf*pbat_max,
                              vtype=GRB.CONTINUOUS,
                              name='Qb')



    model.setObjective(expr=gp.quicksum(pow((1 + inflation) / (1 + interest), year) *
                                        (w_loss*(gp.quicksum(cost_energy * aux_Ploss[year, day, hour, line]
                                                     for line in lines_id)+gp.quicksum(cost_energy * 0.06 * up_flexibility[year, day, hour, bus]
                                                     for bus in netx[0].bus.index)) * cluster_data[year]['days_count'][day]
                                         * cluster_data[year]['c_loss'][day] +
                                         gp.quicksum(load_shedding_cost * load_shed[year, day, hour, bus]
                                                     for bus in netx[0].bus.index) * cluster_data[year]['days_count'][day]
                                         + gp.quicksum(res_curtailement_cost * curtailed_res_power[year, day, hour, gen]
                                                       for gen in netx[year].sgen.index) *
                                         cluster_data[year]['days_count'][day] +
                                         gp.quicksum(flexibility_cost * down_flexibility[year, day, hour, gen]
                                                     for gen in netx[year].sgen.index) *
                                         cluster_data[year]['days_count'][day] * cluster_data[year]['action_c'][day] +
                                         gp.quicksum(flexibility_cost * up_flexibility[year, day, hour, bus]
                                                     for bus in netx[0].bus.index) * cluster_data[year]['days_count'][day]
                                         * cluster_data[year]['action_c'][day])
                                        for hour in hours
                                        for day in days
                                        for year in years)
    + gp.quicksum(pow((1 + inflation) / (1 + interest), year) *
                                                                         gp.quicksum(netx[0].line.loc[line, 'length_km'] *
                                                                                     line_types.loc[line_type,
                                                                                                    'cost_per_km_€'] *
                                                                                     in_up[year, line, line_type]
                                                                                     for line_type in line_types.index
                                                                                     for line in lines_id)
                                                                         for year in years) +
                            gp.quicksum(pow((1 + inflation) / (1 + interest), year) *
                                        cluster_data[year]['days_count'][day]*
                                        cost_energy * (bes_power_ch[year,day,hour, bes] -
                                                       bes_power_dch[year,day,hour, bes])
                                        for bes in possible_bes_locations
                                        for hour in hours
                                        for day in days
                                        for year in years) +
                            gp.quicksum(pow((1 + inflation) / (1 + interest), year) *
                                        bat_cost_p * bes_size_power[year, bes]
                                        + bat_cost_e * bes_size_energy[year, bes]
                                        for bes in possible_bes_locations
                                        for year in years),
                       sense=GRB.MINIMIZE)



    ## Constraints for maximum one investement in bes per location
    for bes in possible_bes_locations:
        model.addConstr(gp.quicksum(bes_place[year, bes] for year in years) <= 1)

    for year in years:
        st.write('Build model for year ',year+1)
        ##Constraints on bes investment per year (Energy MWh should provide at least 4h autonomy)
        for bes in possible_bes_locations:
            model.addConstr(bes_size_power[year, bes] >= bes_place[year, bes] * pbat_min)
            model.addConstr(bes_size_power[year, bes] <= bes_place[year, bes] * pbat_max)
            model.addConstr(bes_size_energy[year, bes] >= bes_place[year, bes] * pbat_min * 4)
            model.addConstr(bes_size_energy[year, bes] <= bes_place[year, bes] * pbat_max * 4)
            model.addConstr(bes_size_power[year, bes] * 4 <= bes_size_energy[year, bes])
        ## Constraints for upgrades in lines (ensure logic on investment and auxiliary variables)
        #defer = 0
        for line in lines_id:
            for t in line_types.index:
                if netx[0].line.loc[line,'type']!=line_types.loc[t,'type']:
                    model.addConstr(in_up[year, line, t] == 0)
                    model.addConstr(aux_up[year, line, t] == 0)
                if netx[0].line.loc[line,'max_i_ka']>=line_types.loc[t,'max_i_ka']:
                    model.addConstr(in_up[year, line, t] == 0)
                    model.addConstr(aux_up[year, line, t] == 0)

                if year >= 1:
                    model.addConstr(aux_up[year, line, t] >= aux_up[year - 1, line, t])
                    model.addConstr(in_up[year, line, t] == aux_up[year, line, t] - aux_up[year - 1, line, t])
                else:
                    model.addConstr(in_up[year, line, t] == aux_up[year, line, t])
                if (not(line in possible_upgrades[:,0])) | (not(year in possible_upgrades[:,1])):
                    model.addConstr(in_up[year, line, t] == 0)
                    model.addConstr(aux_up[year, line, t] == 0)
                if year<year_investments:
                    model.addConstr(in_up[year, line, t] == 0)
                    model.addConstr(aux_up[year, line, t] == 0)


        for day in days:
            for hour in hours:
                for line in lines_id:
                    bus_end = Lines_DF.loc[line, 'to_bus']
                    bus_from = Lines_DF.loc[line, 'from_bus']
                    r = Lines_DF.loc[line, 'length_km'] * Lines_DF.loc[line, 'r_ohm_per_km'] / zb
                    x = Lines_DF.loc[line, 'length_km'] * Lines_DF.loc[line, 'x_ohm_per_km'] / zb
                    lines_from_end = list(pp.get_connected_elements(netx[0], 'line', bus_end))
                    lines_from_end.remove(line)
                    lines_from_end=[line for line in lines_from_end if line not in netx[0].line.index[netx[0].line.in_service==False].to_list()]
                    gen_id = netx[year].sgen.index[netx[year].sgen['bus'] == bus_end]
                    # Auxiliary variable calculation
                    mi = 30* netx[0].line.loc[line, 'length_km'] * netx[0].line.loc[line, 'r_ohm_per_km'] * \
                                 (1.5 * netx[0].line.loc[line, 'max_i_ka']) ** 2
                    mv = 0.2
                    # model.addConstr(aux_Ploss[year, day, hour, line] == r * i_sq[year, day, hour, line])
                    # model.addConstr(aux_Qloss[year, day, hour, line] == x * i_sq[year, day, hour, line])
                    model.addConstr(aux_Ploss[year, day, hour, line] - r * i_sq[year, day, hour, line] <=
                                            mi * gp.quicksum(aux_up[year, line, t] for t in line_types.index))
                    model.addConstr(aux_Ploss[year, day, hour, line] - r * i_sq[year, day, hour, line] >=
                                            - mi * gp.quicksum(aux_up[year, line, t] for t in line_types.index))
                    model.addConstr(aux_Qloss[year, day, hour, line] - x * i_sq[year, day, hour, line] <=
                                            mi * gp.quicksum(aux_up[year, line, t] for t in line_types.index))
                    model.addConstr(aux_Qloss[year, day, hour, line] - x * i_sq[year, day, hour, line] >=
                                            - mi * gp.quicksum(aux_up[year, line, t] for t in line_types.index))
                    model.addConstr(
                                aux_Vdrop[year, day, hour, line] - (r ** 2 + x ** 2) * i_sq[year, day, hour, line] +
                                2 * r * p_line[year, day, hour, line] +
                                2 * x * q_line[year, day, hour, line]
                                <= mv * gp.quicksum(aux_up[year, line, t] for t in line_types.index))
                    model.addConstr(
                                aux_Vdrop[year, day, hour, line] - (r ** 2 + x ** 2) * i_sq[year, day, hour, line] +
                                2 * r * p_line[year, day, hour, line] +
                                2 * x * q_line[year, day, hour, line]
                                >= -mv * gp.quicksum(aux_up[year, line, t] for t in line_types.index))
                    for t in line_types.index:
                        if line_types.loc[t, 'max_i_ka'] <= netx[0].line.loc[line, 'max_i_ka']:
                            model.addConstr(aux_up[year, line, t] == 0)
                        r_t = netx[0].line.loc[line, 'length_km'] * line_types.loc[t, 'r_ohm_per_km'] / zb
                        x_t = netx[0].line.loc[line, 'length_km'] * line_types.loc[t, 'r_ohm_per_km'] / zb
                        model.addConstr(aux_Ploss[year, day, hour, line] -
                                                r_t * i_sq[year, day, hour, line] <= mi * (1 - aux_up[year, line, t]))
                        model.addConstr(aux_Ploss[year, day, hour, line] -
                                                r_t * i_sq[year, day, hour, line] >= -mi * (1 - aux_up[year, line, t]))
                        model.addConstr(aux_Qloss[year, day, hour, line] - x_t * i_sq[year, day, hour, line] <=
                                                mi * (1 - aux_up[year, line, t]))
                        model.addConstr(aux_Qloss[year, day, hour, line] - x_t * i_sq[year, day, hour, line] >=
                                                - mi * (1 - aux_up[year, line, t]))
                        model.addConstr(aux_Vdrop[year, day, hour, line] - (r_t ** 2 + x_t ** 2) * i_sq[
                                    year, day, hour, line] +
                                                2 * r_t * p_line[year, day, hour, line] +
                                                2 * x_t * q_line[year, day, hour, line] <=
                                                mv * (1 - aux_up[year, line, t]))
                        model.addConstr(aux_Vdrop[year, day, hour, line] - (r_t ** 2 + x_t ** 2) * i_sq[
                                    year, day, hour, line] +
                                                2 * r_t * p_line[year, day, hour, line] +
                                                2 * x_t * q_line[year, day, hour, line] >=
                                                - mv * (1 - aux_up[year, line, t]))
                    # Voltage Drop
                    model.addConstr(v_sq[year, day, hour, bus_end] == v_sq[year, day, hour, bus_from] +
                                            aux_Vdrop[year, day, hour, line])
                    # Power at bus
                    model.addQConstr(p_line[year, day, hour, line] ** 2 + q_line[year, day, hour, line] ** 2 <=
                                             v_sq[year, day, hour, bus_from] * i_sq[year, day, hour, line])
                    # Active Power balance cluster_data[year]['curves'][day]
                    if netx[0].bus.loc[bus_end,'name'] in possible_bes_locations:
                        pv_power = cluster_data[year]['curves'][day]['PV'][netx[0].bus.loc[bus_end,'name']][hour] \
                            if netx[0].bus.loc[bus_end,'name'] in cluster_data[year]['curves'][day]['PV'].keys() else 0
                        load = 0 if (('aux' in netx[0].bus.loc[bus_end,'name'])|(bus_end==netx[0].ext_grid.bus[0])) \
                            else cluster_data[year]['curves'][day]['load'][netx[0].bus.loc[bus_end, 'name']][hour]
                        model.addConstr(p_line[year, day, hour, line] == bes_power_ch[year, day, hour, bus_end] -
                                        bes_power_dch[year, day, hour, bus_end] +
                                        load
                                        - pv_power +
                                        gp.quicksum(curtailed_res_power[year, day, hour, gid] +
                                                    down_flexibility[year, day, hour, gid] for gid in gen_id) -
                                        load_shed[year, day, hour, bus_end] +
                                        aux_Ploss[year, day, hour, line] -
                                        up_flexibility[year, day, hour, bus_end] +
                                        gp.quicksum(p_line[year, day, hour, ld] for ld in lines_from_end))
                    # Reactive Power balance
                        model.addConstr(q_line[year, day, hour, line] == -bes_q[year, day, hour, bus_end] +
                                        (cluster_data[year]['curves'][day]['load'][netx[0].bus.loc[bus_end,'name']][hour]
                                        - load_shed[year, day, hour, bus_end]
                                        - up_flexibility[year, day, hour, bus_end]) *
                                        tan_phi[netx[0].bus.loc[bus_end,'name']] +
                                        aux_Qloss[year, day, hour, line] +
                                        gp.quicksum(res_q[year, day, hour, gid] for gid in gen_id)+
                                        gp.quicksum(q_line[year, day, hour, ld] for ld in lines_from_end))
                    else:
                        pv_power = cluster_data[year]['curves'][day]['PV'][netx[0].bus.loc[bus_end,'name']][hour] \
                            if netx[0].bus.loc[bus_end,'name'] in cluster_data[year]['curves'][day]['PV'].keys() else 0
                        load = 0 if (('aux' in netx[0].bus.loc[bus_end,'name'])|(bus_end==netx[0].ext_grid.bus[0])) \
                            else cluster_data[year]['curves'][day]['load'][netx[0].bus.loc[bus_end, 'name']][hour]
                        model.addConstr(p_line[year, day, hour, line] ==
                                        load
                                        - pv_power +
                                        gp.quicksum(curtailed_res_power[year, day, hour, gid] +
                                                    down_flexibility[year, day, hour, gid] for gid in gen_id) -
                                        load_shed[year, day, hour, bus_end] +
                                        aux_Ploss[year, day, hour, line] -
                                        up_flexibility[year, day, hour, bus_end] +
                                        gp.quicksum(p_line[year, day, hour, ld] for ld in lines_from_end))
                        # Reactive Power balance
                        tan_phi_b = 0 if load == 0 else tan_phi[netx[0].bus.loc[bus_end, 'name']]
                        model.addConstr(q_line[year, day, hour, line] ==
                                        (load - load_shed[year, day, hour, bus_end]
                                         - up_flexibility[year, day, hour, bus_end]) * tan_phi_b +
                                        aux_Qloss[year, day, hour, line] +
                                        gp.quicksum(res_q[year, day, hour, gid] for gid in gen_id) +
                                        gp.quicksum(q_line[year, day, hour, ld] for ld in lines_from_end))
                    #Line Limit
                    model.addConstr(i_sq[year, day, hour, line] * (ib ** 2) <=
                                            (1 - gp.quicksum(aux_up[year, line, t] for t in line_types.index)) *
                                            netx[0].line.loc[line, 'max_i_ka'] ** 2 +
                                            gp.quicksum(aux_up[year, line, t] *
                                                        line_types.loc[t, 'max_i_ka'] ** 2 for t in line_types.index))
                slack = netx[0].ext_grid.bus[0]
                for b in netx[0].bus.index:
                    # Voltage Limits
                    if (b == slack) | ('aux'  in netx[0].bus.loc[b,'name']):
                        if (b == slack):
                            model.addConstr(v_sq[year, day, hour, b] == 1.0)
                        else:
                            model.addConstr(v_sq[year, day, hour, b] <= 1.1 ** 2)
                            model.addConstr(v_sq[year, day, hour, b] >= 0.9 ** 2)
                        model.addConstr(up_flexibility[year, day, hour, b] == 0)
                        model.addConstr(load_shed[year, day, hour, b] == 0)
                    else:
                        model.addConstr(v_sq[year, day, hour, b] <= 1.1**2)
                        model.addConstr(v_sq[year, day, hour, b] >= 0.9**2)
                        model.addConstr(load_shed[year, day, hour, b] <=
                                        max(0,cluster_data[year]['curves'][day]['load'][netx[0].bus.loc[b, 'name']][hour]))
                        model.addConstr(up_flexibility[year, day, hour, b] >= 0)
                        model.addConstr(up_flexibility[year, day, hour, b] <=
                                        load_flex_max *
                                        max(0,cluster_data[year]['curves'][day]['load'][netx[0].bus.loc[b, 'name']][hour]))
                        model.addConstr(up_flexibility[year, day, hour, b] <= (1-gp.quicksum(aux_up[year, line, t]
                                                                                         for line in lines_id for t in line_types.index)))
                # Static generator Limits
                if netx[year].sgen.index.empty:
                    for g in netx[years[-1]].sgen.index:
                        model.addConstr(down_flexibility[year, day, hour, g] == 0.00)
                        model.addConstr(curtailed_res_power[year, day, hour, g] == 0)
                        model.addConstr(res_q[year, day, hour, g] == 0)
                else:
                    for g in netx[year].sgen.index:
                        pv_power = cluster_data[year]['curves'][day]['PV'][netx[year].bus.loc[netx[year].sgen.loc[g,'bus'],'name']][hour]
                        model.addConstr(curtailed_res_power[year, day, hour, g] <= pv_power)
                        model.addConstr(down_flexibility[year, day, hour, g] <= res_flex_max*pv_power)
                        model.addConstr(res_q[year, day, hour, g] <= np.tan(np.arccos(pf_pv))*(pv_power-curtailed_res_power[year, day, hour, g]))
                        model.addConstr(res_q[year, day, hour, g] >= -np.tan(np.arccos(pf_pv)) * (
                                    pv_power - curtailed_res_power[year, day, hour, g]))
                        model.addConstr(down_flexibility[year, day, hour, g] >= 0.00)
                for bes in possible_bes_locations:
                    model.addConstr(bes_size_power[year, bes] >= bes_power_ch[year, day, hour, bes])
                    model.addConstr(bes_size_power[year, bes] >= bes_power_dch[year, day, hour, bes])
                    model.addConstr(bes_q[year, day, hour, bes] <= qf * bes_power_ch[year, day, hour, bes])
                    model.addConstr(bes_q[year, day, hour, bes] <= qf * bes_power_dch[year, day, hour, bes])
                    model.addConstr(bes_q[year, day, hour, bes] >= -qf * bes_power_ch[year, day, hour, bes])
                    model.addConstr(bes_q[year, day, hour, bes] >= -qf * bes_power_dch[year, day, hour, bes])
                    if (hour == 0) | (hour == 23):
                        model.addConstr(bes_size_energy[year, bes]*soc_init/100 == bes_energy[year, day, hour, bes])
                    else:
                        model.addConstr(bes_size_energy[year, bes] * soc_max/100 >= bes_energy[year, day, hour, bes])
                        model.addConstr(bes_size_energy[year, bes] * soc_min/100 <= bes_energy[year, day, hour, bes])
                        model.addConstr(bes_energy[year, day, hour, bes] == bes_energy[year, day, hour-1, bes]
                                                + bes_power_ch[year, day, hour, bes] * nch -
                                                bes_power_dch[year, day, hour, bes] * nd)

    model.setParam('MIPGap', 0.01)
    model.setParam('OutputFlag', 1)
    model.setParam("TimeLimit", 5000.0)
    st.write("Optimization Problem Successfully built")
    # Optimize the model
    model.optimize()

    ##How to get vars, how to update constraints##
    # Print results
    if model.status == GRB.OPTIMAL:
        st.write("Optimization Problem Solved")
        relaxations = np.zeros((years[-1] + 1, days[-1] + 1, hours[-1] + 1, netx[0].line.shape[0]))
        for year in years:
            for day in days:
                for hour in hours:
                    for line in lines_id:
                        bus_from = netx[0].line.loc[line, 'from_bus']
                        str1 = '[' + str(year) + ',' + str(day) + ',' + str(hour) + ',' + str(line) + ']'
                        str2 = '[' + str(year) + ',' + str(day) + ',' + str(hour) + ',' + str(bus_from) + ']'
                        relaxations[year, day, hour, line] = abs(model.getVarByName('CurrentSq' + str1).x *
                                                                 model.getVarByName('VoltageSq' + str2).x -
                                                                 model.getVarByName('ActivePower' + str1).x ** 2 -
                                                                 model.getVarByName('ReactivePower' + str1).x ** 2)
        # Get Results
        loading = np.zeros((years[-1] + 1, days[-1] + 1, hours[-1] + 1, netx[0].line.shape[0]))
        loading_max = np.zeros((years[-1] + 1, days[-1] + 1, hours[-1] + 1, netx[0].line.shape[0]))
        p = np.zeros((years[-1] + 1, days[-1] + 1, hours[-1] + 1, netx[0].line.shape[0]))
        q = np.zeros((years[-1] + 1, days[-1] + 1, hours[-1] + 1, netx[0].line.shape[0]))
        v = np.zeros((years[-1] + 1, days[-1] + 1, hours[-1] + 1, netx[0].bus.shape[0]))
        ls = np.zeros((years[-1] + 1, days[-1] + 1, hours[-1] + 1, netx[0].bus.shape[0]))
        p_cur = np.zeros((years[-1] + 1, days[-1] + 1, hours[-1] + 1, netx[years[-1]].sgen.shape[0]))
        q_pv = np.zeros((years[-1] + 1, days[-1] + 1, hours[-1] + 1, netx[years[-1]].sgen.shape[0]))
        flex_d = np.zeros((years[-1] + 1, days[-1] + 1, hours[-1] + 1, netx[years[-1]].sgen.shape[0]))
        flex_u = np.zeros((years[-1] + 1, days[-1] + 1, hours[-1] + 1, netx[0].bus.shape[0]))
        upgrades = np.zeros((years[-1] + 1, types, netx[0].line.shape[0]))
        p_loss = np.zeros(years[-1] + 1)
        costs = np.zeros((years[-1] + 1, 7))
        bes_p = np.zeros((years[-1] + 1, days[-1] + 1, hours[-1] + 1, len(possible_bes_locations)))
        bes_soc = np.zeros((years[-1] + 1, days[-1] + 1, hours[-1] + 1, len(possible_bes_locations)))
        Pb_max = np.zeros(len(possible_bes_locations))
        Eb = np.zeros(len(possible_bes_locations))
        for year in years:

            p_loss[year] = sum(cluster_data[year]['days_count'][day]*cluster_data[year]['c_loss'][day]
                               * model.getVarByName(
                'ploss' + '[' + str(year) + ',' + str(day) + ',' + str(hour) + ',' + str(line) + ']').x
                               for line in lines_id for day in days for hour in hours)
            costs[year, 0] = cost_energy * p_loss[year]
            costs[year, 1] = sum(load_shedding_cost * cluster_data[year]['days_count'][day] *
                                 model.getVarByName(
                                     'LoadShed' + '[' + str(year) + ',' + str(day) + ',' + str(hour) + ',' + str(
                                         bus) + ']').x
                                 for bus in netx[0].bus.index for day in days for hour in hours)
            costs[year, 2] = sum(res_curtailement_cost * cluster_data[year]['days_count'][day] *
                                 model.getVarByName(
                                     'GenCur' + '[' + str(year) + ',' + str(day) + ',' + str(hour) + ',' + str(
                                         gen) + ']').x
                                 for gen in netx[0].gen.index for day in days for hour in hours)
            costs[year, 3] = sum(flexibility_cost * cluster_data[year]['days_count'][day] *
                                 cluster_data[year]['action_c'][day] *
                                 model.getVarByName(
                                     'FlexDo' + '[' + str(year) + ',' + str(day) + ',' + str(hour) + ',' + str(
                                         gen) + ']').x
                                 for gen in netx[0].sgen.index for day in days for hour in hours) + \
                             sum(flexibility_cost * cluster_data[year]['days_count'][day] * cluster_data[year]['action_c'][day] *
                                 model.getVarByName(
                                     'FlexUp' + '[' + str(year) + ',' + str(day) + ',' + str(hour) + ',' + str(
                                         bus) + ']').x
                                 for bus in netx[0].bus.index for day in days for hour in hours)
            costs[year, 4] = sum(netx[0].line.loc[line, 'length_km'] * line_types.loc[line_type, 'cost_per_km_€']
                                 * model.getVarByName(
                'upgrade' + '[' + str(year) + ',' + str(line) + ',' + str(line_type) + ']').x
                                 for line_type in line_types.index
                                 for line in lines_id)

            costs[year, 5] = cost_energy *  sum(cluster_data[year]['days_count'][day]*sum(model.getVarByName(
                                'Pch' + '[' + str(year) + ',' + str(day) + ',' + str(hour) + ',' + str(bes) +']').x -
                                 model.getVarByName(
                                     'Pdc' + '[' + str(year) + ',' + str(day) + ',' + str(hour) + ',' + str(bes) +']').x
                                  for hour in hours for bes in possible_bes_locations)for day in days)
            if len(possible_bes_locations)==0:
                costs[year, 6] = 0
            else:
                costs[year, 6] = (bat_cost_p*sum(model.getVarByName('BES_size_power' + '['+str(year) + ','+ str(bes) +']').x
                                             for bes in possible_bes_locations) if year==0 else 0) + \
                                (bat_cost_e*sum(model.getVarByName('BES_size_energy' + '['+str(year) + ','+ str(bes) +']').x
                                             for bes in possible_bes_locations) if year==0 else 0)
                for bes in range(len(possible_bes_locations)):
                    str1 = '['+ str(year) +  ',' +str(possible_bes_locations[bes]) + ']'
                    Pb_max[year, bes] = round(model.getVarByName('BES_size_power' + str1).x,2)
                    Eb[year, bes] = round(model.getVarByName('BES_size_energy' + str1).x,2)
            for day in days:
                for hour in hours:
                    for line in lines_id:
                        max_i = (1 - sum(
                            model.getVarByName('newline' + '[' + str(year) + ',' + str(line) + ',' + str(t) + ']').x
                            for t in line_types.index)) * netx[0].line.loc[line, 'max_i_ka'] + \
                                sum(model.getVarByName(
                                    'newline' + '[' + str(year) + ',' + str(line) + ',' + str(t) + ']').x *
                                    line_types.loc[t, 'max_i_ka'] for t in line_types.index)
                        str1 = '[' + str(year) + ',' + str(day) + ',' + str(hour) + ',' + str(line) + ']'
                        loading[year, day, hour, line] = ((model.getVarByName(
                            'CurrentSq' + str1).x ** 0.5) * ib) / max_i
                        loading_max[year, day, hour, line] = ((model.getVarByName(
                            'CurrentSq' + str1).x ** 0.5) * ib) / max_i
                        p[year, day, hour, line] = model.getVarByName('ActivePower' + str1).x
                        q[year, day, hour, line] = model.getVarByName('ReactivePower' + str1).x
                    for bus in netx[0].bus.index:
                        str1 = '[' + str(year) + ',' + str(day) + ',' + str(hour) + ',' + str(bus) + ']'
                        ls[year, day, hour, bus] = model.getVarByName('LoadShed' + str1).x
                        flex_u[year, day, hour, bus] = model.getVarByName('FlexUp' + str1).x
                        v[year, day, hour, bus] = model.getVarByName('VoltageSq' + str1).x ** 0.5
                        # Pb[year, day, hour, bus] = model.Pbat[year, day, hour, bus].value
                        # Pb_max[year, bus] = model.Pmax_bat[year, bus].value
                    for gen in netx[years[-1]].sgen.index:
                        str1 = '[' + str(year) + ',' + str(day) + ',' + str(hour) + ',' + str(gen) + ']'
                        p_cur[year, day, hour, gen] = model.getVarByName('RESCur' + str1).x
                        q_pv[year, day, hour, gen] = model.getVarByName('RES_Q' + str1).x
                        flex_d[year, day, hour, gen] = model.getVarByName('FlexDo' + str1).x
                    for bes in range(len(possible_bes_locations)):
                        str1 = '[' + str(year) + ',' + str(day) + ',' + str(hour) + ',' + str(possible_bes_locations[bes]) + ']'
                        bes_p[year, day, hour, bes] = model.getVarByName('Pdc' + str1).x - model.getVarByName('Pch' + str1).x
                        bes_soc[year, day, hour, bes] = 100*model.getVarByName('energy' + str1).x/Eb[bes] if Eb[bes]>=0.01 else 0
                for line_type in range(types):
                    for line in lines_id:
                        str1 = '[' + str(year) + ',' + str(line) + ',' + str(line_types.index[line_type]) + ']'
                        upgrades[year, line_type, line] = model.getVarByName('upgrade' + str1).x


        flags = (relaxations >= 1e-3).sum(axis=1) >= 1
        Solution.P = p
        Solution.Q = q
        Solution.L = loading
        Solution.Lmax = loading_max
        Solution.V = v
        Solution.Ls = ls
        Solution.P_cur = p_cur
        Solution.RES_q = q_pv
        Solution.Flex_d = flex_d
        Solution.Flex_u = flex_u
        Solution.P_loss = p_loss
        Solution.upgrades = upgrades
        Solution.costs = costs
        Solution.Pb_max = Pb_max
        Solution.Eb = Eb
        Solution.Pb = bes_p
        Solution.SoC = bes_soc

        # print("Objective value:", value(model.obj))
        print("Objective value:", model.getObjective().getValue())
        return Solution, flags, model.getObjective().getValue(), check_upgrades(upgrades, line_types, netx[0].line.name.values)
    else:
        print('infeasible')
        return 0





def max_RES(line_types, netx, P,settings,cosphi):
    ###Generate tanphi for loads
    tan_phi = pd.Series(index = netx[0].load.name.to_list())
    for j in tan_phi.index:
        tan_phi.loc[j] = np.tan(np.arccos(cosphi.loc[j].values[0]))
    ####
    types = line_types.shape[0]

    vb = netx[0].bus['vn_kv'].unique()[0]
    sb = 1
    zb = vb * vb / sb
    ib = sb / (vb * (3 ** 0.5))
    #Pull static data from settings
    pf_pv = ast.literal_eval(settings['res_power_factor_limit'])
    budget_constraint = ast.literal_eval(settings['budget_constraint'])
    max_RES_minus_flex = (1-ast.literal_eval(settings['res_flexibility_max'])/100)
    ######
    ##Renumber lines
    Lines_DF  = order_lines(netx)
    for line in Lines_DF.index:
        netx[0].line.loc[line,'from_bus']=Lines_DF.loc[line,'from_bus']
        netx[0].line.loc[line, 'to_bus'] = Lines_DF.loc[line, 'to_bus']

    ###Possible upgrades based on load flow results

    model = gp.Model("new")
    lines_id = Lines_DF.index.to_list()
    # Create decision variables
    p_line = model.addVars(lines_id, lb=-5000, ub=5000, vtype=GRB.CONTINUOUS, name='ActivePower')
    aux_Ploss = model.addVars(lines_id, lb=0, ub=5000, vtype=GRB.CONTINUOUS, name='Ploss')
    I_sq = model.addVars(lines_id, lb=0, ub=5000, vtype=GRB.CONTINUOUS, name='i_sq')
    q_line = model.addVars(lines_id, lb=-5000, ub=5000, vtype=GRB.CONTINUOUS, name='ReactivePower')
    v_sq = model.addVars(netx[0].bus.index, lb=0.0, ub=float('inf'),
                         vtype=GRB.CONTINUOUS, name='VoltageSq')
    PV_Rated = model.addVars(netx[0].bus.index, lb=0.0, ub=netx[0].bus.sn_mva.max(),
                         vtype=GRB.CONTINUOUS, name='PVrated')

    res_q = model.addVars(netx[0].bus.index, lb=-50, ub=50,
                                        vtype=GRB.CONTINUOUS, name='RES_Q')

    # down_flexibility = model.addVars(years, days, hours, netx[years[-1]].sgen.index, lb=0.0, ub=float('inf'),
    #                                  vtype=GRB.CONTINUOUS, name='FlexDo')
    up_flexibility = model.addVars(netx[0].bus.index, lb=0.0, ub=float('inf'),vtype=GRB.CONTINUOUS, name='FlexUp')
    aux_up = model.addVars(lines_id, line_types.index, vtype=GRB.BINARY, lb=0, ub=1, name='newline')

    aux_Vdrop = model.addVars(lines_id, lb=-50, ub=50, vtype=GRB.CONTINUOUS, name='vdrop')



    model.setObjective(expr=-gp.quicksum(PV_Rated[bus] for bus in netx[0].bus.index)
                            +0.1*gp.quicksum(aux_Ploss[line] for line in lines_id), sense=GRB.MINIMIZE)


    ###Investment Constraint
    model.addConstr(gp.quicksum(Lines_DF.loc[line, 'length_km'] * line_types.loc[line_type,'cost_per_km_€'] *
                                                                                     aux_up[line, line_type]
                                                                                     for line_type in line_types.index
                                                                                     for line in lines_id) <=  budget_constraint)

    ###Investment feasibility
    # model.addConstr(gp.quicksum(Lines_DF.loc[line, 'length_km'] * line_types.loc[line_type, 'cost_per_km'] *
    #                                 aux_up[line, line_type]
    #                                 for line_type in line_types.index
    #                                 for line in lines_id) <= c1*
    #                     (c_tot_P*UoS*gp.quicksum(PV_Rated[bus] for bus in netx2.bus.index)-
    #                      cf*(flexibility_cost+UoS)*gp.quicksum(up_flexibility[bus] for bus in netx2.bus.index)))

    for line in lines_id:
        bus_end = Lines_DF.loc[line, 'to_bus']
        bus_from = Lines_DF.loc[line, 'from_bus']
        r = Lines_DF.loc[line, 'length_km'] * Lines_DF.loc[line, 'r_ohm_per_km'] / zb
        x = Lines_DF.loc[line, 'length_km'] * Lines_DF.loc[line, 'x_ohm_per_km'] / zb
        lines_from_end = list(pp.get_connected_elements(netx[0], 'line', bus_end))
        lines_from_end.remove(line)
        lines_from_end = [line for line in lines_from_end if
                          line not in netx[0].line.index[netx[0].line.in_service == False].to_list()]
        print(bus_from,bus_end,lines_from_end)

        # Auxiliary variable calculation

        mv = 0.2

        model.addConstr(aux_Vdrop[line] +
                                2 * r * p_line[line] +
                                2 * x * q_line[line]
                                <= mv * gp.quicksum(aux_up[line, t] for t in line_types.index))
        model.addConstr(aux_Vdrop[line] +
                                2 * r * p_line[line] +
                                2 * x * q_line[ line]
                                >= -mv * gp.quicksum(aux_up[line, t] for t in line_types.index))

        model.addConstr(aux_Ploss[line] - r * I_sq[line]
                                <= mv * gp.quicksum(aux_up[line, t] for t in line_types.index))
        model.addConstr(aux_Ploss[line] - r * I_sq[line]
                                >= -mv * gp.quicksum(aux_up[line, t] for t in line_types.index))

        for t in line_types.index:
                if line_types.loc[t, 'max_i_ka'] <= netx[0].line.loc[line, 'max_i_ka']:
                        model.addConstr(aux_up[line, t] == 0)
                print(netx[0].line.loc[line, 'name'],netx[0].line.loc[line, 'type'],t,line_types.loc[t, 'type'])
                if line_types.loc[t, 'type'] != netx[0].line.loc[line, 'type']:
                        model.addConstr(aux_up[line, t] == 0)
                r_t = netx[0].line.loc[line, 'length_km'] * line_types.loc[t, 'r_ohm_per_km'] / zb
                x_t = netx[0].line.loc[line, 'length_km'] * line_types.loc[t, 'r_ohm_per_km'] / zb
                model.addConstr(aux_Vdrop[line] +
                                                2 * r_t * p_line[line] +
                                                2 * x_t * q_line[line] <=
                                                mv * (1 - aux_up[line, t]))
                model.addConstr(aux_Vdrop[line] +
                                                2 * r_t * p_line[line] +
                                                2 * x_t * q_line[line] >=
                                                - mv * (1 - aux_up[ line, t]))

                model.addConstr(aux_Ploss[line] - r_t * I_sq[line]  <=
                                                mv * (1 - aux_up[line, t]))
                model.addConstr(aux_Ploss[line] - r_t * I_sq[line] >=
                                                - mv * (1 - aux_up[ line, t]))
        # Voltage Drop
        model.addConstr(v_sq[bus_end] == v_sq[bus_from] +
                                            aux_Vdrop[line])

        # Active Power balance cluster_data[year]['curves'][day]
        load = 0 if (('aux' in netx[0].bus.loc[bus_end,'name'])|(bus_end==netx[0].ext_grid.bus[0])) \
            else P[netx[0].bus.loc[bus_end, 'name']]
        model.addConstr(p_line[line] == load + up_flexibility[bus_end]
                                        - PV_Rated[bus_end]*max_RES_minus_flex +
                                        gp.quicksum(p_line[ld] for ld in lines_from_end))
        # Reactive Power balance
        tan_phi_b = 0 if load == 0 else tan_phi[netx[0].bus.loc[bus_end, 'name']]
        model.addConstr(q_line[line] ==
                                        (load) * tan_phi_b +
                                        -res_q[bus_end] +
                                        gp.quicksum(q_line[ld] for ld in lines_from_end))
        # Current Limit
        model.addConstr(q_line[line]**2+p_line[line]**2 <= I_sq[line])
        #Line Limit
        model.addConstr(q_line[line]**2+p_line[line]**2<=
                                            (1 - gp.quicksum(aux_up[line, t] for t in line_types.index)) *
                                ((3**0.5)*vb*netx[0].line.loc[line, 'max_i_ka']) ** 2 +
                                            gp.quicksum(aux_up[line, t] *
                                                        ((3**0.5)*vb*line_types.loc[t, 'max_i_ka']) ** 2 for t in line_types.index))
        slack = netx[0].ext_grid.bus[0]
        for b in netx[0].bus.index:
            model.addConstr(res_q[b] <= np.tan(np.arccos(pf_pv)) * PV_Rated[b]*max_RES_minus_flex)
            model.addConstr(res_q[b] >= -np.tan(np.arccos(pf_pv)) * PV_Rated[b]*max_RES_minus_flex)
            # Voltage Limits
            if (b == slack) | ('aux'  in netx[0].bus.loc[b,'name']):
                model.addConstr(PV_Rated[b] == 0)
                model.addConstr(up_flexibility[b] == 0)
                if (b == slack):
                    model.addConstr(v_sq[b] == 1.0)
                else:
                    model.addConstr(v_sq[b] <= 1.1 ** 2)
                    model.addConstr(v_sq[b] >= 0.9 ** 2)
            else:
                model.addConstr(v_sq[b] <= 1.1**2)
                model.addConstr(v_sq[b] >= 0.9**2)
                model.addConstr(up_flexibility[b] <= PV_Rated[b]*(1-max_RES_minus_flex))
                model.addConstr(PV_Rated[b]<=netx[0].bus.loc[b,'sn_mva']*1)
                #model.addConstr(up_flexibility[b] <= max(0,P[netx.bus.loc[b, 'name']]/10))


    model.setParam('MIPGap', 0.01)
    model.setParam('OutputFlag', 0)
    model.setParam("TimeLimit", 5000.0)

    # Optimize the model
    model.optimize()

    ##How to get vars, how to update constraints##
    # Print results
    if model.status == GRB.OPTIMAL:
        # Get Results
        loading = np.zeros(netx[0].line.shape[0])
        I = np.zeros(netx[0].line.shape[0])
        p = np.zeros(netx[0].line.shape[0])
        q = np.zeros(netx[0].line.shape[0])
        v = np.zeros(netx[0].bus.shape[0])
        q_pv = np.zeros(netx[0].bus.shape[0])
        PVs = np.zeros(netx[0].bus.shape[0])
        upgrades = np.zeros(( types, netx[0].line.shape[0]))
        flexibility = np.zeros(netx[0].bus.shape[0])
        for line in lines_id:
            str1 = '[' + str(line) + ']'
            max_i = (1 - sum(
                model.getVarByName('newline' + '['  + str(line) + ',' + str(t) + ']').x
                for t in line_types.index)) * netx[0].line.loc[line, 'max_i_ka'] + \
                    sum(model.getVarByName(
                        'newline' + '[' + str(line) + ',' + str(t) + ']').x *
                        line_types.loc[t, 'max_i_ka'] for t in line_types.index)
            str2 = '['+str(netx[0].line.loc[line,'from_bus'])+']'
            v_sqb = model.getVarByName('VoltageSq' + str2).x * (vb**2)
            p[line] = model.getVarByName('ActivePower' + str1).x
            q[line] = model.getVarByName('ReactivePower' + str1).x
            I[line] = (model.getVarByName('i_sq' + str1).x ** 0.5)*ib
            loading[line] =100*((p[line]**2+q[line]**2)/(3*v_sqb)) / (max_i)**2
        for bus in netx[0].bus.index:
                str1 = '[' + str(bus) + ']'
                v[bus] = model.getVarByName('VoltageSq' + str1).x ** 0.5
                PVs[bus] = model.getVarByName('PVrated' + str1).x
                q_pv[bus] = model.getVarByName('RES_Q' + str1).x
                flexibility[bus] = model.getVarByName('FlexUp' + str1).x
        for line_type in range(types):
            for line in lines_id:
                str1 = '[' + str(line) + ',' + str(line_types.index[line_type]) + ']'
                upgrades[line_type, line] = model.getVarByName('newline' + str1).x



        # print("Objective value:", value(model.obj))
        print("Objective value:", model.getObjective().getValue())
        PV_sol = pd.DataFrame(PVs, index=netx[0].bus.name)
        return PV_sol,  -model.getObjective().getValue()/max_RES_minus_flex,  check_upgrades_max_RES(upgrades, line_types, netx[0].line.name.values,netx[0].line.length_km.values)
    else:
        print('infeasible')
        return 0

def get_PV_normalized_curve(geodata):
    latitude = geodata.loc[:, 'LAT'].mean()
    longitude = geodata.loc[:, 'LON'].mean()
    startyear = 2019
    endyear = 2019
    optimalinclination = 1
    outputformat = 'json'
    pvtechchoice = 'crystSi'
    peakpower = 1e3
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
        Power_curve = pd.DataFrame(hourly_data)['P'] / 1e6  # W to MW
    return Power_curve


def run_max_res_opt(Line_types, net, Psub,settings,cosphi):
    net_up = net.deepcopy()
    networks = generate_list_of_networks(settings, net)
    PVs = get_PV_normalized_curve(get_geodata(net))
    id_max = (PVs-Psub.sum(axis=1)/Psub.sum().max()).idxmax()
    sol, obj, upgrades = max_RES(line_types=Line_types, netx=networks, P=Psub.loc[id_max,:],
                                                     settings=settings, cosphi=cosphi)
    pv_locs = sol.index[(sol.fillna(0)>=0.01).values[:,0]]
    Ppv = pd.DataFrame(index=range(8760),columns=Power_curves.columns)
    for col in Ppv.columns:
        if col in pv_locs:
            Ppv.loc[:, col] = PVs * sol.loc[col].values[0]
        else:
            Ppv.loc[:, col] = PVs * 0
    for it,up in upgrades.iterrows():
        net_up.line.loc[net_up.line.name==up['Replace Line'],'max_i_ka'] = Line_types.loc['new Type','max_i_ka']

    settings_pf = settings.copy()
    settings_pf['horizon']=str(1)
    year_results = run_pfs(net=net, cosphi=cosphi, Pl=Power_curves-Ppv, settings=settings_pf)

    return sol.round(2)*1000, obj, upgrades

def run_cost_optimization(net,Line_types,opt_input_scenarios,settings,cosphi,year_results):
    networks = generate_list_of_networks(settings, net)
    sol, flags, obj, upgrades = run_opt_BESS(line_types=Line_types, netx=networks,
                                                               cluster_data=opt_input_scenarios,
                                                               settings=settings, cosphi=cosphi, lf_results=year_results,
                                                               w_loss=1)
    return sol, obj, upgrades

def run_investment_defferal_optimization(net,Line_types,opt_input_scenarios,settings,cosphi,year_results):
    networks = generate_list_of_networks(settings, net)
    sol, flags, obj, upgrades = run_opt_BESS(line_types=Line_types, netx=networks,
                                                               cluster_data=opt_input_scenarios,
                                                               settings=settings, cosphi=cosphi, lf_results=year_results,
                                                               w_loss=0.00001)
    return sol, obj, upgrades

###Code Testing

import json
Line_types = pd.read_csv('LineTypes.csv')
net = pp.from_json('topology.json')
net.line.loc[0,'in_service']=False
settings = read_config(filename='configs/test3.cfg')
cosphi = pd.read_csv('coshpi.csv', index_col=0, delimiter=';')
Power_curves = pd.read_csv('P.csv',index_col=0, delimiter=';')
Power_curves.index = range(8760)

with open('PF_results.json', "r") as f:
    year_results = json.load(f)

year_results = {int(k): v for k, v in year_results.items()}
for t in year_results.keys():
    year_results[t]['loading'] = np.array(year_results[t]['loading'])
    year_results[t]['v'] = np.array(year_results[t]['v'])
#
run_max_res_opt(Line_types, net, Power_curves,settings,cosphi)

#networks = generate_list_of_networks(settings, net)
#geodata = get_geodata(net)
#Ppv = get_pv_power_curves(settings, geodata=geodata)


# #
# #
#opt_input_scenarios = get_scenarios(Ppv, Power_curves, ast.literal_eval(settings['horizon']), ast.literal_eval(settings['load_groth_rate'])/100, year_results)
print('a')
# #
# sol, flags, obj, upgrades = run_opt_BESS(line_types=Line_types, netx=networks,
#                                                            cluster_data=opt_input_scenarios,
#                                                            settings=settings, cosphi=cosphi, lf_results=year_results,
#                                                            line_names=net.line.name,
#                                                            w_loss=1)

# # print('a')