import streamlit as st
import streamlit.components.v1 as components
import configparser
import time
import pandas as pd
import ast
from pf_toolbox import get_geodata,get_pv_power_curves,run_pfs, read_config, plot_network_with_lf_res,generate_boxplots,lines_df_presented, bus_df_presented
import os
from Service.Configuration.Topology.topology_tab_toolbox import main_code_planning_settings_topology, check_if_loops_exists,generate_diagram
from Service.Configuration.PowerCurvesTab.PowerCurvesTabTool import check_P_file, check_cosphi_file
from Service.Configuration.EquipmentTab.EquipmentTabTool import check_equipment_file
import numpy as np
from optmization_toolbox import get_scenarios, run_cost_optimization, create_cost_analysis_graph,run_investment_defferal_optimization
import pandapower as pp

users = {
    "annel": "annel123"
}


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def init():
    # Set the background color
    set_custom_styles()  # Light blue color

# Add custom CSS for background color
def set_custom_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f0f8ff; /* Light blue background */
            font-family: 'Arial', sans-serif; /* Set font to Arial */
        }
        
        .stTabs [data-baseweb="tab-list"] {
		gap: 4px;
            }

	    .stTabs [data-baseweb="tab"] {
		height: 40px;
        white-space: pre-wrap;
		background-color: rgba(0, 0, 0, 0.0470588);
		color: black;
		border-radius: 4px 4px 1px 1px;
		gap: 2px;
		padding-top: 5px;
		padding-bottom: 5px;
        }

        

        .stTitle {
            font-family: 'Courier New', monospace; /* Set font for title */
            font-size: 24px; /* Font size for labels */
            color: black; /* Gold color for labels (username, password) */
        }
        

        /* Style for text input labels */
        .stTextInput label {
            color: black; /* Gold color for labels (username, password) */
            font-size: 18px; /* Font size for labels */
            font-family: 'Verdana', sans-serif; /* Font for labels */
        }
        
        /* Style the dropdown button */
        .stSelectbox div[data-baseweb="select"] {
            background-color: gray; /* Dark background */
            color: #FFD700; /* Gold text */
            font-size: 16px; /* Font size */
            font-family: 'Verdana', sans-serif; /* Font family */
        }

        /* Hover effect for dropdown items */
        .stSelectbox div[data-baseweb="select"] .st-mx-2:hover {
            background-color: #4CAF50; /* Green hover background */
            color: whit

        .stTextInput input {
            font-family: 'Verdana', sans-serif; /* Set font for input fields */
            color: white; /* White text in input fields */
            background-color: #333333; /* Dark background for input fields */
        }

        /* Custom styling for specific text elements */
        .custom-text {
            color: black; /* Green color */
            font-size: 18px; /* Set text size */
            font-family: 'Roboto', sans-serif; /* Use a different font */
            font-weight: bold; /* Bold text */
        }

        </style>
        """,
        unsafe_allow_html=True
    )

def generate_empty_cfg(file_name):
    """
    Generates a .cfg file with predefined settings.

    Parameters:
        file_name (str): The name of the configuration file to be generated.
    """
    # Define the content of the configuration file
    config_content = """[Settings]
pv_locations = []
pv_powers = []
pv_installation_year = []
flexibility_cost = []
energy_cost = []
load_shedding_cost = []
res_curtailement_cost = []
interest_rate = []
inflation_rate = []
battery_system_cost_power_mw = []
battery_system_cost_energy_mwh = []
minimum_battery_storage_size_mw = []
maximum_battery_storage_size_mw = []
charge_discharge_efficiency = []
cosphi_b = []
soc_min_percentage = []
soc_max_percentage = []
soc_init = []
candidate_storage_bus = []
horizon = []
load_groth_rate = []
res_curtailment_cost = []
res_power_factor_limit = []
flexibility_max_l = []
res_flexibility_max = []
"""

    # Write the content to the file
    try:
        with open('configs/'+file_name+'.cfg', "w") as file:
            file.write(config_content)
        print(f"Configuration file '{file_name}' has been created successfully.")
        return 0
    except Exception as e:
        print(f"An error occurred while creating the configuration file: {e}")
        return 0

def general_planning_settings_tab():
    name = st.text_input("Project Name", placeholder=st.session_state.project_name)
    if len(name)!=0:
        st.session_state.project_name = name
    print(os.path.exists('configs/'+name+'.cfg'))
    if not(os.path.exists('/configs/'+name+'.cfg')):
        generate_empty_cfg(name)
    col1, col2, buff2 = st.columns([2, 2, 1])
    with col1:
        horizon = st.text_input("Number of years",
                              placeholder=st.session_state.horizon,key=11)
        if len(horizon)!=0:
            st.session_state.horizon = horizon
            if (not(str.isdigit(st.session_state.horizon))):
                error_message("Enter Integer Value, e.g. 10,15 etc.'")
                if len(st.session_state.horizon)!=0:
                    st.session_state.horizon = ''
            else:
                st.session_state.horizon = horizon
    #####################################################################
    with col2:
        groth = st.text_input("Load Groth Rate (%) per Year",
                                               placeholder=st.session_state.groth,key=12)
        if len(groth)!=0:
            st.session_state.groth = groth
            if not(is_float(st.session_state.groth)):#|str.isdigit(load_groth))):
                error_message("Enter Float Value, e.g. 3.5'")
                if len(st.session_state.groth)!=0:
                    st.session_state.groth = ''
            else:
                st.session_state.groth = groth
    if (len(st.session_state.groth)!=0)&(len(st.session_state.horizon)!=0)&(len(st.session_state.project_name)!=0):
        success_message("Scenario Settings are correct")
    #if (len(groth)!=0)|(len(horizon)!=0):
        #st.rerun()
    return 0

def success_message(text):
    st.markdown(
        f'<h1 style="font-family: Verdana; '
        f'color: green; font-size: 12px; '
        f'font-weight: bold;">{text}</h1>',
        unsafe_allow_html=True)

def topology_file_upload_handler():
    if st.session_state.topology_file is None:
        uploaded_file = st.file_uploader("Choose a topology file", type=["xlsx", ".json"], key=1)
        st.session_state.topology_file = uploaded_file
    else:
        if st.button('New topology file'):
            st.session_state.topology_file = None
            st.rerun()

def print_topology_network():
    if st.session_state.topology_pandas_ready:
        success_message("Topology Format is correct")
        generate_diagram(st.session_state.topology_pandas)
        with open('network_map.html', 'r', encoding='utf-8') as file:
            html_content = file.read()
            components.html(html_content, width=1000, height=400, scrolling=True)

def deal_with_loop():
    st.session_state.topology_pandas = check_if_loops_exists(st.session_state.topology_pandas)
    if st.session_state.topology_pandas.line[~st.session_state.topology_pandas.line.is_stub].shape[0] >= 1:
        error_message("Distribution Network has Loop")
        st.session_state.topology_pandas_ready = False
        with st.popover("System contains loop"):
            st.write("Choose a line as normally de-energized:")
            # Add a select box inside the modal
            line_options = st.session_state.topology_pandas.line[
                ~st.session_state.topology_pandas.line.is_stub].name.to_list()
            de_energized_line = st.selectbox("Options:", line_options, index=None)
            if de_energized_line is not None:
                st.session_state.topology_pandas.line.loc[
                    st.session_state.topology_pandas.line.name == de_energized_line, 'in_service'] = False
                st.session_state.topology_pandas = check_if_loops_exists(st.session_state.topology_pandas)
                if st.session_state.topology_pandas.line[~st.session_state.topology_pandas.line.is_stub].shape[0] == 0:
                    st.session_state.topology_pandas_ready = True
    else:
        st.session_state.topology_pandas_ready = True

def topology_tab():
    topology_file_upload_handler()
    print('edw')
    print(st.session_state.topology_file)
    if st.session_state.topology_file is not None:
        net, msg = main_code_planning_settings_topology(st.session_state.topology_file)
        if (net is not None):
            if st.session_state.topology_pandas is None:
                st.session_state.topology_pandas = net
            deal_with_loop()
        else:
            error_message(msg)
            st.session_state.topology_pandas_ready = False
    if st.session_state.topology_pandas is not None:
        if (st.session_state.topology_pandas.line.shape[0]>=st.session_state.topology_pandas.bus.shape[0])&st.session_state.topology_pandas_ready:
            if st.button('Modify de-energized lines'):
                st.session_state.topology_pandas_ready = False
                st.session_state.topology_pandas.line.in_service= True
                st.rerun()
                #deal_with_loop()
                #print_topology_network()
    print_topology_network()







def declare_locations_of_pv():
    buses_names = st.session_state.topology_pandas.load.name
    # Initialize configparser
    config = configparser.ConfigParser()
    # Section to modify 'general' settings
    config.read('settings_spain.cfg')
    new_year = st.selectbox("Select PV Installation Year",
                                range(1,int(config.get('Settings','horizon'))+1),key=21)
    new_bus = st.selectbox("Location", buses_names,key=22)
    new_power = st.text_input("Nominal Power (kW)",'',key=23)
    try:
        float(new_power)
    except:
        st.write(':red[Choose numeric value for PV power]')
        return 0
    installations = ast.literal_eval(config.get('Settings','pv_installation_year'))
    locations = ast.literal_eval(config.get('Settings','pv_locations'))
    powers = ast.literal_eval(config.get('Settings', 'pv_powers'))
    installations.append(int(new_year)-1)
    locations.append(new_bus)
    if new_power:
        powers.append(float(new_power))
    config.set('Settings', 'pv_installation_year', value=str(installations))
    config.set('Settings', 'pv_locations', value=str(locations))
    config.set('Settings', 'pv_powers', value=str(powers))
    if st.button('Add new PV unit',key=24):
        with open('settings_spain.cfg', 'w') as configfile:
            config.write(configfile)
            # New PV unit added
        st.success("New PV unit added")
        st.rerun()
        return 0

def delete_PVs(id):
    config = configparser.ConfigParser()
    # Section to modify 'general' settings
    config.read('settings_spain.cfg')
    ###
    installations = ast.literal_eval(config.get('Settings','pv_installation_year'))
    locations = ast.literal_eval(config.get('Settings','pv_locations'))
    powers = ast.literal_eval(config.get('Settings', 'pv_powers'))
    ###
    if len(installations) >= 1:
        del installations[id]
    if len(locations)>=1:
        del locations[id]
    if len(powers) >= 1:
        del powers[id]
    config.set('Settings', 'pv_installation_year', value=str(installations))
    config.set('Settings', 'pv_locations', value=str(locations))
    config.set('Settings', 'pv_powers', value=str(powers))
    with open('settings_spain.cfg', 'w') as configfile:
        config.write(configfile)
    return 0

def future_PV_config():
    config = configparser.ConfigParser()
    # Section to modify 'general' settings
    config.read('settings_spain.cfg')
    year = pd.DataFrame(ast.literal_eval(config.get('Settings', 'pv_installation_year')), columns=['Year'])
    location = pd.DataFrame(ast.literal_eval(config.get('Settings', 'pv_locations')), columns=['location'])
    powers = pd.DataFrame(ast.literal_eval(config.get('Settings', 'pv_powers')), columns=['Nominal Power (kW)'])
    PVs = pd.concat([year, location, powers], axis=1)
    PVs.index.name = 'id'
    st.write(PVs)

    if not (st.session_state.configure_PV):
        if st.button('Configure PV units'):
            st.session_state.configure_PV = True
            st.rerun()
    else:
        declare_locations_of_pv()
        id = st.selectbox("Id", PVs.index.tolist())
        if st.button('Delete PV connection'):
            delete_PVs(id)
            st.rerun()

        if st.button('Completed Future PV units installations'):
            st.session_state.configure_PV = False
            st.rerun()
    return PVs

def PV_tab():
    PVs = future_PV_config()
    if PVs.shape[0] == 0:
        st.session_state.PV_data = None
    else:
        st.session_state.PV_data = PVs

def cosphi_file_change():
    st.session_state.cosphi = None
    st.session_state.cosphi_msg = None
    st.session_state.cosphi_processed = False

def load_tab():
    with st.form('Demand data files Upload'):
        st.markdown(
                f'<h1 style="font-family: Verdana; '
                f'color: black; font-size: 20px; '
                f'font-weight: bold;">{"Upload Load Curves"}</h1>',
                unsafe_allow_html=True)
        if st.session_state.topology_pandas_ready is not None:
            active_power_file = st.file_uploader("Choose a csv file", type=["csv"], key=41)
            if active_power_file is not None:
                msg, P_curve = check_P_file(st.session_state.topology_pandas.load.name.to_list(),active_power_file)
                st.session_state.P_curve_msg = msg
                if P_curve is not None:
                    P_curve.index = range(8760)
                    st.write(P_curve.head())
                    success_message(msg)
                    st.session_state.P_curve = P_curve
                    st.session_state.P_curve_processed = True
                else:
                    st.session_state.P_curve_processed = False
                    error_message(msg)

            else:
                st.session_state.P_curve_msg = ''
                st.session_state.P_curve_processed = False



                #Cosphi
        st.markdown(
                f'<h1 style="font-family: Verdana; '
                f'color: black; font-size: 20px; '
                f'font-weight: bold;">{"Upload Cosphi"}</h1>',
                unsafe_allow_html=True)

        cosphi_file = st.file_uploader("Choose a csv file", type=["csv"], key=42)
        if st.session_state.topology_pandas_ready is not None:
            if cosphi_file is not None:
                st.session_state.cosphi_msg, cosphi = check_cosphi_file(st.session_state.topology_pandas.load.name.to_list(),
                                                cosphi_file)
                if (cosphi is not None):
                    success_message(st.session_state.cosphi_msg)
                    st.write(cosphi.head())
                    st.session_state.cosphi = cosphi
                    st.session_state.cosphi_processed = True
                else:
                    error_message(st.session_state.cosphi_msg)
                    st.session_state.cosphi_processed = False
            else:
                st.session_state.cosphi_msg=''
                st.session_state.cosphi_processed = False

        submitted = st.form_submit_button('Submit Active Power & Cosphi Files')
        print('Submitted:',submitted)
        if submitted:
            st.rerun()




    return 0

def types_tab():
    with st.form('Equipment Type Upload'):
        equipment_file = st.file_uploader("Choose a csv file", type=["csv"], key=51)
        if (equipment_file is not None):
            msg, line_types = check_equipment_file(equipment_file)
            st.session_state.types_msg = msg
            if line_types is not None:
                line_types.index = line_types.Name
                line_types.drop(columns=['Name'],inplace=True)
                st.write(line_types.head())
                st.session_state.line_types = line_types
                success_message(st.session_state.types_msg)
                if not(st.session_state.line_types_processed):
                    success_message(st.session_state.types_msg)
                    st.session_state.line_types_processed = True
                    st.session_state.EquipmentFile = equipment_file
            else:
                error_message(st.session_state.types_msg)
        else:
            print('I do not have file')
            st.session_state.line_types_msg = ''
            if st.session_state.line_types_processed:
                print('not processed lines')
                st.session_state.line_types_processed = False

        submitted_line_types = st.form_submit_button('Submit Equipment Types File')
        if submitted_line_types:
            st.rerun()

def error_message(text):
    st.markdown(
        f'<h1 style="font-family: Verdana; '
        f'color: red; font-size: 18px; '
        f'font-weight: bold;">{text}</h1>',
        unsafe_allow_html=True)

def get_settings_progress():
    progress = 0.0
    if (len(st.session_state.groth)!=0)&(len(st.session_state.horizon)!=0)&(len(st.session_state.project_name)!=0):
        progress = progress + 0.2
    if st.session_state.topology_pandas_ready is not None:
        progress = progress + 0.2
    if (st.session_state.cosphi_processed)&(st.session_state.P_curve_processed):
        progress = progress + 0.2
    if (st.session_state.line_types_processed):
        progress = progress + 0.2
    return np.round(progress,2)

def tab_opt_parameters():
    tabs_opt_settings = st.tabs(["Economic Parameters", "Flexibility Settings"])
    config = configparser.ConfigParser()
    # Section to modify 'general' settings
    config.read(st.session_state.project_name + '.cfg')
    with tabs_opt_settings[0]:
        economic_settings()
    with tabs_opt_settings[1]:
        sub_tab_flex_settings()



def planning_tab():
    if st.sidebar.button("Scenario Configuration"):
        st.session_state.scenario_configured = False
        st.rerun()
    st.markdown(
        f'<h1 style="font-family: Verdana; '
        f'color: black; font-size: 20px; '
        f'font-weight: bold;">{"Scenario Planning"}</h1>',
        unsafe_allow_html=True)
    save_opt_settings()
    Objective = st.selectbox("Select Optimization Goal", ['Cost Reduction', 'Investment Deferral',
                                                          'Optimal Investment for RES maximization'])
    check_if_ready_to_opt(Objective)
    if st.session_state.opt_settings_ready:
        tabs_opt = st.tabs(['Optimization Parameters','Run Optimization'])
        with tabs_opt[0]:
            tab_opt_parameters()
        with tabs_opt[1]:
            if st.session_state.opt_scenarios is None:
                if st.button('Compute Optimization Scenarios'):
                    settings = read_config(filename='configs/' + st.session_state.project_name + '.cfg')
                    PVs = get_pv_power_curves(settings, get_geodata(st.session_state.topology_pandas))
                    success_message('Generating planning scenarios out of input files')
                    st.session_state.opt_scenarios = get_scenarios(PVs, st.session_state.P_curve, int(st.session_state.horizon), float(st.session_state.groth)/100, st.session_state.year_results)
            if Objective=='Cost Reduction':
                # Convert and write JSON object to file
                if st.session_state.sol_opt_ec is None:
                    if st.button('Compute Planning Optimization'):
                        settings = read_config(filename='configs/' + st.session_state.project_name + '.cfg')
                        st.write("Planning Problem Building...")
                        sol, obj, upgrades = run_cost_optimization(net=st.session_state.topology_pandas,
                                                                 Line_types=st.session_state.line_types,
                                                                opt_input_scenarios=st.session_state.opt_scenarios,
                                                                settings=settings, cosphi=st.session_state.cosphi,
                                                                year_results=st.session_state.year_results)
                        create_cost_analysis_graph(sol, settings, st.session_state.project_name)
                        with open( st.session_state.project_name + '_cost_analysis.html', 'r',
                                  encoding='utf-8') as file:
                            html_content = file.read()
                        components.html(html_content, width=1000, height=400, scrolling=True)
                        st.write(upgrades)
            if Objective=='Investment Deferral':
                # Convert and write JSON object to file
                if st.session_state.sol_opt_ec is None:
                    if st.button('Compute Planning Optimization'):
                        settings = read_config(filename='configs/' + st.session_state.project_name + '.cfg')
                        st.write("Planning Problem Building...")
                        sol, obj, upgrades = run_investment_defferal_optimization(net=st.session_state.topology_pandas,
                                                                 Line_types=st.session_state.line_types,
                                                                opt_input_scenarios=st.session_state.opt_scenarios,
                                                                settings=settings, cosphi=st.session_state.cosphi,
                                                                year_results=st.session_state.year_results)
                        create_cost_analysis_graph(sol, settings, st.session_state.project_name)
                        with open( st.session_state.project_name + '_cost_analysis.html', 'r',
                                  encoding='utf-8') as file:
                            html_content = file.read()
                        components.html(html_content, width=1000, height=400, scrolling=True)
                        st.write(upgrades)
    else:
        tabs_opt = st.tabs(['Optimization Parameters'])
        with tabs_opt[0]:
            tab_opt_parameters()

def economic_settings():
    st.subheader("Optimization economic settings")
    inflation_rate = st.text_input("Inflation rate (%)", '')
    if not(is_float(inflation_rate)):
        error_message("Enter Float Value between 0-100, e.g. 3.5")
    else:
        if (float(inflation_rate)>100)|(float(inflation_rate)<0):
            error_message("Enter Float Value between 0-100, e.g. 3.5")
        else:
            st.session_state.inflation_rate = inflation_rate
    year_of_investments = st.text_input("Investements applicable after year:", '')
    if (not(str.isdigit(year_of_investments))):
        error_message("Enter int value between 1 and "+ st.session_state.horizon)
    else:
        if (int(year_of_investments) > int(st.session_state.horizon))|(int(year_of_investments)<1):
            error_message("Enter int value between 1 and "+st.session_state.horizon)
        else:
         st.session_state.year_of_investments = year_of_investments


    interest_rate = st.text_input("Interest Rate (%)", '')
    if not(is_float(interest_rate)):
        error_message("Enter Float Value between 0-100, e.g. 3.5")
    else:
        if (float(interest_rate)>100)|(float(interest_rate)<0):
            error_message("Enter Float Value between 0-100, e.g. 3.5")
        else:
            st.session_state.interest_rate = interest_rate

    flexibility_cost = st.text_input("Flexibility Price (€/MWh)", '')
    if not(is_float(flexibility_cost)):
        error_message("Enter Float Value >0, e.g. 50")
    else:
        if float(flexibility_cost)<0:
            error_message("Enter Float Value >0, e.g. 50")
        else:
            st.session_state.flexibility_cost = flexibility_cost

    load_shedding_cost = st.text_input("Involuntary Load Shedding Price (€/MWh)", '')
    if not(is_float(load_shedding_cost)):
        error_message("Enter Float Value >0, e.g. 50")
    else:
        if float(load_shedding_cost)<0:
            error_message("Enter Float Value >0, e.g. 50")
        else:
            st.session_state.load_shedding_cost = load_shedding_cost
    curtailment_cost = st.text_input("Involuntary RES curtailment Price (€/MWh)", '')
    if not(is_float(curtailment_cost)):
        error_message("Enter Float Value >0, e.g. 50")
    else:
        if float(curtailment_cost)<0:
            error_message("Enter Float Value >0, e.g. 50")
        else:
            st.session_state.curtailment_cost = curtailment_cost
    energy_cost = st.text_input("Energy Price (€/MWh)", '')
    if not(is_float(energy_cost)):
        error_message("Enter Float Value >0, e.g. 50")
    else:
        if float(energy_cost)<0:
            error_message("Enter Float Value >0, e.g. 50")
        else:
            st.session_state.energy_cost = energy_cost

def sub_tab_flex_settings():
    tabs_flex_settings = st.tabs(["Storage", "RES", "Demand"])
    with tabs_flex_settings[0]:
        st.subheader("Flexibility settings")
        storage_checked = st.checkbox("Consider Storage")
        if storage_checked:
            if 'locations_BES' not in st.session_state:
                st.session_state.locations_BES = []
            st.subheader("Energy Storage Parameters")
            bus = st.selectbox(label='Select candidate bus for storage', index=None,
                               options=st.session_state.topology_pandas.bus.name.values[1:])
            if st.button('Add new candidate bus'):
                if (bus is not None) & (bus not in st.session_state.locations_BES):
                    st.session_state.locations_BES.append(bus)
            if len(st.session_state.locations_BES) >= 1:
                r_bus = st.selectbox(label='Bus to remove candidate bus for storage', index=None,
                                     options=st.session_state.locations_BES)
                if (st.button('Remove bus')) & (r_bus is not None):
                    st.session_state.locations_BES.remove(r_bus)
                    # st.rerun()

            st.write(st.session_state.locations_BES)
            battery_system_cost_power_mw = st.text_input("battery system cost - Power (€/MW)", '')
            if not (is_float(battery_system_cost_power_mw)):
                error_message("Enter Float Value between >0, e.g. 500")
            else:
                if (float(battery_system_cost_power_mw) <= 0):
                    error_message("Enter Float Value between 0-100, e.g. 3.5")
                else:
                    st.session_state.BES_cost_P = battery_system_cost_power_mw
            battery_system_cost_energy_mwh = st.text_input("battery system cost - Energy (€/MWh)", '')
            if not (is_float(battery_system_cost_energy_mwh)):
                error_message("Enter Float Value between >0, e.g. 500")
            else:
                if (float(battery_system_cost_energy_mwh) <= 0):
                    error_message("Enter Float Value between >0, e.g. 500")
                else:
                    st.session_state.BES_cost_E = battery_system_cost_energy_mwh

            minimum_battery_storage_size_mw = st.text_input("Minimum battery system size (MW)", '')
            if not (is_float(minimum_battery_storage_size_mw)):
                error_message("Enter Float Value between >0, e.g. 500")
            else:
                if (float(minimum_battery_storage_size_mw) <= 0):
                    error_message("Enter Float Value between >0, e.g. 500")
                else:
                    if (st.session_state.BES_max_P is not None):
                        if  (minimum_battery_storage_size_mw>=st.session_state.BES_max_P):
                            error_message("Enter Float Value lower than BES maximum power size")
                        else:
                            st.session_state.BES_min_P = minimum_battery_storage_size_mw
                    else:
                        st.session_state.BES_min_P = minimum_battery_storage_size_mw
            maximum_battery_storage_size_mw = st.text_input("Maximum battery system size (MW)", '')
            if not (is_float(maximum_battery_storage_size_mw)):
                error_message("Enter Float Value between >0, e.g. 500")
            else:
                if (float(maximum_battery_storage_size_mw) <= 0):
                    error_message("Enter Float Value between >0, e.g. 500")
                else:
                    if (st.session_state.BES_min_P is not None):
                        if (st.session_state.BES_min_P>=maximum_battery_storage_size_mw):
                            error_message("Enter Float Value greater than BES minimum power size")
                        else:
                            st.session_state.BES_max_P = maximum_battery_storage_size_mw
                    else:
                        st.session_state.BES_max_P = maximum_battery_storage_size_mw
            ###Efficiency
            charge_discharge_efficiency = st.text_input("Storage System Efficiency (%):", '')
            if not (is_float(charge_discharge_efficiency)):
                error_message("Enter Float Value between 80-100, e.g. 95.2")
            else:
                if (float(charge_discharge_efficiency) < 80)|(float(charge_discharge_efficiency) > 100):
                    error_message("Enter Float Value between 80-100, e.g. 95.2")
                else:
                    st.session_state.BES_efficiency = charge_discharge_efficiency
            ####Cosphi
            cosphi_b = st.text_input("Power Factor Limit:", '')
            if not (is_float(cosphi_b)):
                error_message("Enter Float Value between 0.8-1, e.g. 0.85")
            else:
                if (float(cosphi_b) < 0.8)|(float(cosphi_b) > 1):
                    error_message("Enter Float Value between 0.8-1, e.g. 0.85")
                else:
                    st.session_state.BES_cosphi = cosphi_b
            soc_min_percentage = st.text_input("Minimum State of Charge (%):", '')
            if not (is_float(soc_min_percentage)):
                error_message("Enter Float Value between 20-70, e.g. 30")
            else:
                if (float(soc_min_percentage) < 30)|(float(soc_min_percentage) > 70):
                    error_message("Enter Float Value between 20-70, e.g. 30")
                else:
                    st.session_state.soc_min = soc_min_percentage
            soc_max_percentage = st.text_input("Maximum State of Charge (%)", '')
            if not (is_float(soc_max_percentage)):
                error_message("Enter Float Value between 71-100, e.g. 95")
            else:
                if (float(soc_max_percentage) < 71)|(float(soc_max_percentage) > 100):
                    error_message("Enter Float Value between 71-100, e.g. 95")
                else:
                    st.session_state.soc_max = soc_max_percentage
            if (st.session_state.soc_max is not None)&(st.session_state.soc_min is not None):
                soc_init = st.text_input("SoC (%) at start of the day", '')
                if not (is_float(soc_init)):
                    error_message("Enter Float Value between SOC min and max")
                else:
                    if (float(soc_init) < float(st.session_state.soc_min)) | (float(soc_init) > float(st.session_state.soc_max)):
                        error_message("Enter Float Value between SOC min and max")
                    else:
                        st.session_state.soc_init = soc_init


    with tabs_flex_settings[1]:
        Flexibility_max = st.text_input("Maximum available Flexibility (% of available power)", '')
        if not (is_float(Flexibility_max)):
            error_message("Enter Float Value between 0-100, e.g. 3.5")
        else:
            if (float(Flexibility_max) > 100) | (float(Flexibility_max) < 0):
                error_message("Enter Float Value between 0-100, e.g. 3.5")
            else:
                st.session_state.RES_flex_max = Flexibility_max
        RES_Q_flex_checked = st.checkbox("Reactive Power Control")
        if RES_Q_flex_checked:
            res_power_factor_limit = st.text_input("RES Power Factor Limit", '')
            if not (is_float(res_power_factor_limit)):
                error_message("Enter Float Value between 0.8-1, e.g. 0.95")
            else:
                if (float(res_power_factor_limit) > 1) | (float(res_power_factor_limit) < 0.8):
                    error_message("Enter Float Value between 0.8-1, e.g. 0.95")
                else:
                    st.session_state.RES_PF_max = res_power_factor_limit
    with tabs_flex_settings[2]:
        Flexibility_max_L = st.text_input("Maximum available Flexibility (% of demand)", '')
        if not (is_float(Flexibility_max_L)):
            error_message("Enter Float Value between 0-100, e.g. 3.5")
        else:
            if (float(Flexibility_max_L) > 100) | (float(Flexibility_max_L) < 0):
                error_message("Enter Float Value between 0-100, e.g. 3.5")
            else:
                st.session_state.Load_flex_max = Flexibility_max_L


def check_if_ready_to_opt(Objective):
    settings = read_config(filename='configs/'+st.session_state.project_name+'.cfg')
    print(settings)
    print(st.session_state.project_name)
    if Objective=='Cost Reduction':
        if settings['year_of_investments'] == '[]':
            st.session_state.st.session_state.year_of_investments = False
            return 0
        if settings['interest_rate'] == '[]':
            st.session_state.opt_settings_ready = False
            return 0
        if settings['inflation_rate'] == '[]':
            st.session_state.opt_settings_ready = False
            return 0
        if settings['energy_cost'] == '[]':
            st.session_state.opt_settings_ready = False
            return 0
        if settings['load_shedding_cost'] == '[]':
            st.session_state.opt_settings_ready = False
            return 0
        if settings['flexibility_cost'] == '[]':
            st.session_state.opt_settings_ready = False
            return 0
        if settings['res_curtailment_cost'] == '[]':
            st.session_state.opt_settings_ready = False
            return 0
        if settings['res_curtailment_cost'] == '[]':
            st.session_state.opt_settings_ready = False
            return 0
        if settings['horizon'] == '[]':
            st.session_state.opt_settings_ready = False
            return 0
        if settings['load_groth_rate'] == '[]':
            st.session_state.opt_settings_ready = False
            return 0
        if settings['res_flexibility_max'] == '[]':
            st.session_state.opt_settings_ready = False
            return 0
        if settings['flexibility_max_l'] == '[]':
            st.session_state.opt_settings_ready = False
            return 0
        st.session_state.opt_settings_ready = True

def save_opt_settings():
    if st.button('Save Planning settings'):
        config = configparser.ConfigParser()
        config.read('configs/' + st.session_state.project_name + '.cfg')
        if st.session_state.flexibility_cost is not None:
            config.set('Settings', 'flexibility_cost', value=st.session_state.flexibility_cost)
        if st.session_state.energy_cost is not None:
            config.set('Settings', 'energy_cost', value=st.session_state.energy_cost)
        if st.session_state.flexibility_cost is not None:
            config.set('Settings', 'flexibility_cost', value=st.session_state.flexibility_cost)
        if st.session_state.load_shedding_cost is not None:
            config.set('Settings', 'load_shedding_cost', value=st.session_state.load_shedding_cost)
        if st.session_state.curtailment_cost is not None:
            config.set('Settings', 'res_curtailment_cost', value=st.session_state.curtailment_cost)
        if st.session_state.interest_rate is not None:
            config.set('Settings', 'interest_rate', value=st.session_state.interest_rate)
        if st.session_state.year_of_investments is not None:
            config.set('Settings', 'year_of_investments', value=st.session_state.year_of_investments)
        if st.session_state.inflation_rate is not None:
            config.set('Settings', 'inflation_rate', value=st.session_state.inflation_rate)
        if st.session_state.BES_max_P is not None:
            config.set('Settings', 'maximum_battery_storage_size_mw', value=st.session_state.BES_max_P)
        if st.session_state.BES_min_P is not None:
            config.set('Settings', 'minimum_battery_storage_size_mw', value=st.session_state.BES_min_P)
        if st.session_state.soc_max is not None:
            config.set('Settings', 'soc_max_percentage', value=st.session_state.soc_max)
        if st.session_state.soc_min is not None:
            config.set('Settings', 'soc_min_percentage', value=st.session_state.soc_min)
        if st.session_state.soc_init is not None:
            config.set('Settings', 'soc_init', value=st.session_state.soc_init)
        if st.session_state.BES_cost_E is not None:
            config.set('Settings', 'battery_system_cost_energy_mwh', value=st.session_state.BES_cost_E)
        if st.session_state.BES_cost_P is not None:
            config.set('Settings', 'battery_system_cost_power_mw', value=st.session_state.BES_cost_P)
        if st.session_state.BES_efficiency is not None:
            config.set('Settings', 'charge_discharge_efficiency', value=st.session_state.BES_efficiency)
        if st.session_state.BES_cosphi is not None:
            config.set('Settings', 'cosphi_b', value=st.session_state.BES_cosphi)
        if st.session_state.locations_BES is not None:
            config.set('Settings', 'candidate_storage_bus', value=str(st.session_state.locations_BES))
        if st.session_state.RES_PF_max is not None:
            config.set('Settings', 'res_power_factor_limit', value=st.session_state.RES_PF_max)
        if st.session_state.Load_flex_max is not None:
            config.set('Settings', 'flexibility_max_l', value=st.session_state.Load_flex_max)
        if st.session_state.RES_flex_max is not None:
            config.set('Settings', 'res_flexibility_max', value=st.session_state.RES_flex_max)

        with open('configs/' + st.session_state.project_name + '.cfg', 'w') as configfile:
            config.write(configfile)
            # Save the updated configuration back to the file
            st.markdown(
                f'<p style='
                f'color:Yellowgreen;'
                f'font-size:24px;border-radius:2%;">{"Scenario Saved"}</p>',
                unsafe_allow_html=True)
            #st.session_state.scenario_configured = True
            time.sleep(1)
            st.rerun()

def main_app():
    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()
    if st.session_state.activate_scenario:
        if not(st.session_state.scenario_configured):
            scenario_configuration()
        else:
            if st.session_state.move_to_planning_tab:
                planning_tab()
            else:
                PF_tab()

    else:
        st.markdown(
                f'<h1 style="font-family: Verdana; '
                f'color: black; font-size: 20px; '
                f'font-weight: bold;">{"Distribution System Planning"}</h1>',
                unsafe_allow_html=True)

        if st.button("Load Scenario"):
            st.write(":red[No Scenarios available]")

        if st.button("Create Scenario"):
            click_create_scenario_button()

def click_create_scenario_button():
    st.session_state.activate_scenario = True
    st.rerun()

def click_login_button(us_nm, pass_w):
    if us_nm in users and users[us_nm] == pass_w:
        st.session_state.logged_in = True
        st.session_state.username = us_nm
        st.rerun()
    elif len(us_nm)+len(pass_w)>=1:
        st.write(":red[Invalid username or password]")
    else:
        st.write("")
# Define the login page function

def login_page():
    # Create a two-column layout
    col1, col2 = st.columns([1, 2])  # Adjust the proportions as needed

    # Left column for the image
    with col2:
        st.image("tool_pic.png", use_column_width=True)
    with col1:
        # Input fields for login
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        # Login button
        if st.button("Login"):
            click_login_button(username,password)

def scenario_configuration():
    progress = get_settings_progress()
    #st.progress(progress, text='Scenario Configuration Progress:' + str(progress*100) + '%')
    if st.sidebar.button('Save Scenario Settings'):
        st.session_state.year_results = None
        st.session_state.opt_scenarios = None
        st.session_state.sol_opt_ec = None
        st.session_state.ec_upgrades = None
        st.session_state.sol_opt_ec = None
        st.session_state.ec_upgrades = None
        st.session_state.move_to_planning_tab = False
        st.session_state.bus_df = None
        st.session_state.lines_df = None
        config = configparser.ConfigParser()
        config.read('configs/' + st.session_state.project_name + '.cfg')
        config.set('Settings', 'horizon', value=st.session_state.horizon)
        config.set('Settings', 'load_groth_rate', value=str(float(st.session_state.groth) / 100))
        if st.session_state.PV_data is None:
            config.set('Settings', 'pv_locations', value='[]')
            config.set('Settings', 'pv_powers', value='[]')
            config.set('Settings', 'pv_installation_year', value='[]')
        else:
            print(st.session_state.PV_data)
            config.set('Settings', 'pv_locations', value=str(st.session_state.PV_data['location'].to_list()))
            config.set('Settings', 'pv_powers', value=str(st.session_state.PV_data['Nominal Power (kW)'].to_list()))
            config.set('Settings', 'pv_installation_year', value=str(st.session_state.PV_data['Year'].to_list()))

        with open('configs/' + st.session_state.project_name + '.cfg', 'w') as configfile:
            config.write(configfile)
            # Save the updated configuration back to the file
            st.markdown(
                f'<p style='
                f'color:Yellowgreen;'
                f'font-size:24px;border-radius:2%;">{"Scenario Saved"}</p>',
                unsafe_allow_html=True)
            st.session_state.scenario_configured = True
            time.sleep(1)
            st.rerun()
    st.markdown(
        f'<h1 style="font-family: Verdana; '
        f'color: black; font-size: 20px; '
        f'font-weight: bold;">{"Scenario Configuration"}</h1>',
        unsafe_allow_html=True)
    if (st.session_state.topology_file is None) | (len(st.session_state.horizon) == 0) | (
            len(st.session_state.groth) == 0):
        tabs = st.tabs(['General Planning Settings', 'Network Topology'])
        with tabs[0]:
            general_planning_settings_tab()
        with tabs[1]:
            topology_tab()
        if (st.session_state.topology_file is not None) & (len(st.session_state.horizon) != 0) & (
                len(st.session_state.groth) != 0):
            st.rerun()

    else:
        print('Here2')
        tabs = st.tabs(['Scenario Settings', 'Network Topology',
                        'Future PV installations', 'Load Curves',
                        'Equipment Costs & Data'])
        with tabs[0]:
            general_planning_settings_tab()
        with tabs[1]:
            topology_tab()
        with tabs[2]:
            PV_tab()
        with tabs[3]:
            load_tab()
        with tabs[4]:
            types_tab()

def PF_tab():
    if st.sidebar.button("Scenario Configuration"):
        st.session_state.scenario_configured = False
        st.rerun()
    if (st.session_state.bus_df is not None) & (st.session_state.lines_df is not None):
        if st.sidebar.button("System Planning"):
            st.session_state.move_to_planning_tab = True
            st.rerun()
    st.markdown(
        f'<h1 style="font-family: Verdana; '
        f'color: black; font-size: 20px; '
        f'font-weight: bold;">{"Load Flow Analysis"}</h1>',
        unsafe_allow_html=True)
    if st.session_state.year_results is None:
        if st.button('Run Load Flow Analysis'):
            settings = read_config(filename='configs/' + st.session_state.project_name + '.cfg')
            ###########################
            st.session_state.year_results = run_pfs(net=st.session_state.topology_pandas,
                                   cosphi=st.session_state.cosphi, Pl=st.session_state.P_curve, settings=settings)
            success_message('Load Flow Executed')
            success_message('Preparing Results ...')
            #plot_network_with_lf_res(st.session_state.topology_pandas, year_results, settings=settings)
            generate_boxplots(net=st.session_state.topology_pandas, year_results=st.session_state.year_results, settings = settings)
            plot_network_with_lf_res(net=st.session_state.topology_pandas, year_results=st.session_state.year_results, settings = settings)
            st.session_state.lines_df = lines_df_presented(st.session_state.topology_pandas, st.session_state.year_results)
            st.session_state.bus_df = bus_df_presented(st.session_state.topology_pandas, st.session_state.year_results)
    if (st.session_state.bus_df is not None) & (st.session_state.lines_df is not None):
        tabs_load_flow = st.tabs(['Analysis','Boxplot Graphs','Map'])
        with tabs_load_flow[0]:
            st.write("Line Results")
            st.dataframe(st.session_state.lines_df)
            st.write("Bus Results")
            st.dataframe(st.session_state.bus_df)
        with tabs_load_flow[1]:
            st.title("Line Results")
            # Create a list of  options
            options = st.session_state.topology_pandas.line.name  # Integer options from 1 to 10
            # Create a selectbox for integer selection
            selected_value = st.selectbox("Select the line:", options)
            try:
                with open('Figures/boxplots/boxplot_per_year_' + selected_value + '.html', 'r', encoding='utf-8') as file:
                    html_content = file.read()
                components.html(html_content, width=1000, height=400, scrolling=True)
            except FileNotFoundError:
                st.error("HTML file not found.")
            st.title("BUS Results")
            # Create a list of  options
            options = st.session_state.topology_pandas.bus.name  # Integer options from 1 to 10
            # Create a selectbox for integer selection
            selected_value = st.selectbox("Select the Bus:", options)
            try:
                with open('Figures/boxplots/boxplot_per_year_' + selected_value + '.html', 'r', encoding='utf-8') as file:
                    html_content = file.read()
                components.html(html_content, width=1000, height=400, scrolling=True)
            except FileNotFoundError:
                st.error("HTML file not found.")
        with tabs_load_flow[2]:
            settings = read_config(filename='configs/' + st.session_state.project_name + '.cfg')
            # Create a list of integer options
            options = list(range(1, ast.literal_eval(settings['horizon']) + 1))  # Integer options from 1 to 10
            # Title of the app
            st.write("Power Flow Results on Map")
            # Create a selectbox for integer selection
            selected_value = st.selectbox("Select the year:", options)
            try:
                with open('network_map' + str(selected_value - 1) + '.html', 'r', encoding='utf-8') as file:
                    html_content = file.read()
                components.html(html_content, width=1000, height=400, scrolling=True)
            except FileNotFoundError:
                st.error("HTML file not found.")
