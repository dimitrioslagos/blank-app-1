import streamlit as st
import streamlit.components.v1 as components
import configparser
import time
import pandas as pd
import ast
from fast_PF import get_pv_power_curves
import os
from Service.Configuration.Topology.topology_tab_toolbox import main_code_planning_settings_topology, check_if_loops_exists,generate_diagram
from Service.Configuration.PowerCurvesTab.PowerCurvesTabTool import check_P_file, check_cosphi_file
from Service.Configuration.EquipmentTab.EquipmentTabTool import check_equipment_file
import numpy as np

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
    if (len(groth)!=0)|(len(horizon)!=0):
        st.rerun()
    return 0

def success_message(text):
    st.markdown(
        f'<h1 style="font-family: Verdana; '
        f'color: green; font-size: 12px; '
        f'font-weight: bold;">{text}</h1>',
        unsafe_allow_html=True)

def topology_tab():
    with st.form('Topology Upload'):
        uploaded_file = st.file_uploader("Choose a topology file", type=["xlsx", ".json"], key=1)
        if uploaded_file is None:
            st.session_state.topology_pandas = None
            st.session_state.topology_pandas_ready = False
        else:
            net, msg = main_code_planning_settings_topology(uploaded_file)
            if net is not None:
                if st.session_state.topology_pandas is None:
                    st.session_state.topology_pandas = net
                st.session_state.topology_pandas = check_if_loops_exists(st.session_state.topology_pandas)
                if st.session_state.topology_pandas.line[~st.session_state.topology_pandas.line.is_stub].shape[0]>=1:
                    error_message("Distribution Network has Loop")
                    with st.popover("System contains loop"):
                        st.write("Choose a line as normally de-energized:")
                        # Add a select box inside the modal
                        de_energized_line = st.selectbox(
                                "Options:",
                            st.session_state.topology_pandas.line[~st.session_state.topology_pandas.line.is_stub].name.to_list()
                            )
                        st.session_state.topology_pandas.line.loc[st.session_state.topology_pandas.line.name==
                                                                      de_energized_line,'in_service']=False
                else:
                    st.session_state.topology_file = uploaded_file
                    st.session_state.topology_pandas_ready = True
            else:
                st.markdown(
                            f'<h1 style="font-family: Verdana; '
                            f'color: red; font-size: 12px; '
                            f'font-weight: bold;">{msg}</h1>',
                            unsafe_allow_html=True)
        submit2 = st.form_submit_button('Submit Topology File')
        if submit2:
            st.rerun()
    if st.session_state.topology_pandas_ready:
        success_message("Topology Format is correct")
        if not(os.path.exists('Maps/network_map.html')):
            generate_diagram(st.session_state.topology_pandas)
            with open('Maps/network_map.html', 'r', encoding='utf-8') as file:
                html_content = file.read()
            components.html(html_content, width=1000, height=400, scrolling=True)





def get_geodata():
    buses = pd.read_excel(st.session_state.topology_file, sheet_name='Busses', index_col=0)
    geodata = pd.DataFrame(columns=['ID','NAME','LAT','LON'])
    geodata['ID'] = st.session_state.topology_pandas.bus.name
    geodata['NAME'] = st.session_state.topology_pandas.bus.name
    geodata['LAT'] = st.session_state.topology_pandas.bus_geodata.x
    geodata['LON'] = st.session_state.topology_pandas.bus_geodata.x
    return geodata

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
    return PVs.shape[0]

def PV_tab():
    n_PVs = future_PV_config()
    if st.session_state.PV_curves is None:
        if n_PVs >= 1:
            geodata = get_geodata()
            PVs = get_pv_power_curves(settings_file_name='settings_spain.cfg', geodata=geodata)
            st.session_state.PV_curves = PVs
    else:
        if n_PVs == 0:
            st.session_state.PV_curves = None
        else:
            if n_PVs != st.session_state.PV_curves.shape[1]:
                geodata = get_geodata()
                PVs = get_pv_power_curves(settings_file_name='settings_spain.cfg', geodata=geodata)
                st.session_state.PV_curves = PVs
            else:
                st.write(st.session_state.PV_curves)

def cosphi_file_change():
    print('0')
    st.session_state.cosphi = None
    st.session_state.cosphi_msg = None
    st.session_state.cosphi_processed = False

def load_tab():
    with st.form('Demand data files Upload'):
        print('Runny....')
        st.markdown(
                f'<h1 style="font-family: Verdana; '
                f'color: black; font-size: 20px; '
                f'font-weight: bold;">{"Upload Load Curves"}</h1>',
                unsafe_allow_html=True)
        if st.session_state.topology_pandas_ready is not None:
            active_power_file = st.file_uploader("Choose a csv file", type=["csv"], key=41)
            if active_power_file is not None:
                print('file_lines_now')
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
        print('topology start')
        if (equipment_file is not None):
            msg, line_types = check_equipment_file(equipment_file)
            st.session_state.line_types_msg = msg
            print('I have file')
            if line_types is not None:
                print('I have file and line types')
                st.write(line_types.head())
                st.session_state.line_types = line_types
                if not(st.session_state.line_types_processed):
                    print('processed lines')
                    success_message(st.session_state.line_types_msg)
                    st.session_state.line_types_processed = True
                    st.session_state.EquipmentFile = equipment_file
            else:
                error_message(st.session_state.line_types_msg)
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

def main_app():
    progress = get_settings_progress()
    st.progress(progress, text='Scenario Configuration Progress:' + str(progress*100) + '%')
    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

    if st.session_state.activate_scenario:
        if st.sidebar.button('Save Scenario Settings'):
            config = configparser.ConfigParser()
            config.read('configs/'+st.session_state.project_name+'.cfg')
            config.set('Settings', 'horizon', value=st.session_state.horizon)
            config.set('Settings', 'load_groth_rate', value=str(float(st.session_state.groth) / 100))
            with open('configs/'+st.session_state.project_name+'.cfg', 'w') as configfile:
                config.write(configfile)
                # Save the updated configuration back to the file
                st.markdown(
                    f'<p style='
                    f'color:Yellowgreen;'
                    f'font-size:24px;border-radius:2%;">{"Scenario Saved"}</p>',
                    unsafe_allow_html=True)
                time.sleep(1)
        st.markdown(
            f'<h1 style="font-family: Verdana; '
            f'color: black; font-size: 20px; '
            f'font-weight: bold;">{"Scenario Configuration"}</h1>',
            unsafe_allow_html=True)
        if (st.session_state.topology_file is None) | (len(st.session_state.horizon)==0) | (len(st.session_state.groth)==0):
            tabs = st.tabs(['General Planning Settings','Network Topology'])
            with tabs[0]:
                general_planning_settings_tab()
            with tabs[1]:
                topology_tab()
            if (st.session_state.topology_file is not None) & (len(st.session_state.horizon)!=0) & (len(st.session_state.groth)!=0):
                st.rerun()

        else:
            print('Here2')
            tabs = st.tabs(['Scenario Settings','Network Topology',
                            'Future PV installations','Load Curves',
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
