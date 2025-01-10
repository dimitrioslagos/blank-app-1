import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import ast
#from pf_toolbox import run_pfs
from fast_PF import generate_pp_net, read_config, get_pv_power_curves, plot_network_with_lf_res,generate_boxplots
import configparser
#from clustering_toolbox import get_scenarios
#from optmization_toolbox import run_opt_BESS, max_RES, create_cost_analysis_graph, check_upgrades





# Define a dictionary to store usernames and passwords (for demo purposes) - Replaced with database
users = {
    "annel": "annel123"
}

# Initialize session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""




if 'topology_file' not in st.session_state:
    st.session_state.topology_file = False

if 'scenario_settings' not in st.session_state:
    st.session_state.scenario_settings = False
##State
if 'new_PV_button_clicked' not in st.session_state:
    st.session_state.new_PV_button_clicked = False

if 'ready_for_PF' not in st.session_state:
    st.session_state.ready_for_PF = False

if 'logged_in' not in st.session_state:
    tabs=[]
else:
    tabs = ['Scenario Configuration']

def check_if_settings_can_run_opt_multi():
    config = configparser.ConfigParser()
    config.read('settings_spain.cfg')
    data= []
    data.append(config.get('Settings', 'horizon'))
    data.append(config.get('Settings', 'interest_rate'))
    data.append(config.get('Settings', 'inflation_rate'))
    data.append(config.get('Settings', 'flexibility_cost'))
    data.append(config.get('Settings', 'load_shedding_cost'))
    data.append(config.get('Settings', 'res_curtailement_cost'))
    if [] in data:
        return False
    else:
        return True

if st.session_state.ready_for_PF:
    tabs.append('Load Flow Analysis')

if ('year_results' in st.session_state):
    tabs.append('Optimization Settings')

def generate_scenarios_for_opt(geodata_file):
    PVs = get_pv_power_curves(settings_file_name='settings_spain.cfg', geodata_file=geodata_file)


if ('opt_input_scenarios' in st.session_state):
    tabs.append('Optimization Results')




# Function to save configuration to file
def save_config(config_file='settings.cfg'):
    with open(config_file, 'w') as configfile:
        config.write(configfile)

def declare_locations_of_BES(file):
    buses_names = pd.read_excel(file, sheet_name='Busses', index_col=0).CT
    new_bus = st.selectbox("Location", buses_names, key=200)
    return new_bus


def declare_locations_of_pv(file):
    buses_names = pd.read_excel(file, sheet_name='Busses', index_col=0).CT
    # Initialize configparser
    config = configparser.ConfigParser()
    # Section to modify 'general' settings
    config.read('settings_spain.cfg')
    flag=0
    while flag==0:
        new_year = st.selectbox("Select PV Installation Year", range(1,int(config.get('Settings','horizon'))+1))
        new_bus = st.selectbox("Location", buses_names)
        new_power = st.text_input("Nominal Power (kW)",'')
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
        if st.button('Add new PV unit',key=10):
            with open('settings_spain.cfg', 'w') as configfile:
                config.write(configfile)
            # New PV unit added
            st.success("New PV unit added")
            # Content for the first tab
        if st.button('Completed Future PV units installations',key=11):
            flag=1
        if flag == 1:
            return True
        else:
            return False

tabs=[]
# Create tabs
#tab1, tab2, tab3, tab4 = st.tabs(["Scenario Configuration","Topology Data Input", "Load Flow", "Optimization Input"])
if len(tabs)>=1:
    tabs_st = st.tabs(tabs)
    with tabs_st[0]:
        if st.session_state.topology_file:
            tabs2 = st.tabs(['Scenario Settings','Network Topology', 'Future PV installations', 'Geodata','Load Curves','Equipment Costs & Data'])
        else:
            tabs2 = st.tabs(['Scenario Settings', 'Network Topology'])
        with tabs2[0]:
            # Initialize configparser
            config = configparser.ConfigParser()
            st.title("Configuration Settings")
            # Section to modify 'general' settings
            config.read('settings_spain.cfg')
            st.subheader("General Settings")
            horizon = st.text_input("Number of years", '')
            load_groth = st.text_input("Load Groth Rate (%) per Year", '')
            # Update configuration based on user input
            if st.button('Save Changes'):
                config.set('Settings','horizon',value=horizon)
                config.set('Settings', 'load_groth_rate',value=str(float(load_groth)/100))
                with open('settings_spain.cfg', 'w') as configfile:
                    config.write(configfile)
                st.session_state.scenario_settings = True
                # Save the updated configuration back to the file
                st.success("Configuration saved successfully!")
        with tabs2[1]:
            # Set the title of the app
            st.title("Topology File Upload")
            # Create a file uploader widget
            uploaded_file = st.file_uploader("Choose a excel file", type=["xlsx", "xls"], key=4)
            st.session_state.topology_file = True

        if len(tabs2)>=3:
            with tabs2[2]:
                # Check if a file has been uploaded
                st.subheader("PV connections in the next years")
                config = configparser.ConfigParser()
                # Section to modify 'general' settings
                config.read('settings_spain.cfg')
                year = pd.DataFrame(ast.literal_eval(config.get('Settings', 'pv_installation_year')),columns=['Year'])
                location = pd.DataFrame(ast.literal_eval(config.get('Settings', 'pv_locations')), columns=['location'])
                powers = pd.DataFrame(ast.literal_eval(config.get('Settings', 'pv_powers')), columns=['Nominal Power (kW)'])
                st.write(pd.concat([year,location,powers],axis=1))
                if (st.session_state.new_PV_button_clicked):
                    if st.button('Define new PV connections', key=100):
                        config = configparser.ConfigParser()
                        # Section to modify 'general' settings
                        config.read('settings_spain.cfg')
                        config.set('Settings', 'pv_installation_year', value='[]')
                        config.set('Settings', 'pv_locations', value='[]')
                        config.set('Settings', 'pv_powers', value='[]')
                        with open('settings_spain.cfg', 'w') as configfile:
                            config.write(configfile)
                        st.session_state.new_PV_button_clicked = False

                if uploaded_file is not None:
                    print('here')
                    # Read the CSV file into a DataFrame
                    ##Generate Networks
                    if not(st.session_state.new_PV_button_clicked):
                        st.session_state.new_PV_button_clicked = declare_locations_of_pv(uploaded_file)
                    if st.session_state.new_PV_button_clicked:
                        networks = generate_pp_net(xlsx_filename=uploaded_file, settings_file='settings_spain.cfg')
                        # Display the first few rows of the DataFrame
                        st.write("Preview of Line Data:")
                        st.write(networks[0].line.head())
                        # Display the first few rows of the DataFrame
                        st.write("Preview of Busses Data:")
                        st.write(networks[0].bus.head())
                else:
                    st.write("Please upload a Topology  file.")
            with tabs2[3]:
                st.subheader("Upload Geodata File")
                # Create a file uploader widget
                uploaded_file2 = st.file_uploader("Choose a csv file", type=["csv"], key=1)
                if uploaded_file2 is not None:
                    geodata = pd.read_csv(uploaded_file2, delimiter=';')
                    st.write(geodata.head())
                    PVs = get_pv_power_curves(settings_file_name='settings_spain.cfg', geodata=geodata)
                if (uploaded_file is not None):
                    st.session_state.ready_for_curves = True
            with tabs2[4]:
                settings = read_config(filename='settings_spain.cfg')
                st.subheader("Upload Load Curves")
                uploaded_file3 = st.file_uploader("Choose a csv file", type=["csv"], key=2)
                if uploaded_file3 is not None:
                    P_curve = pd.read_csv(uploaded_file3, index_col=0)
                    P_curve.index = range(8760)
                    st.write(P_curve.head())
                st.subheader("Upload Cosphi Values")
                uploaded_file4 = st.file_uploader("Choose a csv file", type=["csv"], key=3)
                if uploaded_file4 is not None:
                    cosphi = pd.read_csv(uploaded_file4, index_col=0)['0']
                    st.write(cosphi.head())
                load_factor = ast.literal_eval(settings['load_groth_rate'])
                Horizon = ast.literal_eval(settings['horizon'])
                if (uploaded_file3 is not None) & (uploaded_file4 is not None) & (st.session_state.topology_file ):
                    st.session_state.ready_for_PF = True
            with tabs2[5]:
                equipment_file = st.file_uploader("Choose a csv file", type=["csv"], key=15)
                if equipment_file is not None:
                    if 'types' not in st.session_state:
                        types = pd.read_csv(equipment_file, index_col=0, delimiter=';')
                        st.session_state.types = types
                    else:
                        st.write(st.session_state.types.head())
                else:
                    st.write('Upload Equipment Type File')
        # Conditionally display content of the second tab
        if (st.session_state.ready_for_PF):
            with tabs_st[1]:
                # Create a button in the first tab
                st.subheader("Calculate Power Flow")
                if st.button('Run Power Flow'):
                    # Update session state to unlock the second tab
                    st.session_state.button_clicked = True
                    st.write("Calculating Power Flow .... ")
                    year_results = run_pfs(networks=networks, T=Horizon, cosphi=cosphi, Pl=P_curve, Ppv=PVs)
                    st.session_state.year_results = year_results
                    st.write("Power Flow Completed")
                    plot_network_with_lf_res(networks, year_results)
                    generate_boxplots(networks, year_results)


                # Create a list of integer options
                options = list(range(1, Horizon+1))  # Integer options from 1 to 10
                # Title of the app
                st.title("Power Flow Results on Map")
                # Create a selectbox for integer selection
                selected_value = st.selectbox("Select the year:", options)
                try:
                    with open('network_map'+str(selected_value-1)+'.html', 'r', encoding='utf-8') as file:
                        html_content = file.read()
                    components.html(html_content, width=1000, height=400, scrolling=True)
                except FileNotFoundError:
                    st.error("HTML file not found.")
                # Title of the app
                st.title("Line Results")
                # Create a list of  options
                options = networks[0].line.name  # Integer options from 1 to 10
                # Create a selectbox for integer selection
                selected_value = st.selectbox("Select the line:", options)
                try:
                    with open('Figures/boxplots/boxplot_per_year_'+selected_value+'.html', 'r', encoding='utf-8') as file:
                        html_content = file.read()
                    components.html(html_content, width=1000, height=400, scrolling=True)
                except FileNotFoundError:
                    st.error("HTML file not found.")
                # Title of the app
                st.title("BUS Results")
                # Create a list of  options
                options = networks[0].bus.name  # Integer options from 1 to 10
                # Create a selectbox for integer selection
                selected_value = st.selectbox("Select the Bus:", options)
                try:
                    with open('Figures/boxplots/boxplot_per_year_' + selected_value + '.html', 'r', encoding='utf-8') as file:
                        html_content = file.read()
                    components.html(html_content, width=1000, height=400, scrolling=True)
                except FileNotFoundError:
                    st.error("HTML file not found.")

        Flag=('year_results' in st.session_state)

        if Flag:
            with tabs_st[2]:
                tabs_opt_settings = st.tabs(["Economic Parameters", "Flexibility Settings"])
                # Initialize configparser
                config = configparser.ConfigParser()
                # Section to modify 'general' settings
                config.read('settings_spain.cfg')
                with tabs_opt_settings[0]:
                    st.subheader("Optimization economic settings")
                    inflation_rate = st.text_input("Inflation rate (%)", '')
                    interest_rate = st.text_input("Interest Rate (%)", '')
                    flexibility_cost = st.text_input("Flexibility Price (€/MWh)", '')
                    load_shedding_cost = st.text_input("Involuntary Load Shedding Price (€/MWh)", '')
                    curtailment_cost = st.text_input("Involuntary RES curtailment Price (€/MWh)", '')
                with tabs_opt_settings[1]:
                    tabs_flex_settings = st.tabs(["Storage", "RES", "Demand"])
                    with tabs_flex_settings[0]:
                        st.subheader("Flexibility settings")
                        storage_checked = st.checkbox("Consider Storage")
                        if storage_checked:
                            if 'locations_BES' not in st.session_state:
                                st.session_state.locations_BES = []
                            st.subheader("Energy Storage Parameters")
                            bus = declare_locations_of_BES(uploaded_file)
                            if st.button('Add new candidate bus', key=34):
                                if not(st.session_state.locations_BES is None):
                                    if bus not in st.session_state.locations_BES:
                                        st.session_state.locations_BES.append(bus)
                                else:
                                    st.session_state.locations_BES=[bus]
                            st.write(st.session_state.locations_BES)
                            battery_system_cost_power_mw = st.text_input("battery system cost - Power (€/MW)", '')
                            battery_system_cost_energy_mwh = st.text_input("battery system cost - Energy (€/MWh)", '')
                            minimum_battery_storage_size_mw = st.text_input("Minimum battery system size (MW)", '')
                            maximum_battery_storage_size_mw = st.text_input("Maximum battery system size (MW)", '')
                            charge_discharge_efficiency = st.text_input("Storage System Efficiency (%):", '')
                            cosphi_b = st.text_input("Power Factor Limit:", '')
                            soc_min_percentage = st.text_input("Minimum State of Charge (%):", '')
                            soc_max_percentage = st.text_input("Maximum State of Charge (%)", '')
                            soc_init = st.text_input("SoC (%) at start of the day", '')
                    with tabs_flex_settings[1]:
                        Flexibility_max = st.text_input("Maximum available Flexibility (% of available power)", '')
                        RES_Q_flex_checked = st.checkbox("Reactive Power Control")
                        if RES_Q_flex_checked:
                            res_power_factor_limit = st.text_input("RES Power Factor Limit", '')
                    with tabs_flex_settings[2]:
                        Flexibility_max_L = st.text_input("Maximum available Flexibility (% of demand)", '')

                if st.button('Save Changes', key=20):
                    config.set('Settings', 'inflation_rate', value=str(float(inflation_rate) / 100))
                    config.set('Settings', 'interest_rate', value=str(float(interest_rate) / 100))
                    config.set('Settings', 'flexibility_cost', value=str(float(flexibility_cost)))
                    config.set('Settings', 'load_shedding_cost', value=str(float(load_shedding_cost)))
                    config.set('Settings', 'res_curtailment_cost', value=str(float(curtailment_cost)))
                    config.set('Settings', 'Flexibility_max_L', value=str(float(Flexibility_max_L)/100))
                    if storage_checked:
                        config.set('Settings', 'candidate_storage_bus', value=str(st.session_state.locations_BES))
                        if battery_system_cost_power_mw:
                            config.set('Settings', 'battery_system_cost_power_mw', value=battery_system_cost_power_mw)
                        config.set('Settings', 'battery_system_cost_energy_mwh', value=battery_system_cost_energy_mwh)
                        config.set('Settings', 'minimum_battery_storage_size_mw', value=minimum_battery_storage_size_mw)
                        config.set('Settings', 'maximum_battery_storage_size_mw', value=maximum_battery_storage_size_mw)
                        config.set('Settings', 'soc_min_percentage', value=soc_min_percentage)
                        config.set('Settings', 'soc_max_percentage', value=soc_max_percentage)
                        config.set('Settings', 'soc_init', value=soc_init)
                        if charge_discharge_efficiency:
                            config.set('Settings', 'charge_discharge_efficiency',
                                       value=str(float(charge_discharge_efficiency) / 100))
                        config.set('Settings', 'cosphi_b', value=cosphi_b)
                    else:
                        config.set('Settings', 'candidate_storage_bus', value='[]')
                        config.set('Settings', 'battery_system_cost_power_mw', value='[]')
                        config.set('Settings', 'battery_system_cost_energy_mwh', value='[]')
                        config.set('Settings', 'minimum_battery_storage_size_mw', value='[]')
                        config.set('Settings', 'maximum_battery_storage_size_mw', value='[]')
                        config.set('Settings', 'soc_min_percentage', value='[]')
                        config.set('Settings', 'soc_max_percentage', value='[]')
                        config.set('Settings', 'soc_init', value='[]')
                        config.set('Settings', 'charge_discharge_efficiency', value='[]')
                        config.set('Settings', 'cosphi_b', value='[]')
                        st.session_state.locations_BES = []
                    if RES_Q_flex_checked:
                        config.set('Settings', 'RES_Flexibility_max', value=str(float(interest_rate) / 100))
                        config.set('Settings', 'res_power_factor_limit', value=str(float(res_power_factor_limit)))
                    else:
                        config.set('Settings', 'RES_Flexibility_max', value=str([]))
                        config.set('Settings', 'res_power_factor_limit', value=str([]))

                    with open('settings_spain.cfg', 'w') as configfile:
                        config.write(configfile)
                    # Save the updated configuration back to the file
                    st.success("Configuration saved successfully!")
                    if 'opt_input_scenarios' not in st.session_state:
                        st.write("Computing optimization scenarios...")
                        opt_input_scenarios = get_scenarios(PVs, P_curve, Horizon, load_factor, st.session_state.year_results)
                        st.write("Optimization scenarios computed")
                        st.session_state.opt_input_scenarios = opt_input_scenarios
                    else:
                        st.write("Optimization Scenarios have been computed")

        if ('opt_input_scenarios' in st.session_state):
            with tabs_st[3]:
                Optimization_Target = st.selectbox("Select Optimization Goal", ['Cost Reduction', 'Investment Defferal'])
                if st.button('Run Optimization', key=40):
                    if Optimization_Target=='Cost Reduction':
                        w_loss = 1
                        if 'opt_results_c' not in st.session_state:
                            st.write("Solving optimization...")
                            sol, flags, obj, upgrades = run_opt_BESS(line_types=st.session_state.types, netx=networks,
                                                           cluster_data=st.session_state.opt_input_scenarios,
                                                           settings=settings, cosphi=cosphi, lf_results=st.session_state.year_results,
                                                           line_names=pd.read_excel(uploaded_file, sheet_name='Lines').name,w_loss=w_loss)
                            st.write("Optimization Solved")
                            st.session_state.opt_results_c = sol
                            st.session_state.upgrades_d = upgrades
                        else:
                            st.write("Optimization Solved")

                        try:
                            if st.session_state.upgrades_d.empty:
                                st.write('No upgrades required')
                            else:
                                st.write('Upgades Computed:')
                                st.write(st.session_state.upgrades_d)
                            create_cost_analysis_graph(st.session_state.opt_results_c, settings, 'cr_')
                            with open('cr_' + 'cost_analysis.html', 'r',
                                      encoding='utf-8') as file:
                                html_content = file.read()
                            components.html(html_content, width=1000, height=400, scrolling=True)
                        except FileNotFoundError:
                            st.error("HTML file not found.")
                    else:
                        w_loss = 0.001
                        if 'opt_results_d' not in st.session_state:
                            st.write("Solving optimization...")
                            sol, flags, obj, upgrades = run_opt_BESS(line_types=st.session_state.types, netx=networks,
                                                           cluster_data=st.session_state.opt_input_scenarios,
                                                           settings=settings, cosphi=cosphi, lf_results=st.session_state.year_results,
                                                           line_names=pd.read_excel(uploaded_file, sheet_name='Lines').name,
                                                           w_loss=w_loss)
                            st.write("Optimization Solved")
                            st.session_state.opt_results_d = sol
                            st.session_state.upgrades_d = upgrades
                        else:
                            st.write("Optimization Solved")
                        try:
                            if st.session_state.upgrades_d.empty:
                                st.write('No upgrades required')
                            else:
                                st.write('Upgades Computed:')
                                st.write(st.session_state.upgrades_d)
                            create_cost_analysis_graph(st.session_state.opt_results_d, settings, 'id_')
                            with open('id_' + 'cost_analysis.html', 'r', encoding='utf-8') as file:
                                html_content = file.read()
                            components.html(html_content, width=1000, height=400, scrolling=True)
                        except FileNotFoundError:
                            st.error("HTML file not found.")

