import streamlit as st
import streamlit.components.v1 as components
from app_appearance_toolbox import init, login_page, main_app

# Initialize session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# Initialize session state for login
if "activate_scenario" not in st.session_state:
    st.session_state.activate_scenario = False

# Initialize session state for topology file
if "topology_file" not in st.session_state:
    st.session_state.topology_file = None
# Initialize session state for topology file
if "P_curve_file" not in st.session_state:
    st.session_state.P_curve_file = None
# Initialize session state for topology file
if "cosphi_file" not in st.session_state:
    st.session_state.cosphi_file = None
# Initialize session state for topology file
if "PowerCurvesFile" not in st.session_state:
    st.session_state.PowerCurvesFile = None
# Initialize session state for topology file
if "EquipmentFile" not in st.session_state:
    st.session_state.EquipmentFile = None
# Initialize session state for topology file
if "configure_PV" not in st.session_state:
    st.session_state.configure_PV = False
# Initialize session state for topology file
if "PV_curves" not in st.session_state:
    st.session_state.PV_curves = False
# Initialize session state for topology file
if "line_types" not in st.session_state:
    st.session_state.line_types = None
# Initialize session state for topology file

# Initialize session state for topology file
if "groth" not in st.session_state:
    st.session_state.groth = ''
if "project_name" not in st.session_state:
    st.session_state.project_name = ''
if "topology_file" not in st.session_state:
    st.session_state.topology_file = None
if "topology_pandas" not in st.session_state:
    st.session_state.topology_pandas = None
if "topology_pandas_ready" not in st.session_state:
    st.session_state.topology_pandas_ready = None
if "horizon" not in st.session_state:
    st.session_state.horizon = ''
if "cosphi" not in st.session_state:
    st.session_state.cosphi = None
if "P_curve" not in st.session_state:
    st.session_state.P_curve = None
if "line_types" not in st.session_state:
    st.session_state.line_types = None
if "P_curve_processed" not in st.session_state:
    st.session_state.P_curve_processed = False
if "P_curve_msg" not in st.session_state:
    st.session_state.P_curve_msg = None
if "cosphi_processed" not in st.session_state:
    st.session_state.cosphi_processed = False
if "cosphi_msg" not in st.session_state:
    st.session_state.cosphi_msg = None
if "line_types_processed" not in st.session_state:
    st.session_state.line_types_processed = False
if "types_msg" not in st.session_state:
    st.session_state.types_msg = None

if "scenario_configured" not in st.session_state:
    st.session_state.scenario_configured = False
if "lines_df" not in st.session_state:
    st.session_state.lines_df = None
if "bus_df" not in st.session_state:
    st.session_state.bus_df = None
if "move_to_planning_tab" not in st.session_state:
    st.session_state.move_to_planning_tab = False
if "opt_settings_ready" not in st.session_state:
    st.session_state.opt_settings_ready = False
if "locations_BES" not in st.session_state:
    st.session_state.locations_BES = []
if "inflation_rate" not in st.session_state:
    st.session_state.inflation_rate = None
if "interest_rate" not in st.session_state:
    st.session_state.interest_rate = None
if "flexibility_cost" not in st.session_state:
    st.session_state.flexibility_cost = None
if "load_shedding_cost" not in st.session_state:
    st.session_state.load_shedding_cost = None
if "curtailment_cost" not in st.session_state:
    st.session_state.curtailment_cost = None
if "energy_cost" not in st.session_state:
    st.session_state.energy_cost = None
if "RES_flex_max" not in st.session_state:
    st.session_state.RES_flex_max = None
if "RES_PF_max" not in st.session_state:
    st.session_state.RES_PF_max = None
if "Load_flex_max" not in st.session_state:
    st.session_state.Load_flex_max = None
if "BES_cost_P" not in st.session_state:
    st.session_state.BES_cost_P = None
if "BES_cost_E" not in st.session_state:
    st.session_state.BES_cost_E = None
if "BES_min_P" not in st.session_state:
    st.session_state.BES_min_P = None
if "BES_max_P" not in st.session_state:
    st.session_state.BES_max_P = None
if "BES_efficiency" not in st.session_state:
    st.session_state.BES_efficiency = None
if "BES_cosphi" not in st.session_state:
    st.session_state.BES_cosphi = None
if "SOC_min" not in st.session_state:
    st.session_state.soc_min = None
if "SOC_max" not in st.session_state:
    st.session_state.soc_max = None
if "SOC_init" not in st.session_state:
    st.session_state.soc_init = None

if "year_results" not in st.session_state:
    st.session_state.year_results = None
if "opt_scenarios" not in st.session_state:
    st.session_state.opt_scenarios = None

init()

#
# if not st.session_state.logged_in:
#     login_page()
# else:
main_app()

