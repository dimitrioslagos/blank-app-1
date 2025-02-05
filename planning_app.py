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

if "scenario_configured" not in st.session_state:
    st.session_state.scenario_configured = False
if "lines_df" not in st.session_state:
    st.session_state.lines_df = False





init()

#
# if not st.session_state.logged_in:
#     login_page()
# else:
main_app()

