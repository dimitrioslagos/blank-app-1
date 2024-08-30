import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from fast_PF import generate_pp_net, read_config, get_pv_power_curves
# Initialize session state for button click if it doesn't exist
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

# Create tabs
tab1, tab2 = st.tabs(["Topology Data Input", "Load Flow"])

# Content for the first tab
with tab1:
    # Set the title of the app
    st.title("Topology File Upload")
    # Create a file uploader widget
    uploaded_file = st.file_uploader("Choose a excel file",  type=["xlsx", "xls"])
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        ##Generate Networks
        networks = generate_pp_net(xlsx_filename=uploaded_file, settings_file='settings_spain.cfg')
        settings = read_config(filename='settings_spain.cfg')
        # Display the first few rows of the DataFrame
        st.write("Preview of Line Data:")
        st.write(networks[0].line.head())
        # Display the first few rows of the DataFrame
        st.write("Preview of Busses Data:")
        st.write(networks[0].bus.head())
    else:
        st.write("Please upload a Topology  file.")
        
    st.subheader("Upload Geodata File")
    # Create a file uploader widget
    uploaded_file2 = st.file_uploader("Choose a csv file", type=["csv"])
    if uploaded_file2 is not None:
        PVs = get_pv_power_curves(settings_file_name='settings_spain.cfg', geodata_file=uploaded_file2)

    # Create a button in the first tab
    if st.button('Run Power Flow'):
        # Update session state to unlock the second tab
        st.session_state.button_clicked = True
        st.write("Power Flow Completed")
    else:
        st.write("Run Power Flow Analysis")

# Conditionally display content of the second tab

with tab2:
    st.subheader("Load Flow Results")

    # Display HTML content in the second tab
    try:
        with open('pandapower_network_map.html', 'r', encoding='utf-8') as file:
            html_content = file.read()
        components.html(html_content, height=300, scrolling=True)
    except FileNotFoundError:
        st.error("HTML file not found.")
