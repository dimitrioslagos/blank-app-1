import streamlit as st

import streamlit as st
import streamlit.components.v1 as components

# Create a two-column layout
col1, col2 = st.columns(2)

# Read the HTML file
with open("pandapower_network_map.html", "r") as file:
    html_content = file.read()


# Display HTML content in the left column
with col1:
    st.subheader("Load Flow Results")
    components.html(html_content, height=300, scrolling=True)

