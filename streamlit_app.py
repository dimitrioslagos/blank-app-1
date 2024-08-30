import streamlit as st

import streamlit as st
import streamlit.components.v1 as components

# Read the HTML file
with open("pandapower_network_map.html", "r") as file:
    html_content = file.read()

# Display the HTML in Streamlit
components.html(html_content, height=500)  # You can adjust the height as needed
