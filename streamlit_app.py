import streamlit as st
import streamlit.components.v1 as components

# Initialize session state for button click if it doesn't exist
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

# Create tabs
tab1, tab2 = st.tabs(["PF Results", "Tab 2" if st.session_state.button_clicked else "Tab 2 (Locked)"])

# Content for the first tab
with tab1:
    st.subheader("First Tab - Interactive Button")

    # Create a button in the first tab
    if st.button('Click Me in Tab 1'):
        # Update session state to unlock the second tab
        st.session_state.button_clicked = True
        st.write("Button clicked in Tab 1! Second tab is now available.")
    else:
        st.write("Click the button to execute the code and unlock the second tab.")

# Conditionally display content of the second tab
if st.session_state.button_clicked:
    with tab2:
        st.subheader("Load Flow Results")

        # Display HTML content in the second tab
        try:
            with open('pandapower_network_map.html', 'r', encoding='utf-8') as file:
                html_content = file.read()
            components.html(html_content, height=300, scrolling=True)
        except FileNotFoundError:
            st.error("HTML file not found.")
else:
    st.warning("The second tab will be unlocked once you click the button in the first tab.")

