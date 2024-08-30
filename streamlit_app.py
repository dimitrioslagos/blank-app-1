import streamlit as st

import streamlit as st
import streamlit.components.v1 as components


tab1, tab2 = st.tabs(["PF Results", "Tab 2"])
# Content for the first tab
with tab1:
    st.subheader("First Tab - Interactive Button")

    # Create a button in the first tab
    if st.button('Click Me in Tab 1'):
        # Code to run when the button is clicked
        st.write("Button clicked in Tab 1! Running code...")

        # Example code to run (e.g., a simple computation)
        st.write(f"The result of the computation is: {result}")
    else:
        st.write("Click the button to execute the code.")



# Display HTML content in the left column
with tab2:
    st.subheader("Load Flow Results")
    components.html(html_content, height=300, scrolling=True)

