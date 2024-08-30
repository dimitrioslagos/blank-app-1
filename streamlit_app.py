import streamlit as st
import streamlit.components.v1 as components

# Initialize session state for button click if it doesn't exist
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

# Create tabs
tab1, tab2 = st.tabs(["PF Results", "Tab 2" if st.session_state.button_clicked else "Tab 2 (Locked)"])

# Content for the first tab
with tab1:
    # Set the title of the app
    st.title("CSV File Upload Example")

    # Create a file uploader widget
    uploaded_file = st.file_uploader("Choose a CSV file",  type=["xlsx", "xls"])

    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)
    
        # Display the first few rows of the DataFrame
        st.write("Preview of the uploaded CSV file:")
        st.write(df.head())
    
        # Optionally, display some basic information about the DataFrame
        st.write("Data Summary:")
        st.write(f"Number of rows: {len(df)}")
        st.write(f"Number of columns: {len(df.columns)}")
        
        # Show descriptive statistics
        st.write("Descriptive Statistics:")
        st.write(df.describe())
    
        # Optionally, you can add more operations, like plotting graphs or filtering data
        st.write("You can add more operations here, such as data visualization or analysis.")
    
    else:
        st.write("Please upload a CSV file.")
        
        st.subheader("Calculate Load Flow")

    # Create a button in the first tab
    if st.button('Run Power Flow'):
        # Update session state to unlock the second tab
        st.session_state.button_clicked = True
        st.write("Power Flow Completed")
    else:
        st.write("Run Power Flow Analysis")

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

