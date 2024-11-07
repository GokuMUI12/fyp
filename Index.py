from blocks import sidebar
import streamlit as st

st.set_page_config(layout="wide")

col1, col2 = st.columns(2)
with col1:
    st.header(":violet[Seismic Fault Segmenatation Using UNET Architecture.] ")
      
col1, col2 = st.columns(2)
with col1:
    st.header("Supervised By: ", divider=True)
    st.write("")  
    st.write(":green[Dr Sadaf Hussain]")
    st.write("")      
    st.subheader("Group Members: ", divider=True)
    st.write(":blue[Khawar Hasnain (SP-2021-BSCS-110)]")
    st.write(":orange[Muhammad Ammar (SP-2021-BSCS-089)]")

with col2:
    st.video("https://www.youtube.com/watch?v=FN8IAb0rG9A")

sidebar()