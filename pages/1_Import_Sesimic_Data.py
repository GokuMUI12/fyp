import os
import subprocess
import streamlit as st
import segyio
import plotly.express as px
from visualizations import VISUALIZATION
from classes import SegyIO3D, Numpy3D
from blocks import sidebar

def get_segy_header(file_name):
    f = segyio.open(file_name, ignore_geometry = True)
    return segyio.tools.wrap(f.text[0])
    
def get_inline_xline_position(file_name):
    f = segyio.open(file_name, ignore_geometry = True)
    _ntraces    = len(f.trace)
    _inlines    = []
    _crosslines = []

    for i in range(_ntraces):
        headeri = f.header[i]
        _inlines.append(headeri[segyio.su.iline])
        _crosslines.append(headeri[segyio.su.xline])

    print(f'{_ntraces} traces')
    print(f'first 10 inlines: {_inlines[:10]}')
    print(f'first 10 crosslines: {_crosslines[:10]}')

    return _inlines, _crosslines

st.markdown("# 💾 Import Seismic from File")

data_option = st.radio(
     "🔘 There are a few options:",
     ('Segy3D', 'Numpy3D'), index=0, horizontal=True)

if st.button("Download F3 Seismic Dataset"):
    with st.spinner('Downloading dataset...'):
        subprocess.run(["pip", "install", "gdown"], check=True)
        subprocess.run(["gdown", "https://drive.google.com/uc?id=1R9u37AZm8N1rY5NZU2a8Nz7nyuwyma4X"], check=True)
    st.success('Dataset is downloaded')


if data_option == 'Segy3D':
    st.title("Import Seismic in SEGY format")
    if 'filename' not in st.session_state:
        st.session_state.filename = ''

    if 'inline_byte' not in st.session_state:
        # These are default settings for most of the SEGY files
        st.session_state.inline_byte = 189
        st.session_state.xline_byte = 193

    if 'failed_seismic' not in st.session_state:
        # This flag is raised if something went wrong
        st.session_state.failed_seismic = False

    filename = st.text_input('Paste the whole file path here. e.g. D:\F3.segy or D:\Kerry3D.segy  Or use the test dataset enter in the field: Dutch_Government_F3_entire_8bit_seismic.segy')

    # If the user chooses another filename, then we have to reset params
    if filename != st.session_state.filename:
        st.session_state.inline_byte = 189
        st.session_state.xline_byte = 193
        st.session_state.failed_seismic = False

    st.session_state.filename = filename
    st.write('The selected file is: ', filename)

    tab1, tab2 = st.tabs(["Headers", "Visualize"])

    if filename:
        with tab1:
            st.code(get_segy_header(filename))
       
        
        with tab2:
            with st.expander("Inline/Xline fields in the trace headers"):
                with st.form("my_form"):
                    st.session_state.inline_byte = st.number_input('Inline', value=int(st.session_state.inline_byte), format='%i')
                    st.session_state.xline_byte = st.number_input('Xline', value=int(st.session_state.xline_byte), format='%i')
                    submitted = st.form_submit_button("Read File")
            try:
                segyfile = SegyIO3D(filename, st.session_state.inline_byte, st.session_state.xline_byte)
                st.session_state.seismic_type = "3D"
    
                viz = VISUALIZATION(segyfile, st.session_state.seismic_type)
                viz.viz_data_3d(segyfile, is_fspect=False)

                st.session_state.seismic = segyfile
                # st.success('It appears that the survey is correctly read. AI/ML methods are now available in this app!')

            except RuntimeError as err: 
                st.session_state.failed_seismic = True
                st.write(err)

elif data_option == 'Numpy3D':
    st.title("Import Seismic As Numpy Array")

    if 'filename' not in st.session_state:
        st.session_state.filename = ''
    np_text = 'Please pass here the whole path,'
    filename = st.text_input(np_text, st.session_state.filename)

    st.session_state.filename = filename
    st.write('The selected file is: ', filename)
    if filename:
        try:
            seismic = Numpy3D(filename)
            st.session_state.seismic_type = "3D"
            seismic.make_axis_devisable_by(4)
            st.session_state.seismic = seismic

            viz = VISUALIZATION(seismic, st.session_state.seismic_type)
            viz.viz_data_3d(seismic, is_fspect=False)

        except RuntimeError as err: 
            st.write(err)
else:
    st.error("StreamLit Error.")
   
sidebar()