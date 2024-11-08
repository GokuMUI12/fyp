import streamlit as st
import numpy as np
from PIL import Image
from utils import save_to_numpy, save_to_segy_3d
from classes import Numpy3D
import os


def sidebar():
    with st.sidebar:
        st.markdown("""
        """.format(st.session_state.seismic_type if 'seismic_type' in st.session_state else "--", \
            os.path.basename(st.session_state.filename) if 'filename' in st.session_state else "--"))
        st.markdown("""
        &nbsp;
        """)
 

def crop_and_load_volume(data, converted_to_numpy3d, cropped_info):
    with st.form("Cropping"):
        col1, col2, col3 = st.columns(3)
        inlines_indx = col1.slider( 'Select a range for Inlines',
        0, data.get_n_ilines()-1, (0, data.get_n_ilines())) 

        xlines_indx = col2.slider( 'Select a range for Xlines',
        0, data.get_n_xlines()-1, (0, data.get_n_xlines())) 
        
        zslice_indx = col3.slider( 'Select a range for Zslice',
        0, data.get_n_zslices()-1, (0, data.get_n_zslices())) 
        
        submitted = st.form_submit_button("Submit")
        if submitted:
            cropped_info = np.array([[inlines_indx[0], inlines_indx[1]], \
                [xlines_indx[0], xlines_indx[1]], \
                    [zslice_indx[0], zslice_indx[1]]])
            np_data = data.cropped_numpy(inlines_indx[0], inlines_indx[1], \
            xlines_indx[0], xlines_indx[1],\
                zslice_indx[0], zslice_indx[1])
            converted_to_numpy3d = Numpy3D(np_data)
            col1, col2, col3 = st.columns(3)
            col1.info(f"Number of Inlines [{inlines_indx[1]-inlines_indx[0]}]")
            col2.info(f"Number of Xlines [{xlines_indx[1] - xlines_indx[0]}]")
            col3.info(f"Time [{zslice_indx[1]-zslice_indx[0]}]")
            st.success('Volume is loaded')
    return converted_to_numpy3d, cropped_info

def save_data_form(session_state, seismic, numpy_data, status):
    with st.form("save_data"):
        col1, col2, col3 = st.columns(3)
        path = col1.text_input("Path to Folder")
        file_name = col2.text_input("File Name")
        file_format = col3.radio( "What format? ", ('SEGY', 'NUMPY_SUBSET'))
        submitted = st.form_submit_button("Save")
        if submitted:
            with st.spinner('Wait... Exporting the volume'):
                if file_format == "SEGY":
                    if seismic.get_str_format() == "SEGY":
                        if seismic.get_str_dim() == "2D":
                            print("Shut up. ")
                        else:
                            save_to_segy_3d(seismic, path+file_name, numpy_data, session_state)
                        status = "Saving SEGY complete"
                    else:
                        status = "Error: you can not save a SEGY file since the initial seismic was not SEGY."
                else:
                    save_to_numpy(path+file_name, numpy_data)
                    status = "Saving NUMPY complete"
                    # option = st.radio( "Option", ('Save subset', 'Save in original dimensions - It will create the volume in RAM. Are you sure?'))
                    # if st.form_submit_button("Save "):
                    #     if option == 'Save subset':
                    #         status = None
                    #         save_to_numpy(path+file_name, numpy_data)
                    #     else:
                    #         status = "Save in original dimensions"
    return status