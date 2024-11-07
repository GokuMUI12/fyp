from blocks import sidebar, crop_and_load_volume, save_data_form
from visualizations import VISUALIZATION
from classes import Numpy3D
from utils import find_files_in_directory, predict_with_mask, std_mean_normalization
import streamlit as st
import keras.src.models.functional
from keras.models import load_model
# from model.unet3 import *
from utils import cross_entropy_balanced

st.markdown("# 3D Fault Segmentation using 3D U-Net model")

if "seismic" not in st.session_state:
        st.error("Please import 3D seismic first")
        st.stop()

seismic = st.session_state.seismic
seismic_type = st.session_state.seismic_type

module_name = 'Faults Segmentation on 3D data'

# Initialize state

if module_name not in st.session_state:
    st.session_state[module_name] = {"numpy_data" : None,
        "numpy_result" : None, "is_predicted" : False, 'cropped_info' : None , \
        "step1_status" : None, "step2_status" : None, "step3_status" : None} 

with st.expander("Load the data into RAM"):
    st.info("Here is your original seismic on your disk.")

    step1_viz = VISUALIZATION(seismic, st.session_state.seismic_type)
    step1_viz.viz_data_3d(seismic, key=10, is_fspect=False)

    st.info("Use sliders to crop the volume in 3 dimensions.")

    st.session_state[module_name]['numpy_data'], st.session_state[module_name]['cropped_info'] = \
        crop_and_load_volume(seismic, st.session_state[module_name]['numpy_data'], \
        st.session_state[module_name]['cropped_info'])

    if st.session_state[module_name]['numpy_data'] is not None:    
        st.info("The cropped data.")
        step1_crop_viz = VISUALIZATION(st.session_state[module_name]['numpy_data'] , st.session_state.seismic_type)
        step1_crop_viz.viz_data_3d(st.session_state[module_name]['numpy_data'] , key = 20, is_fspect=False)

with st.expander("Now you can predict faults"):

    inference_form = st.form("Inference")
    weight_file_list = sorted(find_files_in_directory('model/', '.keras'))
    weight_selected = inference_form.selectbox(
        'Click Submit',
        (weight_file_list))

    if (len(weight_file_list) == 0):
        st.error('''There is no weights in the model folder. 
        Please download the pretrained models from [GITHUB_Link]
        and place them here /model.
        ''')
    inference_submit = inference_form.form_submit_button("Submit")
    if inference_submit:

        #TODO may be two times memory allocation
        numpy_data = st.session_state[module_name]['numpy_data'].get_cube()

        loaded_model = load_model(f"model/{weight_selected}", 
                                  custom_objects={'cross_entropy_balanced': cross_entropy_balanced})
        
        print("Successfully Loaded the Model.")
        numpy_data = std_mean_normalization(numpy_data)
        print (numpy_data.shape)
        _ , predict = predict_with_mask(loaded_model, numpy_data.T)
        st.session_state[module_name]['numpy_result'] = Numpy3D(100*predict.T)
        st.session_state[module_name]['is_predicted'] = True
    if st.session_state[module_name]['is_predicted']:
        step2_viz = VISUALIZATION(st.session_state[module_name]['numpy_data']  , st.session_state.seismic_type)
        step2_viz.viz_sidebyside_3d(st.session_state[module_name]['numpy_data'], st.session_state[module_name]['numpy_result'] , key=30)


with st.expander("Save your predictions"):
    st.write("")
    if st.session_state[module_name]['is_predicted']:
        st.session_state[module_name]['step3_status'] = save_data_form(st.session_state[module_name], seismic, st.session_state[module_name]['numpy_result'].get_cube(), st.session_state[module_name]['step3_status'])
        st.info(st.session_state[module_name]['step3_status'])


sidebar()