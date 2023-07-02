import tempfile
from glob import glob
from pathlib import Path
import random

import cv2
import numpy as np

import streamlit as st

from converter import Mc2RealConverter


@st.cache_resource
def get_model():
    return Mc2RealConverter()


def get_uploaded_image(file):
    if file is None:
        return None
    
    temp_dir_path = Path(tempfile.mkdtemp())
    path = str(temp_dir_path / file.name)
    with open(path, 'wb') as f:
        f.write(file.read())
    image = cv2.imread(path)
    return image[..., ::-1]


def crop(img, dim):
    width, height = img.shape[1], img.shape[0]
    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 

    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img


def crop_and_scale(img):
    crop_size = min(img.shape[0], img.shape[1])
    img = crop(img, (crop_size, crop_size))
    return img


def preprocess_image(mc_image):
    return mc_image.astype(np.float32) / 255


def get_random_image():
    paths = list(glob('data/sample_images/*.png'))
    path = random.choice(paths)
    image = cv2.imread(str(path))[..., ::-1]
    return image


# global state
if 'mc_image' not in st.session_state:
    st.session_state.mc_image = get_random_image()


# sidebar
st.sidebar.title('Options')
blend_factor = st.sidebar.slider('mc-real blend factor', value=0.5, min_value=0.0, max_value=1.0, format='%f')
sr_steps = st.sidebar.number_input('Super-resolution steps', value=200, min_value=1, max_value=500, step=50, format='%d')

# main content
st.title('Minecraft to real converter')

with st.expander('Input Minecraft image', expanded=True):

    mc_file = st.file_uploader('Upload')

    if mc_file:
        st.session_state.mc_image = crop_and_scale(get_uploaded_image(mc_file))
    
    if st.button('Random'):
        st.session_state.mc_image = crop_and_scale(get_random_image())

convert = st.button('Convert', use_container_width=True)
        
col_mc, col_real = st.columns(2)

model = get_model()
real_image = None

with col_mc:
    st.image(st.session_state.mc_image, use_column_width='always')

with col_real:
    if convert:
        mc_image_pre = preprocess_image(st.session_state.mc_image)
        with st.spinner('converting'):
            print(blend_factor, sr_steps)
            real_image = model(mc_image_pre, blend_factor=1-blend_factor, sr_inference_steps=sr_steps)
        st.image(real_image, use_column_width='always')

# if model.last_segmentator_mask is not None:
#     with st.expander('Details'):
#         st.image(model.last_segmentator_mask, caption='Segmentator mask')
#         st.image(model.last_raw_generator_mask, caption='Raw generator mask')
#         st.image(model.last_filtered_generator_mask, caption='Filtered generator mask')
#         st.image(model.last_img_without_opt, caption='Image without optimization')
#         if model.last_img_with_opt is not None:
#             st.image(model.last_img_with_opt, caption='Image with optimization')
