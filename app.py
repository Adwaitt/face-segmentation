import config as cfg

import streamlit as st
import skimage

import matplotlib
import matplotlib.pyplot as plt

import io

from get_models import get_models

from inference import get_output

matplotlib.use("Agg")
plt.rcParams.update({"font.family": "DejaVu Sans"})

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

get_models()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = skimage.io.imread(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run"):
        st.write("Processing...")
        output_image = get_output(image)
        fig, ax = plt.subplots()
        ax.imshow(output_image, cmap="gray")
        ax.axis("off")
        plt.tick_params(
            axis='both',       
            which='both',      
            bottom=False,      
            top=False,         
            left=False,        
            right=False,       
            labelbottom=False, 
            labelleft=False
        )
        st.pyplot(fig)

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format = "png", transparent = True)
        img_buffer.seek(0)

        st.download_button(
            label="Save Output",
            data=img_buffer,
            file_name="processed_image.png",
            mime="image/png"
        )

