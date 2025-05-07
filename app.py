#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
import streamlit as st
# intro
st.set_page_config(layout="wide")

import base64
from PIL import Image
import numpy as np
import requests
import pandas as pd
import cv2 
from apsislipiscanv1.ocr import ImageOCR
from apsislipiscanv1.visualize import  draw_word_polys
from apsisocr.utils import correctPadding
#--------------------------------------------------
# main
#--------------------------------------------------


@st.cache_resource
def load_model():
    ocr=ImageOCR()
    return ocr

ocr=load_model()

def main():
    
    st.title("Apsis-Lipiscanv1 is an ocr system for Printed Bangla Documents")
    
    
    # For newline
    st.write("\n")
    
    # File selection
    st.title("Document selection")
    # Choose your own image
    uploaded_file = st.file_uploader("Upload files", type=["png", "jpeg", "jpg"])
    
    # For newline
    st.write("\n")
    # Instructions
    st.markdown("*click on the top-right corner of an image to enlarge it!*")
    # Set the columns
    cols = st.columns((1,1,1,1))
    cols[0].subheader("Input Image")
    cols[1].subheader("Processed Image")
    cols[2].subheader("Word Detection")
    cols[3].subheader("Text and Reading Order")
    
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        arr = np.array(image)
        cols[0].image(arr)
        with st.spinner('Executing OCR'):
            output=ocr(arr)
        
        cols[1].image(output["rotation"]["rotated_image"])
        # word-detection
        word_det_viz=draw_word_polys(output["rotation"]["rotated_image"],[entry["poly"] for entry in output["words"]])
        cols[2].image(word_det_viz)
        
        # recognition and rdo
        df=pd.DataFrame(output["words"])
        df=df[['text','line_num','word_num']]
        cols[3].dataframe(df)
        lines=[]
        _lines=sorted([_line for _line in df.line_num.unique()])
        for line in _lines:
            ldf=df.loc[df.line_num==line]
            ldf.reset_index(drop=True,inplace=True)
            ldf=ldf.sort_values('word_num')
            _ltext=''
            for idx in range(len(ldf)):
                text=ldf.iloc[idx,0]
                _ltext+=' '+text
            lines.append(_ltext)
        text="\n".join(lines)
        st.text_area("Plain text", value=text,height=400)
        
        # Word Analysis
        st.title("Word Analysis")
        crops=ocr.detector.get_crops(output["rotation"]["rotated_image"],[entry["poly"] for entry in output["words"]])
        crops=[correctPadding(crop,(128,1024)) for crop in crops]
        crops=[ crop[:,:pad_w] for (crop,pad_w) in crops]
        data=[{"image": crop,"text":text} for crop,text in zip(crops,[entry["text"] for entry in output["words"]])]
        
        # Custom CSS to center the table
        st.markdown(
            """
            <style>
            .centered-table {
                display: flex;
                justify-content: center;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Display the table in the center
        st.markdown('<div class="centered-table">', unsafe_allow_html=True)

        # Iterate over the data in chunks of 5 to create rows
        for i in range(0, len(data), 10):
            cols = st.columns(10)  # Create 5 columns for each row
            
            for j in range(10):
                if i + j < len(data):  # Ensure we don't go out of bounds
                    with cols[j]:  # Access the j-th column in the current row
                        st.image(data[i + j]["image"], caption=data[i + j]["text"], use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        
                
if __name__ == '__main__':  
    main()