import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics as mt
import plotly.express as px
import streamlit as st
import random
from PIL import Image
import altair as alt
from htbuilder import HtmlElement, div, hr, a, p, img, styles
from htbuilder.units import percent, px

def main():
    def _max_width_():
        max_width_str = f"max-width: 1000px;"
        st.markdown(
            f"""
        <style>
        .reportview-container .main .block-container{{
            {max_width_str}
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )


    # Hide the Streamlit header and footer
    def hide_header_footer():
        hide_streamlit_style = """
                    <style>
                    footer {visibility: hidden;}
                    </style>
                    """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # increases the width of the text and tables/figures
    _max_width_()

    # hide the footer
    hide_header_footer()

image_nyu = Image.open('nyu.png')
st.image(image_nyu, width=100)

st.title("Food Delivery Times 🍜")

# navigation dropdown

st.sidebar.header("Dashboard")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox('🔎 Select Page',['Introduction','Visualization','Prediction'])

# Dropdown menu for selecting the dataset (currently only "Wine Quality" is available)
select_dataset =  st.sidebar.selectbox('💾 Select Dataset',["Food Delivery Times"])

# Load the wine quality dataset
df = pd.read_csv("deliverytime.csv")

# Dropdown menu for selecting which variable from the dataset to predict
list_variables = df.columns
select_variable =  st.sidebar.selectbox('🎯 Select Variable to Predict',list_variables)

# Introduction page content
if app_mode == 'Introduction':
    # Display dataset details
    st.markdown("### 00 - Show Dataset")

    # Split the page into 10 columns to display information about each wine quality variable
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)

    # Descriptions for each variable in the dataset
    # ... [The code here provides descriptions for each wine quality variable]

    # Allow users to view either the top or bottom rows of the dataset
    num = st.number_input('No. of Rows', 5, 10)
    head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))
    if head == 'Head':
        st.dataframe(df.head(num))
    else:
        st.dataframe(df.tail(num))

    # Display the shape (number of rows and columns) of the dataset
    st.markdown("Number of rows and columns helps us to determine how large the dataset is.")
    st.text('(Rows,Columns)')
    st.write(df.shape)

    st.markdown("### 01 - Description")
    st.dataframe(df.describe())



    st.markdown("### 02 - Missing Values")
    st.markdown("Missing values are known as null or NaN values. Missing data tends to **introduce bias that leads to misleading results.**")
    dfnull = df.isnull().sum()/len(df)*100
    totalmiss = dfnull.sum().round(2)
    st.write("Percentage of total missing values:",totalmiss)
    st.write(dfnull)
    if totalmiss <= 30:
        st.success("Looks good! as we have less then 30 percent of missing values.")
    else:
        st.warning("Poor data quality due to greater than 30 percent of missing value.")
        st.markdown(" > Theoretically, 25 to 30 percent is the maximum missing values are allowed, there's no hard and fast rule to decide this threshold. It can vary from problem to problem.")

    st.markdown("### 03 - Completeness")
    st.markdown(" Completeness is defined as the ratio of non-missing values to total records in dataset.")
    # st.write("Total data length:", len(df))
    nonmissing = (df.notnull().sum().round(2))
    completeness= round(sum(nonmissing)/len(df),2)
    st.write("Completeness ratio:",completeness)
    st.write(nonmissing)
    if completeness >= 0.80:
        st.success("Looks good! as we have completeness ratio greater than 0.85.")

    else:
        st.success("Poor data quality due to low completeness ratio( less than 0.85).")
