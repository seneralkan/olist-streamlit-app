import streamlit as st
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium 
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import time

@st.cache
def load_data(geolocation=False, drop=True):
   
    if geolocation:
        df_geo= pd.read_csv('./data/olist_geolocation_dataset.csv')
        df_geo.rename(columns={'geolocation_lng': 'lng',
                    'geolocation_lat': 'lat'}, inplace=True)

        return df_geo
    else:
        
        if drop:
            df = pd.read_csv('./data/olist_merge.csv')
            col2drop = ['order_id', 'customer_id', 'order_status', 'order_approved_at', 'order_estimated_delivery_date',
                        'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date',
                        'order_item_id', 'product_id', 'seller_id', 'shipping_limit_date',
                        'payment_sequential',
                        'price', 'review_id', 'review_comment_title', 'review_comment_message', 'review_creation_date',
                        'review_answer_timestamp', 'product_name_lenght', 'product_description_lenght', 'product_photos_qty',
                        'customer_unique_id', 'customer_zip_code_prefix',
                        'seller_zip_code_prefix', 'seller_city', 'seller_state', 'state',  
                        'city', 'c_lat_y', 'c_lng_y', 'product_weight_g', 'product_length_cm', 'product_height_cm',
                        'product_width_cm']
            df.rename(columns={'c_lng_x': 'lng',
                        'c_lat_x': 'lat'}, inplace=True)
            

            df.drop(col2drop, axis=1, inplace=True)
            df.drop(['lng','lat'], axis=1, inplace=True)
            return df

        else:
            df = pd.read_csv('./data/olist_merge.csv')
            return df

def display_map(df, col):
    px.set_mapbox_access_token('pk.eyJ1IjoiYWxrYW5zZSIsImEiOiJja3ZzZHR0cnMyanZmMzJrbGd2dGo2MmxzIn0.iu3GHL-EipWtoR7b3FCkyg')
    fig = px.scatter_mapbox(df, lat= 'lat', lon='lng', color = col, zoom=2)
    return fig

def heat_map(df):
    locs = list(zip(df.lat, df.lng))
    m = folium.Map([-22, -43], tiles='cartodbpositron',  zoom_start=10)
    HeatMap(locs).add_to(m)
    #  return st.markdown(m._repr_html_(), unsafe_allow_html=True)
    folium_static(m)

def cat_summary(dataframe, col_name, plot=False):
    df = pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)})
    if plot:
         with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(7, 5))
            ax = sns.countplot(x=dataframe[col_name], data=dataframe)
            st.pyplot(f)

    return df

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.50, 0.75, 0.90, 0.95, 0.99]

    if plot:
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(7, 5))
            ax = sns.histplot(x=dataframe[numerical_col], data=dataframe)
            plt.xlabel(numerical_col)
            plt.title(numerical_col)
            st.pyplot(f)
    return dataframe[numerical_col].describe(quantiles).T

def edaFunc():

    subMenu = st.sidebar.selectbox("Selection", ["EDA", "Visualization","Heatmap", "Correlation Matrix"])
    with st.spinner("Loading the Dataset"):
        df = load_data()
        df_geo =  load_data(geolocation=True)
        df_ = load_data(geolocation=False, drop=False)

    # EDA Sub Menu will show the basic data analysis for the dataset
    # It contains Descriptive Statistics

    if subMenu == "EDA":
        st.subheader("Exploratory Data Analysis")
        st.dataframe(df.head(20))

        with st.expander("Descriptive Statistics"):
            st.dataframe(df.describe().T)
        
        with st.expander("Payment Type Distributions"):
            st.dataframe(cat_summary(df, "payment_type"))
        
        with st.expander("Product Category Distributions"):
            st.dataframe(cat_summary(df, "product_category_name"))

        with st.expander("Customer State Distributions"):
            st.dataframe(cat_summary(df, "customer_state"))

        with st.expander("Freight Value Distributions"):
            st.dataframe(num_summary(df, "freight_value"))

        with st.expander("Payment Distributions"):
            st.dataframe(num_summary(df, "payment_value"))

    elif subMenu == "Visualization":
        st.subheader("Visualization")
        
        with st.expander("Payment Type Distributions"):
            st.dataframe(cat_summary(df, "payment_type", plot=True))
        
        with st.expander("Product Category Distributions"):
            st.dataframe(cat_summary(df, "product_category_name", plot=True))

        with st.expander("Customer State Distributions"):
            st.dataframe(cat_summary(df, "customer_state", plot=True))
        
        with st.expander("Customer City Distributions"):
            st.dataframe(cat_summary(df, "payment_type", plot=True))

        with st.expander("Freight Value Distributions"):
            st.dataframe(num_summary(df, "freight_value", plot=True))

        with st.expander("Payment Distributions"):
            st.dataframe(num_summary(df, "payment_value", plot=True))

    elif subMenu == "Heatmap":
        with st.spinner("Wait for Map Visualisations"):
            time.sleep(5)
        st.subheader('Displaying State Based Order Map')
        st.plotly_chart(display_map(df_geo, "geolocation_state"))
        # st.subheader("Order Heatmap")
        # st.set_option('deprecation.showPyplotGlobalUse', False)
        # st.pyplot(heat_map(df_geo))      

    elif subMenu == "Correlation Matrix":
        st.dataframe(df_.head())
        st.subheader('Correlation Matrix')
        corr = df_.corr()
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True

        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(7, 5))
            ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
        st.pyplot(f)