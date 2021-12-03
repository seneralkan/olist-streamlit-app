import streamlit as st

def aboutFunc():
    st.subheader("About")
    st.markdown("""
    * **Author:** Sener Alkan
    * **Kaggle:** [Sener Alkan - Kaggle](https://www.kaggle.com/seneralkan)
    * **Github:** [Sener Alkan - Github](https://www.github.com/seneralkan)

    You can find the codes my @Github page.

    * ** About Dataset **

    This is a Brazilian ecommerce public dataset of orders made at Olist Store. 
    The dataset has information of 100k orders from 2016 to 2018 made at multiple marketplaces in Brazil.
    Its features allows viewing an order from multiple dimensions: from order status, price, payment and freight performance to customer location, product attributes and finally reviews written by customers. 
    Olist also released a geolocation dataset that relates Brazilian zip codes to lat/lng coordinates.

    * ** EDA Section **

    In EDA section, you can see descriptive statistics of the dataset, visualization of the EDA and location based distribution of the orders with heatmap

    * ** Customer Payment Prediction **
    
    In the sidebar, you can choose the range for the dataset, and you can easily predict customer payment :)
    """)
    st.balloons()