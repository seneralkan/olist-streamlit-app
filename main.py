import streamlit as st
from PIL import Image
from eda import edaFunc
from about import aboutFunc
from ml import run_ml
import os

def load_img(img_file):
	image = Image.open(os.path.join(img_file))
	return image

image = load_img("img/utopia.jpg")
# image = Image.open('../img/utopia.jpg')
st.image(image, width = 750)
# Title
st.title(' Olist (Brazillian E-Commerce) EDA & Sales Prediction')

st.markdown("""
This app performs exploratory data analysis and LightGBM sales prediction for Brazillian E-Commerce stats data!
* **Data source:** [Kaggle.com](https://www.kaggle.com/olistbr/brazilian-ecommerce).
* **Image Source: ** [Valentin Tkach - Behance](https://www.behance.net/depingo/projects)
""")

def main():
    menu = ["Home", "About", "Exploratory Data Analysis", "ML Prediction"]

    select = st.sidebar.selectbox("Menu", menu)

    if select == "Exploratory Data Analysis":
        edaFunc()
    if select == "About":
        aboutFunc()
    if select == "ML Prediction":
        run_ml()

if __name__ == '__main__':
	main()