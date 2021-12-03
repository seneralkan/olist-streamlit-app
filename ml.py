import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

@st.cache(allow_output_mutation=True)
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model

def load_raw_data():
    df = pd.read_csv('./data/olist_merge.csv')
    return df

def create_features():
    df = load_raw_data()
    col2drop = ['order_purchase_timestamp', 'order_id', 'customer_id',  'order_approved_at', 'order_estimated_delivery_date',
             'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date',
             'order_item_id', 'product_id', 'seller_id', 'shipping_limit_date', 
             'price', 'review_id', 'review_comment_title', 'review_comment_message', 'review_creation_date',
             'review_answer_timestamp', 'product_name_lenght', 'product_description_lenght', 'product_photos_qty',
             'customer_unique_id', 'customer_zip_code_prefix',
             'seller_zip_code_prefix', 'seller_city', 'seller_state', 'state', 'c_lat_x',
             'c_lng_x', 'city', 'c_lat_y', 'c_lng_y']

    df.drop(col2drop, axis=1, inplace=True)

    # Missing value for product columns
    df["product_weight_g"] = df["product_weight_g"].fillna(df["product_weight_g"].median())
    df["product_length_cm"] = df["product_length_cm"].fillna(df["product_length_cm"].median())
    df["product_height_cm"] = df["product_height_cm"].fillna(df["product_height_cm"].median())
    df["product_width_cm"] = df["product_width_cm"].fillna(df["product_width_cm"].median())
    df["freight_value"] = df["freight_value"].fillna(df["freight_value"].median())
    df["payment_sequential"] = df["payment_sequential"].fillna(df["payment_sequential"].median())
    # Missing value for review_score
    df["review_score"] = df["review_score"].fillna(df["review_score"].median())
    # Missing value for payment_type
    df["payment_type"] = df["payment_type"].fillna('credit_card')
    # Missing value for payment_installments
    df["payment_installments"] = df["payment_installments"].fillna(df["payment_installments"].median())
    # Missing value for product_category_name
    df["product_category_name"] = df["product_category_name"].fillna("None")
    # Missing value fro payment_values
    df['payment_value'] = df['payment_value'].fillna(df['payment_value'].median())

    Y = df['payment_value']
    df.drop(["payment_value"], axis=1, inplace=True)
    X = df 
    return X, Y

X, Y = create_features()

def user_input_features():
    
    cat_= X["product_category_name"].drop_duplicates()
    CAT_NAME = st.selectbox("What is the product category?", cat_)

    state_ = X["customer_state"].drop_duplicates()
    STATE = st.selectbox("Which city is customer located ?", state_)

    city_ = X["customer_city"].drop_duplicates()
    CITY = st.selectbox("Which city is customer located ?", city_)
    
    pay_type_ = X["payment_type"].drop_duplicates()
    PAY_TYPE = st.selectbox("What is the payment type?", pay_type_)
    
    order_stat_ = X["order_status"].drop_duplicates()
    ORDER_STAT=st.selectbox("Which city is customer located ?", order_stat_)

    FRE_VALU = st.sidebar.slider('FREIGHT VALUE', X.freight_value.min(), X.freight_value.max(), X.freight_value.mean())
    PAY_SQ = st.sidebar.slider('PAYMENT SEQUENTIAL', X.payment_sequential.min(), X.payment_sequential.max(), X.payment_sequential.mean())
    PAY_INS = st.sidebar.slider('PAYMENT INSTALLMENTS', X.payment_installments.min(), X.payment_installments.max(), X.payment_installments.mean())
    REVIEW_SCORE = st.sidebar.slider('REVIEW SCORE', X.review_score.min(), X.review_score.max(), X.review_score.mean())
    
    PRO_WEI = st.sidebar.slider('PRODUCT WEIGHT (G)', X.product_weight_g.min(), X.product_weight_g.max(), X.product_weight_g.mean())
    PRO_LEN = st.sidebar.slider('PRODUCT LENGTH (CM)', X.product_length_cm.min(), X.product_length_cm.max(), X.product_length_cm.mean())
    PRO_HEI = st.sidebar.slider('PRODUCT HEIGHT (CM)', X.product_height_cm.min(), X.product_height_cm.max(), X.product_height_cm.mean())
    PRO_WID = st.sidebar.slider('PRODUCT WIDTH (CM)', X.product_width_cm.min(), X.product_width_cm.max(), X.product_width_cm.mean())
    

    data = {
            'product_category_name': CAT_NAME,
            'customer_state': STATE,
            'customer_city': CITY,
            'payment_type': PAY_TYPE,
            'order_status': ORDER_STAT,
            'freight_value': FRE_VALU,
            'payment_sequential': PAY_SQ,
            'payment_installments': PAY_INS,
            'review_score': REVIEW_SCORE,
            'product_weight_g': PRO_WEI,
            'product_length_cm': PRO_LEN,
            'product_height_cm': PRO_HEI,
            'product_width_cm': PRO_WID,
            }
    features = pd.DataFrame(data, index=[0])

    return features

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    for col in binary_col:
        dataframe[col] = labelencoder.fit_transform(dataframe[col])
    return dataframe


def run_ml():
    st.sidebar.header('Specify Input Parameters')
    df = user_input_features()

    st.header('Specified Input parameters')
    st.write(df)
    
    st.write('---')

    if st.button("Predict"):
        l_col = ['product_category_name', 'customer_city', 'customer_state', 'payment_type', 'order_status']
        df_ = df.copy()
        df__ = label_encoder(df_, l_col)
        model = load_model("model/rf_model.pkl")
        prediction = model.predict(df__)
        st.header('Prediction of Payment Value')
        st.write("Purchased:${}".format(prediction[0]))
        st.write('---')
        st.balloons()