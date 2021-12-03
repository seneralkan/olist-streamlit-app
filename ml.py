import streamlit as st
from sqlalchemy import create_engine
import pandas as pd
from eda import load_data
import datetime
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os
@st.cache
def create_features():
    df = load_data(drop=False)
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["order_delivered_carrier_date"] = pd.to_datetime(df["order_delivered_carrier_date"])
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"])
    df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"])
    df["order_approved_at"] = pd.to_datetime(df["order_approved_at"])
    
    df['month'] = df.order_purchase_timestamp.dt.month
    df['day_of_month'] = df.order_purchase_timestamp.dt.day
    df['day_of_year'] = df.order_purchase_timestamp.dt.dayofyear
    df['week_of_year'] = df.order_purchase_timestamp.dt.weekofyear
    df['day_of_week'] = df.order_purchase_timestamp.dt.dayofweek
    df['year'] = df.order_purchase_timestamp.dt.year
    df["is_wknd"] = df.order_purchase_timestamp.dt.weekday // 4
    df['is_month_start'] = df.order_purchase_timestamp.dt.is_month_start.astype(int)
    df['is_month_end'] = df.order_purchase_timestamp.dt.is_month_end.astype(int)

    df['wd_estimated_delivery_time'] = df.order_estimated_delivery_date - df.order_approved_at

    df['wd_actual_delivery_time'] = df.order_delivered_customer_date - df.order_approved_at

    # Calculate the time between the actual and estimated delivery date.
    # If negative was delivered early, if positive was delivered late.
    df['wd_delivery_time_delta'] = df.wd_actual_delivery_time - df.wd_estimated_delivery_time

    # Calculate the time between the actual and estimated delivery date.
    # If negative was delivered early, if positive was delivered late.
    df['is_late'] = df.order_delivered_customer_date > df.order_estimated_delivery_date

    # With that we can remove the unnecessary columns from the dataset
    cols2drop = ['order_id', 'customer_id', 'order_status', 'order_approved_at', 'order_estimated_delivery_date',
                 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date',
                 'order_item_id', 'product_id', 'seller_id', 'shipping_limit_date', 'freight_value',
                 'payment_sequential',
                 'price', 'review_id', 'review_comment_title', 'review_comment_message', 'review_creation_date',
                 'review_answer_timestamp', 'product_name_lenght', 'product_description_lenght', 'product_photos_qty',
                 'customer_unique_id', 'customer_zip_code_prefix',
                 'customer_state', 'seller_zip_code_prefix', 'seller_city', 'seller_state', 'state', 'c_lat_x',
                 'c_lng_x', 'city', 'c_lat_y', 'c_lng_y', 'product_weight_g', 'product_length_cm', 'product_height_cm',
                 'product_width_cm']
    df.drop(cols2drop, axis=1, inplace=True)

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

    # Missing value for extracted features
    df["wd_estimated_delivery_time"] = df["wd_estimated_delivery_time"].fillna(df["wd_estimated_delivery_time"].median())
    df["wd_actual_delivery_time"] = df["wd_actual_delivery_time"].fillna(df["wd_actual_delivery_time"].median())
    df["wd_delivery_time_delta"] = df["wd_delivery_time_delta"].fillna(df["wd_delivery_time_delta"].median())
    
    df['wd_actual_delivery_time'] = df['wd_actual_delivery_time'].dt.days
    df['wd_delivery_time_delta'] = df['wd_delivery_time_delta'].dt.days
    df['wd_estimated_delivery_time'] = df['wd_estimated_delivery_time'].dt.days

    Y = df["payment_value"]
    #df.drop(["payment_value"], axis=1, inplace=True)
    X = df
    return X, Y

def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model

def user_input_features():
    X, Y = create_features()
    X = X.loc[(X["order_purchase_timestamp"] > "2018-06-02 00:00:00"), :]
    min_date = X["order_purchase_timestamp"].min()
    max_date = X["order_purchase_timestamp"].max()

    ORDER_TIME = st.date_input("What is the order date", value = datetime.date(2018, 6, 3), min_value=min_date, max_value=max_date)
    
    PAY_INS = st.sidebar.slider('PAYMENT INSTALLMENTS', X.payment_installments.min(), X.payment_installments.max(), X.payment_installments.mean())
    REVIEW_SCORE = st.sidebar.slider('REVIEW SCORE', X.review_score.min(), X.review_score.max(), X.review_score.mean())

    WD_ACTUAL = st.number_input('ACTUAL DELIVERY TIME RANGE', X.wd_actual_delivery_time.min(), X.wd_actual_delivery_time.max())
    WD_DELIVERY_DELTA = st.number_input('ACTUAL - ESTIMATED DELIVERY RANGE', X.wd_delivery_time_delta.min(), X.wd_delivery_time_delta.max())
    WD_ESTIMATED = st.number_input('ESTIMATED DELIVERY TIME', X.wd_estimated_delivery_time.min(), X.wd_estimated_delivery_time.max())
    
    # WD_ACTUAL = st.sidebar.slider('ACTUAL DELIVERY TIME RANGE', X.wd_actual_delivery_time.min(), X.wd_actual_delivery_time.max(), X.wd_actual_delivery_time.mean())
    # WD_DELIVERY_DELTA = st.sidebar.slider('ACTUAL - ESTIMATED DELIVERY RANGE', X.wd_delivery_time_delta.min(), X.wd_delivery_time_delta.max(), X.wd_delivery_time_delta.mean())
    # WD_ESTIMATED = st.sidebar.slider('ESTIMATED DELIVERY TIME', X.wd_estimated_delivery_time.min(), X.wd_estimated_delivery_time.max(), X.wd_estimated_delivery_time.mean())
    
    
    # if ORDER_TIME < min_date:
    #     st.success('Order date: `%s`\n\nMinimum date:`%s`' % (ORDER_TIME, min_date))
    # else:
    #     st.error('Error: End date must fall after start date.')

    cat_= X["product_category_name"].drop_duplicates()
    CAT_NAME = st.selectbox("What is the product category?", cat_)
    
    city_ = X["customer_city"].drop_duplicates()
    CITY = st.selectbox("Which city is customer located ?", city_)

    IS_LATE = st.selectbox("The order delivered late or not?", ("True", "False"))
    
    pay_type_ = X["payment_type"].drop_duplicates()
    PAY_TYPE = st.selectbox("What is the payment type?", pay_type_)

    is_wknd_ = X["is_wknd"].drop_duplicates()
    IS_WKND = st.selectbox('IS WEEKEND', is_wknd_)

    is_month = X["is_month_start"].drop_duplicates()
    IS_MONTH_START = st.selectbox('IS MONTH START', is_month)

    is_month_end_ = X["is_month_end"].drop_duplicates()
    IS_MONTH_END = st.selectbox('IS MONTH END', is_month_end_)
    
    month_ = X["month"].drop_duplicates()
    MONTH = st.selectbox('MONTH', month_)
    
    d_month_ = X["day_of_month"].drop_duplicates()
    D_MONTH = st.selectbox('DAY OF MONTH', d_month_)

    d_year_ = X["day_of_year"].drop_duplicates()
    D_YEAR = st.selectbox('DAY OF YEAR', d_year_)

    w_year_ = X["week_of_year"].drop_duplicates()
    W_YEAR = st.selectbox('WEEK OF YEAR', w_year_)

    d_week_ = X["day_of_week"].drop_duplicates()
    D_WEEK = st.selectbox('DAY OF WEEK', d_week_)

    year_ = X["year"].drop_duplicates()
    YEAR = st.selectbox('YEAR',year_)
    
    

    data = {'order_purchase_timestamp': ORDER_TIME,
            'payment_type': PAY_TYPE,
            'payment_installments': PAY_INS,
            'review_score': REVIEW_SCORE,
            'product_category_name': CAT_NAME,
            'customer_city': CITY,
            'month': MONTH,
            'day_of_month': D_MONTH,
            'day_of_year': D_YEAR,
            'week_of_year': W_YEAR,
            'day_of_week': D_WEEK,
            'year': YEAR,
            'is_wknd': IS_WKND,
            'is_month_start': IS_MONTH_START,
            'is_month_end': IS_MONTH_END,
            'wd_estimated_delivery_time': WD_ESTIMATED,
            'wd_actual_delivery_time': WD_ACTUAL,
            'wd_delivery_time_delta': WD_DELIVERY_DELTA,
            'is_late': IS_LATE}
    features = pd.DataFrame(data, index=[0])

    return features

# def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

# def lag_features(dataframe, lags):
#     for lag in lags:
#         dataframe['payment_lag_' + str(lag)] = \
#             dataframe.groupby(
#                 ['review_score', 'product_category_name',
#                  'is_late'])['payment_value'].transform(
#                 lambda x: x.shift(lag)) + random_noise(dataframe)
#     return dataframe

# def roll_mean_features(dataframe, windows):
#     for window in windows:
#         dataframe['sales_roll_mean_' + str(window)] = \
#             dataframe.groupby(['review_score', 'product_category_name',
#                                'is_late'])['payment_value'].transform(
#                 lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
#                 dataframe)
#     return dataframe

# def ewm_features(dataframe, alphas, lags):
#     for alpha in alphas:
#         for lag in lags:
#             dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
#                 dataframe.groupby(['review_score', 'product_category_name',
#                                    'is_late'])['payment_value'] \
#                     .transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
#     return dataframe

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    for col in binary_col:
        dataframe[col] = labelencoder.fit_transform(dataframe[col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def run_ml():
    st.sidebar.header('Specify Input Parameters')
    df = user_input_features()
    st.header('Specified Input parameters')
    st.write(df)
    st.write('---')
    # Creating lag/shifted features
    # df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])
    # df = roll_mean_features(df, [365, 546])
    # alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
    # lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]
    # df = ewm_features(df, alphas, lags)
    # Encoding the user input values
    l_col = ['product_category_name', 'customer_city', 'is_late']
    df = label_encoder(df, l_col)
    ohe_cols = ['review_score', 'day_of_week', 'month', 'year', 'payment_type']
    df = one_hot_encoder(df, ohe_cols)
    cols = [col for col in df.columns if col not in ['order_purchase_timestamp', "year"]]
    X_test = df[cols]
    
    if st.button("Predict"):
        model = load_model("model/lgbm_tuned_model.pkl")
        prediction = model.predict(X_test)
        st.info("Predicted Purchase")
        st.write("Purchased:${}".format(prediction[0]))
        st.balloons()