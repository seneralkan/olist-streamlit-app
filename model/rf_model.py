import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

def load_data(geolocation=False, drop=True):
   
    if geolocation:
        df_geo= pd.read_csv('../data/olist_geolocation_dataset.csv')
        df_geo.rename(columns={'geolocation_lng': 'lng',
                    'geolocation_lat': 'lat'}, inplace=True)

        return df_geo
    else:
        
        if drop:
            df = pd.read_csv('../data/olist_merge.csv')
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
            df = pd.read_csv('../data/olist_merge.csv')
            return df

def create_features():
    df_ = load_data(drop=False)
    df = df_.loc[(df_["order_purchase_timestamp"] > "2018-06-02 00:00:00"), :]
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

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    for col in binary_col:
        dataframe[col] = labelencoder.fit_transform(dataframe[col])
    return dataframe

X, Y = create_features()

l_col = ['product_category_name', 'customer_city', 'customer_state', 'payment_type', 'order_status']
X = label_encoder(X, l_col)
model = RandomForestRegressor()
rf_fit= model.fit(X, Y)
pickle.dump(rf_fit, open("rf_model.pkl", 'wb'))