from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the pre-trained model and encoders
with open('model1.pkl', 'rb') as file:
    data = pickle.load(file)

model=data['model']
encoder=data['label_encoder']
scaler=data['scaler']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    quantity=request.form['quantity']
    unit_price=request.form['unit_price']
    comp_1=request.form['comp_promotions']
    product_score=request.form['product_score']
    competitor_Price_Ratio=request.form['comp_price_ratio']

    new_data=[[quantity,unit_price,comp_1,product_score,competitor_Price_Ratio]]
    
    columns=['qty', 'unit_price', 'comp_1', 'product_score', 'comp_price_diff']

    df = pd.DataFrame(new_data,columns=columns)

    # Define data types for specific columns using a dictionary
    dtype_dict = {'qty':int, 'unit_price':float, 'comp_1':float, 'product_score':float, 'comp_price_diff':float}

    # Create the DataFrame with specified data types
    new_df = pd.DataFrame(new_data, columns=columns)

    # Convert specific columns to the specified data types
    new_df = new_df.astype(dtype_dict)

    for col in new_df.select_dtypes(include=['object']):
        new_df[col] = encoder[col].transform(new_df[col])

    scaled = scaler.transform(new_df)

    return render_template('home.html', result=str(model.predict(scaled)))
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    