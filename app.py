from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import random

app = Flask(__name__)


# دالة لتفادي أخطاء التصنيفات الجديدة
def safe_label_transform(encoder, values):
    known_classes = set(encoder.classes_)
    modified_values = [v if v in known_classes else 'Unknown' for v in values]

    if 'Unknown' not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, 'Unknown')

    return encoder.transform(modified_values)


# تحميل النماذج
with open('one_hot_encoder.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# تحميل بيانات الشركات والموديلات
data = pd.read_csv('data/car_price_prediction.csv')
manufacturer_models = data.groupby('Manufacturer')['Model'].unique().apply(list).to_dict()
manufacturers = sorted(manufacturer_models.keys())
years = sorted(data['Prod. year'].unique(), reverse=True)


@app.route('/')
def start():
    return render_template('start.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    form_data = None

    if request.method == 'POST':
        form_data = {
            'Manufacturer': request.form['manufacturer'],
            'Model': request.form['model'],
            'Prod_year': int(request.form['prod_year']),
            'Category': request.form['category'],
            'Leather interior': request.form['leather'],
            'Fuel type': request.form['fuel'],
            'Engine volume': float(request.form['engine']),
            'Mileage': int(request.form['mileage']),
            'Cylinders': float(request.form['cylinders']),
            'Gear box type': request.form['gear'],
            'Drive wheels': request.form['drive'],
            'Wheel': request.form['wheel'],
            'Color': request.form['color'],
            'Airbags': int(request.form['airbags'])
        }

        df = pd.DataFrame([form_data])
        df['Age'] = datetime.now().year - df['Prod_year']
        df.drop(columns=['Prod_year'], inplace=True)

        one_hot_cols = ['Leather interior', 'Gear box type', 'Drive wheels', 'Wheel']
        encoded = one_hot_encoder.transform(df[one_hot_cols])
        encoded_df = pd.DataFrame(encoded, columns=one_hot_encoder.get_feature_names_out(one_hot_cols))
        df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
        df.drop(columns=one_hot_cols, inplace=True)

        label_cols = ['Manufacturer', 'Model', 'Category', 'Fuel type', 'Color']
        for col in label_cols:
            df[col] = safe_label_transform(label_encoders[col], df[col])

        prediction = model.predict(df)[0]
        ref = f"{random.randint(1, 999999):06}"

        return render_template('result.html', prediction=round(prediction, 2), form_data=form_data, now=datetime.now(),
                               ref=ref)

    return render_template('index.html', prediction=prediction, manufacturers=manufacturers,
                           manufacturer_models=manufacturer_models, years=years)


@app.route('/Info')
def Info():
    return render_template('Info.html')


if __name__ == '__main__':
    app.run(debug=True)
