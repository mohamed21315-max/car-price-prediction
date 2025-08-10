
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pickle

# تحميل البيانات
df = pd.read_csv("C:/Users/mohda/OneDrive/Desktop/car price prediction/car_price_prediction.csv")


# تنظيف البيانات
df = df[df['Price'] > 100]
df = df[df['Engine volume'] > 0]
df = df[df['Mileage'] < 1_000_000]

# المعالجة
df['Engine volume'] = df['Engine volume'].astype(str).str.replace('Turbo', '', regex=False)
df['Engine volume'] = pd.to_numeric(df['Engine volume'], errors='coerce').fillna(0)
df['Mileage'] = df['Mileage'].astype(str).str.replace('km', '', regex=False).str.replace(',', '', regex=False)
df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce').fillna(0)
df['Age'] = 2025 - df['Prod. year']

# حذف الأعمدة غير المستخدمة
df.drop(columns=["ID", "Prod. year", "Doors", "Levy"], errors="ignore", inplace=True)

# الترميز
label_cols = ['Manufacturer', 'Model', 'Category', 'Fuel type', 'Color']
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

one_hot_cols = ['Leather interior', 'Gear box type', 'Drive wheels', 'Wheel']
one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # scikit-learn 1.7+
encoded = one_hot_encoder.fit_transform(df[one_hot_cols])
encoded_df = pd.DataFrame(encoded, columns=one_hot_encoder.get_feature_names_out(one_hot_cols))

# دمج البيانات
df_model = pd.concat([df.drop(columns=one_hot_cols).reset_index(drop=True),
                      encoded_df.reset_index(drop=True)], axis=1)

# التقييس
numeric_cols = ['Engine volume', 'Mileage', 'Cylinders', 'Airbags', 'Age']
scaler = StandardScaler()
df_model[numeric_cols] = scaler.fit_transform(df_model[numeric_cols])

# الفصل بين الميزات والهدف
X = df_model.drop(columns=['Price'])
y = df_model['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب النموذج
model = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# حفظ النماذج
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
with open("one_hot_encoder.pkl", "wb") as f:
    pickle.dump(one_hot_encoder, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ تم تدريب النموذج وحفظه بنجاح.")
