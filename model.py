from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- Data Loading & Preprocessing ---
df = pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',')
# Drop the last two empty columns
df = df.iloc[:, :-2]
# Convert negative numeric values to NaN and fill missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].applymap(lambda x: np.nan if x < 0 else x)
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
# Parse datetime and drop original columns
df['Datetime'] = pd.to_datetime(
    df['Date'].str.strip() + ' ' + df['Time'].str.strip(),
    format='%d/%m/%Y %H.%M.%S'
)
df.columns = df.columns.str.strip()
df.drop(columns=['Date', 'Time'], inplace=True)
# Remove outliers using IQR
df_clean = df.copy()
Q1 = df_clean[numeric_cols].quantile(0.25)
Q3 = df_clean[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
df_clean = df_clean[~((df_clean[numeric_cols] < (Q1 - 1.5 * IQR)) |
                      (df_clean[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# --- Model Training ---
df_model = df_clean.drop(columns=['Datetime'])
X = df_model.drop(columns=['RH'])
y = df_model['RH']
# Split & scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Train RandomForest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# --- Save & Reload Model ---
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
# Load for inference
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# List of feature names for request parsing and form generation
feature_names = X.columns.tolist()

# --- Routes ---
@app.route('/')
def home():
    html_doc = '''
    <html>
    <head><title>AirQuality RH Prediction API</title></head>
    <body>
      <h1>AirQuality Relative Humidity Prediction API</h1>
      <p>This API predicts relative humidity (RH) based on air quality features.</p>
      <ul>
        <li><a href="/predict">/predict &ndash; POST JSON payload</a></li>
        <li><a href="/some-route">/some-route &ndash; HTML form</a></li>
      </ul>
    </body>
    </html>
    '''
    return html_doc, 200, {'Content-Type': 'text/html'}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    try:
        features = [data[name] for name in feature_names]
    except KeyError as e:
        return jsonify({'error': f'Missing feature: {str(e)}'}), 400
    arr = np.array(features).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    pred = model.predict(arr_scaled)[0]
    return jsonify({'predicted_RH': float(pred)})

@app.route('/some-route', methods=['GET', 'POST'])
def some_route():
    prediction = None
    error = None
    if request.method == 'POST':
        try:
            features = [float(request.form.get(name)) for name in feature_names]
            arr = np.array(features).reshape(1, -1)
            arr_scaled = scaler.transform(arr)
            prediction = model.predict(arr_scaled)[0]
        except Exception:
            error = 'Invalid input. Please enter numeric values for all fields.'
    return render_template('form.html',
                           feature_names=feature_names,
                           prediction=prediction,
                           error=error)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

