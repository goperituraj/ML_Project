# AirQuality Relative Humidity Prediction API

This repository contains a Flask-based REST API that predicts relative humidity (RH) using air quality sensor data.

## Features
- Data preprocessing: cleaning, outlier removal, feature scaling
- Model training: Random Forest Regressor
- API endpoints for prediction
- CORS enabled for cross-origin requests

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/airquality-rh-api.git
   cd airquality-rh-api
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place the dataset**
   Download `AirQualityUCI.csv` and put it in the project root.

## Running the API

```bash
python flask_api.py
```
The service will run on `http://0.0.0.0:5000/` by default.

## API Documentation

Visit `http://localhost:5000/` in your browser to see the API documentation.

### POST /predict

- **Description**: Predict relative humidity based on input features.
- **Content-Type**: `application/json`
- **Request Body**: JSON with the following numeric fields:
  ```json
  {
    "CO(GT)": float,
    "PT08.S1(CO)": float,
    "NMHC(GT)": float,
    "C6H6(GT)": float,
    "PT08.S2(NMHC)": float,
    "NOx(GT)": float,
    "PT08.S3(NOx)": float,
    "NO2(GT)": float,
    "PT08.S4(NO2)": float,
    "PT08.S5(O3)": float,
    "T": float,
    "AH": float
  }
  ```
- **Response**: `200 OK` with JSON:
  ```json
  { "predicted_RH": float }
  ```
- **Error**: `400 Bad Request` if a required field is missing.

## Development

- Retraining the model: Update preprocessing or model parameters in `flask_api.py` and rerun.
- To expose on the internet, consider using a production server (e.g., Gunicorn) and Docker.

## License

MIT License. Feel free to open issues or pull requests.
