# Pollen Predictor - AI-Powered Allergy Forecast System

**Course:** ECE 57000 - Introduction to Artificial Intelligence, Purdue University  
**Project Track:** Track 2 - Product Prototype

A full-stack application that predicts pollen severity and identifies allergen drivers using weather data and machine learning models. The system consists of a FastAPI backend with XGBoost models and a React Native mobile application.

## Table of Contents

- [Overview](#overview)
- [Code Structure](#code-structure)
- [Dependencies](#dependencies)
- [Running Instructions](#running-instructions)
- [Code Attribution](#code-attribution)
- [Data and Models](#data-and-models)

## Overview

This project implements an end-to-end pollen prediction system that:

- Predicts daily pollen severity (0-10 scale) using weather-only trained XGBoost models
- Provides 7-day pollen forecasts
- Identifies primary allergen drivers (Tree, Grass, Ragweed, Weed)
- Features a mobile app interface for user interaction
- Uses advanced biological features (VPD, Ventilation Index, Osmotic Shock Index) for improved accuracy

## Code Structure

```
ece570-project/
├── backend/                    # FastAPI backend server
│   ├── app/
│   │   ├── main.py            # FastAPI application entry point
│   │   ├── routers/           # API route handlers
│   │   │   ├── health.py      # Health check endpoint
│   │   │   └── predictions.py # Prediction endpoints
│   │   ├── schemas/           # Pydantic models for request/response validation
│   │   │   └── __init__.py    # API schemas (WeatherInput, PredictionRequest, etc.)
│   │   └── services/          # Business logic
│   │       └── prediction.py  # ML model loading and prediction service
│   ├── requirements.txt       # Python dependencies
│   └── start.sh              # Backend startup script
│
├── my-app/                    # React Native mobile application
│   ├── app/                   # Expo Router app directory
│   │   ├── (tabs)/           # Tab-based navigation
│   │   │   ├── index.tsx    # Main prediction input screen
│   │   │   ├── results.tsx  # Results display screen
│   │   │   └── explore.tsx  # Explore/help screen
│   │   └── _layout.tsx      # Root layout
│   ├── components/           # Reusable React components
│   │   ├── gradient-card.tsx
│   │   ├── haptic-tab.tsx
│   │   ├── predict-button.tsx
│   │   └── weather-input-field.tsx
│   ├── contexts/             # React context providers
│   │   └── prediction-context.tsx
│   ├── services/             # API client
│   │   └── api.ts            # Axios-based API service
│   ├── package.json          # Node.js dependencies
│   └── README.md            # Mobile app specific README
│
├── scripts/                  # Training and data processing scripts
│   ├── data_utils/           # Data preprocessing utilities
│   │   ├── analyze_weather_data.py
│   │   ├── clean_weather_data.py
│   │   └── merge_datasets.py
│   ├── training/             # Model training scripts
│   │   ├── rf.py             # Random Forest baseline
│   │   ├── rf_enhanced_features.py
│   │   ├── rf_weather_only.py
│   │   ├── xgboost_baseline.py
│   │   ├── xgboost_multitype.py
│   │   └── xgboost_weather_only.py
│   └── evaluation/           # Model evaluation scripts
│       ├── compare_all_models.py
│       ├── evaluate_weather_only.py
│       └── project_feasibility_analysis.py
│
├── models/                    # Trained ML models (joblib format)
│   ├── xgboost_total_pollen_bio_v2.joblib
│   ├── xgboost_tree_bio_v2.joblib
│   ├── xgboost_grass_bio_v2.joblib
│   ├── xgboost_ragweed_bio_v2.joblib
│   ├── xgboost_weed_bio_v2.joblib
│   └── model_features_list.joblib
│
├── data/                      # Training datasets
│   ├── allergy_pollen_data.csv
│   ├── weather_cleaned.csv
│   └── combined_allergy_weather.csv
│
└── results/                   # Model evaluation results and visualizations
    ├── model_comparison.csv
    ├── model_comparison.png
    └── weather_only_evaluation.json
```

### Key Components

**Backend (`backend/app/`):**
- `main.py`: FastAPI application setup with CORS middleware
- `routers/predictions.py`: REST API endpoints for daily/weekly predictions and allergen identification
- `services/prediction.py`: Core prediction service that loads models and engineers features
- `schemas/__init__.py`: Pydantic models for API request/response validation

**Frontend (`my-app/`):**
- React Native app using Expo Router for navigation
- Tab-based interface with prediction input, results display, and explore screens
- API integration via Axios client in `services/api.ts`

**Training Scripts (`scripts/training/`):**
- Multiple model training scripts for Random Forest and XGBoost variants
- Feature engineering with biological and meteorological features
- Model evaluation and comparison utilities

## Dependencies

### Backend Dependencies

All Python dependencies are listed in `backend/requirements.txt`:

```
# FastAPI and server
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Machine Learning
scikit-learn==1.3.2
xgboost==2.0.2
joblib==1.3.2

# Data processing
pandas==2.1.3
numpy==1.26.2

# Additional utilities
python-multipart==0.0.6
python-dateutil==2.8.2
```

**Python Version:** Python 3.12+ recommended

### Frontend Dependencies

All Node.js dependencies are listed in `my-app/package.json`. Key dependencies include:

- **React Native:** 0.81.5
- **Expo:** ~54.0.23
- **Expo Router:** ~6.0.14 (file-based routing)
- **Axios:** ^1.13.2 (HTTP client)
- **React Native Reanimated:** ~4.1.1 (animations)
- **Expo Blur:** ~15.0.7 (glassmorphism effects)
- **Expo Linear Gradient:** ~15.0.7 (gradient backgrounds)

**Node.js Version:** >= 20.15.0

### Additional Requirements

- **Expo CLI:** For running the mobile app (`npm install -g expo-cli` or use `npx expo`)
- **iOS Simulator** (macOS only) or **Android Emulator** for mobile app testing
- **Physical device** with Expo Go app for device testing

## Running Instructions

### Prerequisites

1. **Python 3.12+** installed
2. **Node.js 20.15.0+** and npm installed
3. **Trained models** in the `models/` directory (see [Data and Models](#data-and-models))
4. **Training datasets** in the `data/` directory (optional, only needed for retraining)

### Backend Setup and Execution

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the server:**
   
   Option A - Using the startup script:
   ```bash
   chmod +x start.sh
   ./start.sh
   ```
   
   Option B - Direct uvicorn command:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Verify server is running:**
   - API docs: http://localhost:8000/docs
   - Health check: http://localhost:8000/api/v1/health

The backend will automatically load models from the `models/` directory on first prediction request.

### Frontend Setup and Execution

1. **Navigate to mobile app directory:**
   ```bash
   cd my-app
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Configure API URL** (if needed):
   
   Edit `my-app/services/api.ts` and update `API_BASE_URL`:
   - iOS Simulator: `http://localhost:8000/api/v1`
   - Android Emulator: `http://10.0.2.2:8000/api/v1`
   - Physical Device: `http://YOUR_COMPUTER_IP:8000/api/v1`

4. **Start Expo development server:**
   ```bash
   npx expo start
   ```

5. **Run on your platform:**
   - Press `i` for iOS simulator
   - Press `a` for Android emulator
   - Scan QR code with Expo Go app on physical device

### Training Models (Optional)

To retrain models from scratch:

1. **Ensure training data is available:**
   ```bash
   # Data should be in data/ directory
   ls data/combined_allergy_weather.csv
   ```

2. **Run training script:**
   ```bash
   cd scripts/training
   python xgboost_weather_only.py
   ```

3. **Models will be saved to `models/` directory**

### Testing the System

1. **Start backend** (see Backend Setup above)
2. **Start frontend** (see Frontend Setup above)
3. **Use the mobile app:**
   - Navigate to "Predict" tab
   - Enter weather data (temperature, precipitation, wind speed)
   - Tap "Get Prediction"
   - View results on "Results" tab

4. **Test API directly:**
   - Visit http://localhost:8000/docs
   - Use the interactive API documentation to test endpoints

## Code Attribution

### Code Written by Me

All code in this repository was written by me for this project, with the following exceptions:

**Backend:**
- `backend/app/main.py`: FastAPI application setup - **Written by me**
- `backend/app/routers/predictions.py`: API endpoint handlers - **Written by me**
- `backend/app/services/prediction.py`: ML model service and feature engineering - **Written by me**
- `backend/app/schemas/__init__.py`: Pydantic schemas - **Written by me**
- `backend/app/routers/health.py`: Health check endpoint - **Written by me**

**Frontend:**
- `my-app/app/`: All React Native screens and layouts - **Written by me**
- `my-app/components/`: All custom React components - **Written by me**
- `my-app/services/api.ts`: API client implementation - **Written by me**
- `my-app/contexts/prediction-context.tsx`: State management - **Written by me**

**Training Scripts:**
- `scripts/training/*.py`: All model training scripts - **Written by me**
- `scripts/data_utils/*.py`: Data preprocessing utilities - **Written by me**
- `scripts/evaluation/*.py`: Model evaluation scripts - **Written by me**

### Code Adapted from External Sources

**Framework and Library Usage:**
- **FastAPI**: Used framework as-is from https://fastapi.tiangolo.com/ - no code copied, only framework usage
- **Expo Router**: Used framework as-is from https://docs.expo.dev/router/introduction/ - no code copied, only framework usage
- **XGBoost**: Used library as-is from https://xgboost.readthedocs.io/ - no code copied, only library usage
- **React Native**: Used framework as-is - no code copied, only framework usage

**Third-Party Libraries:**
All dependencies listed in `requirements.txt` and `package.json` are used as-is without modification. No code was copied from these libraries.

### Code Copied (with Attribution)

**None.** All code in this repository was either written by me or uses standard frameworks/libraries as intended without copying source code.

### Template/Starter Code

**Mobile App Base:**
- The initial Expo project structure was created using `npx create-expo-app`, but all application-specific code (screens, components, API integration) was written by me.

**No other starter code or templates were used.**

## Data and Models

### Datasets

The project uses the following datasets (located in `data/` directory):

- `allergy_pollen_data.csv`: Historical pollen count data
- `weather_cleaned.csv`: Historical weather data (temperature, precipitation, wind)
- `combined_allergy_weather.csv`: Merged dataset for training

**Note:** These datasets are not automatically downloaded due to their size and potential licensing restrictions. They should be placed in the `data/` directory before running training scripts.

### Trained Models

The trained models are stored in the `models/` directory:

- `xgboost_total_pollen_bio_v2.joblib`: Main pollen severity prediction model
- `xgboost_tree_bio_v2.joblib`: Tree allergen model
- `xgboost_grass_bio_v2.joblib`: Grass allergen model
- `xgboost_ragweed_bio_v2.joblib`: Ragweed allergen model
- `xgboost_weed_bio_v2.joblib`: Weed allergen model
- `model_features_list.joblib`: Feature list for model alignment

**Model Loading:**
Models are loaded automatically by the `PredictionService` class in `backend/app/services/prediction.py` when the first prediction request is made. The service checks for model files in the `models/` directory relative to the project root.

**Note:** Models are not automatically downloaded due to their size (~50-100MB total). They should be included in the submission zip file or made available in the `models/` directory.

### Automatic Download (Not Implemented)

Automatic download of datasets and models is **not implemented** for the following reasons:

1. **Dataset Size:** Training datasets are large (100+ MB) and may have licensing restrictions
2. **Model Size:** Trained models are large (50-100 MB total) and should be included in submission
3. **Data Source:** Datasets may require authentication or have usage restrictions
4. **Submission Requirement:** The project submission should be self-contained with all necessary files

If automatic download is desired, it could be implemented using:
- `requests` library for HTTP downloads
- `gdown` for Google Drive files
- `wget` or `curl` for direct file downloads
- Checksums for verification

However, for this submission, all required files (models and data) should be included in the zip file.

## Additional Notes

### Project Track

This project follows **Track 2: Product Prototype** from the ECE 57000 course requirements:
- **Problem:** Predicting pollen severity for allergy sufferers
- **Solution:** ML-powered prediction system with mobile interface
- **User:** Allergy sufferers who need daily pollen forecasts
- **Evaluation:** Model performance metrics and user-facing prototype

### Model Performance

The weather-only trained XGBoost models achieve:
- **R² Score:** 0.87 (main model)
- **Mean Absolute Error:** Significantly improved over baseline
- **47.9% better accuracy** compared to multitype models for weather-only predictions

### API Endpoints

- `POST /api/v1/predict/daily`: Daily pollen severity prediction
- `POST /api/v1/predict/weekly`: 7-day pollen forecast
- `POST /api/v1/allergen/identify`: Allergen breakdown and identification
- `GET /api/v1/health`: Health check

Full API documentation available at `/docs` when server is running.

---

**For questions or issues, refer to the individual README files in `backend/` and `my-app/` directories for component-specific details.**
