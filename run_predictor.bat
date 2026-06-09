@echo off
color 0A
title Tadawul Risk Analyzer - Full Setup & Run

echo ========================================================
echo       SAUDI MARKET RISK ANALYZER - FULL PIPELINE
echo ========================================================
echo.
echo [STEP 1] Checking and Installing Libraries...
pip install -r requirements.txt
echo.
echo --------------------------------------------------------
echo.
echo [STEP 2] Downloading Market Data ^& Generating 500 Portfolios...
echo (This might take a minute to fetch stock data from Yahoo Finance)
python src/data_generator.py
echo.
echo --------------------------------------------------------
echo.
echo [STEP 3] Training the Artificial Intelligence Models...
echo.
echo --- Training 1/3: Random Forest (Rolling Window) ---
python src/ml_model_rf.py
echo.
echo --- Training 2/3: LSTM Deep Learning (Rolling Window) ---
python src/ml_model_lstm.py
echo.
echo --- Training 3/3: SVM (Rolling Window) ---
python src/ml_model_svm.py
echo.
echo --------------------------------------------------------
echo.
echo [STEP 4] Launching the Interactive Predictor...
echo.
python predict_risk.py

echo.
echo ========================================================
echo All tasks completed. You can close this window.
pause