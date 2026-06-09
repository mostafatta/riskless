PROJECT: TADAWUL PORTFOLIO RISK ANALYZER
========================================
A Python-based engine to classify Saudi Stock Market portfolios into risk categories (Low, Medium, High) using mathematical formulas and Machine Learning.

[1] QUICK START (WINDOWS)
-------------------------
We have provided "One-Click" scripts for the most common tasks:

1. To Analyze a Portfolio:
   Double-click "run_analysis.bat".
   - This downloads the latest data from Yahoo Finance.
   - It calculates Volatility and Beta for the stocks in 'main.py'.
   - It prints the Risk Score and Category.

2. To Use the AI Predictor:
   Double-click "run_predictor.bat".
   - This opens an interactive tool.
   - You can type in any Volatility and Beta numbers.
   - The AI will instantly tell you the Risk Category.

----------------------------------------

[2] MANUAL INSTALLATION (Mac/Linux or Advanced Users)
-----------------------------------------------------
If you are running this from a terminal, first install the required libraries:

    pip install -r requirements.txt

Then you can run the scripts using Python commands:
- Analysis:   python main.py
- Predictor:  python predict_risk.py

----------------------------------------

[3] HOW TO CUSTOMIZE THE PORTFOLIO
----------------------------------
To test your own specific stocks:

1. Open the file "main.py" in any text editor (Notepad, VS Code).
2. Scroll to the bottom section (under if __name__ == "__main__":).
3. Edit the 'my_stocks' list. Use ticker symbols ending in .SR (e.g., '1120.SR').
4. Edit the 'my_weights' list. Ensure they add up to 1.0 (e.g., 0.50, 0.30, 0.20).
5. Save the file and run "run_analysis.bat" again.

----------------------------------------

[4] MACHINE LEARNING WORKFLOW
-----------------------------
If you want to retrain the AI model or see the accuracy report:

1. Generate Training Data:
   Run: python src/data_generator.py
   (This creates 5,000 random portfolios in 'data/processed/portfolio_dataset.csv')

2. Train the Model:
   Run: python src/ml_model.py
   (This trains the Random Forest classifier)

3. View Results:
   After training, open the file "model_results.txt".
   It contains the Accuracy Score and Classification Report proving the model's performance.

----------------------------------------

[5] FILE STRUCTURE
------------------
.
├── main.py                 # The script to analyze a specific portfolio
├── predict_risk.py         # The interactive AI tool
├── requirements.txt        # List of required libraries
├── run_analysis.bat        # Shortcut to run main.py
├── run_predictor.bat       # Shortcut to run predict_risk.py
├── model_results.txt       # The accuracy report of the AI
│
├── models/                 # Contains the saved AI brain (.pkl file)
├── data/                   # Contains raw prices and generated datasets
│
└── src/                    # Source Code (Do not edit unless necessary)
    ├── data_loader.py      # Fetches Yahoo Finance data
    ├── calculations.py     # Matrix math for Volatility & Beta
    ├── risk_labeler.py     # Logic to assign Low/Med/High labels
    ├── data_generator.py   # Creates synthetic data for training
    └── ml_model.py         # Trains the Machine Learning model