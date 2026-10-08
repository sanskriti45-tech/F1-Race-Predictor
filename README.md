````markdown
# 🏎️ F1 Race Predictor

> A machine-learning powered Formula 1 race prediction platform that uses historical race data, qualifying performance, driver and team form, circuit statistics, and season performance to predict upcoming race outcomes.

---

## 📌 Overview

**F1 Race Predictor** is a full-stack Formula 1 analytics and prediction platform built around real Formula 1 data.

The system combines **FastF1**, historical race results, qualifying performance, driver statistics, team performance, circuit history, and machine learning to generate predictions for upcoming races.

The platform is designed to answer questions such as:

- 🏆 Who is most likely to win the next race?
- 🏁 What finishing order does the model predict?
- 📊 What is each driver's estimated win probability?
- 👤 How has a driver performed recently?
- 🏎️ How is a constructor performing?
- 📍 How has a driver or team historically performed at a circuit?
- 📈 How well does the prediction model perform on historical races?

The frontend presents these predictions through an interactive F1-themed dashboard.

---

## ✨ Key Features

### 🏁 Race Prediction

Predicts the expected finishing order for an upcoming Formula 1 race using the trained machine-learning model.

Each prediction includes:

- Predicted finishing position
- Predicted rank
- Estimated win probability
- Driver
- Team
- Circuit
- Qualifying/grid information

---

### 📊 Driver Performance

The system tracks driver performance using historical and recent-race statistics, including:

- Recent finishing-form
- Recent qualifying-form
- Recent points
- Finish rate
- Season average finish
- Season average qualifying position
- Season points
- Number of races completed in the season

---

### 🏎️ Team / Constructor Analysis

Constructor performance is incorporated into the prediction features through:

- Recent team finishing form
- Recent qualifying form
- Recent team points
- Historical circuit performance

The frontend dynamically represents different constructors rather than being limited to a single team.

---

### 📍 Circuit Analysis

Circuit-specific historical performance is included in the model.

Features include:

- Driver average finish at the circuit
- Team average finish at the circuit
- Driver's previous starts at the circuit
- Team's previous starts at the circuit

This allows the model to consider how a driver or constructor has historically performed at a particular track.

---

### 🧠 Machine Learning

The current baseline prediction model uses:

**HistGradientBoostingRegressor**

The model predicts expected finishing position and converts those predictions into relative win probabilities.

The prediction pipeline is designed to avoid using information from the future race result when constructing historical training features.

---

### 📈 Model Evaluation

The project includes historical backtesting and evaluation metrics such as:

- Mean Absolute Error (MAE)
- Rank Correlation
- Top-1 Agreement
- Top-3 Agreement
- Brier Score
- Log Loss
- Number of evaluated races

These metrics allow the model to be evaluated against historical race outcomes.

> **Important:** Win probabilities are currently generated through a heuristic probability layer and should not be interpreted as calibrated real-world probabilities until they have been validated on sufficient real historical data.

---

### ⚡ FastF1 Integration

The project uses **FastF1** to access Formula 1 session and race information.

The backend can retrieve:

- Race schedules
- Race results
- Qualifying results
- Driver information
- Constructor/team information
- Grid positions
- Finishing positions
- Points
- Circuit information

The project includes a live-data verification script to confirm that FastF1 can retrieve real qualifying and race data.

---

### 🔌 Backend API

The backend exposes prediction information through a lightweight API.

Available endpoints include:

```text
GET /api/health
GET /api/prediction/next-race
````

The API provides information about:

* FastF1 availability
* Historical data availability
* Model availability
* Prediction generation status
* Upcoming race
* Predictions
* Feature values
* Backtesting results
* Prediction history
* Errors/status information

---

### 🎨 Interactive Frontend

The frontend provides an F1-inspired dashboard containing:

* Race prediction interface
* Driver selection
* Driver performance dashboard
* Constructor/team views
* Prediction table
* Circuit information
* Model information
* Interactive F1-style car visualization

The frontend dynamically consumes backend data rather than implementing prediction logic itself.

---

## 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │      FastF1 API      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Historical Data    │
                    │  Race + Qualifying   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Engineering  │
                    │ Driver / Team / Track│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Machine Learning     │
                    │ HistGradientBoosting │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Race Predictions     │
                    │ Rank / Finish / Win %│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       REST API       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Interactive Web UI  │
                    └──────────────────────┘
```

---

## 🧮 Feature Engineering

The model uses historical features generated before the target race.

### Driver Features

```text
driver_form_3
driver_form_5
driver_quali_form_3
driver_quali_form_5
driver_points_3
driver_points_5
driver_finish_rate_3
driver_finish_rate_5
```

### Team Features

```text
team_form_3
team_form_5
team_quali_form_3
team_quali_form_5
team_points_3
team_points_5
```

### Circuit Features

```text
circuit_driver_avg_finish
circuit_team_avg_finish
circuit_driver_prior_starts
circuit_team_prior_starts
```

### Season Features

```text
season_avg_finish
season_avg_quali
season_points
season_races_so_far
```

### Race / Qualifying Features

```text
qualifying_position
grid_position
```

The model also tracks:

```text
driver
team
circuit
```

---

## 📂 Project Structure

```text
F1-Race-Predictor/
│
├── frontend/
│   ├── index.html
│   └── assets/
│       └── f1_clean.mp4
│
├── backend/
│   │
│   ├── src/
│   │   ├── app/
│   │   │   └── api.py
│   │   │
│   │   ├── data/
│   │   │   ├── fastf1_client.py
│   │   │   └── loaders.py
│   │   │
│   │   ├── features/
│   │   │   └── dataset.py
│   │   │
│   │   └── models/
│   │       ├── baseline.py
│   │       └── predict.py
│   │
│   ├── scripts/
│   │   ├── build_dataset_and_train.py
│   │   ├── run_backtest.py
│   │   ├── serve_api.py
│   │   └── verify_live_fetch.py
│   │
│   ├── tests/
│   │
│   ├── data/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── ...
│   │
│   ├── models/
│   │
│   ├── pyproject.toml
│   └── README.md
│
└── README.md
```

---

## 🛠️ Technology Stack

### Backend

* Python
* FastF1
* Pandas
* NumPy
* Scikit-learn
* Joblib

### Machine Learning

* HistGradientBoostingRegressor
* Time-aware feature generation
* Historical backtesting
* Ranking evaluation
* Probability evaluation

### Frontend

* HTML5
* CSS3
* JavaScript
* SVG
* REST API integration

### Data

* Formula 1 race data
* Formula 1 qualifying data
* Driver statistics
* Constructor/team statistics
* Circuit statistics

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/F1-Race-Predictor.git
cd F1-Race-Predictor
```

---

## 2. Create a Python Virtual Environment

Navigate to the backend:

```bash
cd backend
```

Create the virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

---

# 🔍 Verify FastF1 Live Data

Before building the dataset, verify that FastF1 can retrieve real Formula 1 data:

```bash
python scripts/verify_live_fetch.py
```

The verification checks that real:

* Race data
* Qualifying data
* Drivers
* Teams
* Positions

can be retrieved successfully.

A successful verification should return non-empty session data.

---

# 🧠 Build the Training Dataset

After verifying FastF1, build the historical dataset and train the baseline model.

For example:

```bash
python scripts/build_dataset_and_train.py --seasons 2022 2023
```

You can provide additional seasons as required.

The pipeline:

```text
FastF1
   ↓
Race + Qualifying Data
   ↓
Data Normalization
   ↓
Feature Engineering
   ↓
Training Dataset
   ↓
Model Training
   ↓
Saved Model
```

---

# 📊 Run Backtesting

Evaluate the trained model using historical races:

```bash
python scripts/run_backtest.py
```

The backtest reports metrics including:

```text
MAE
Rank Correlation
Top-1 Agreement
Top-3 Agreement
Brier Score
Log Loss
```

---

# 🔌 Start the Backend API

Start the prediction API:

```bash
python scripts/serve_api.py --port 8787
```

The API will be available at:

```text
http://localhost:8787
```

Health check:

```text
http://localhost:8787/api/health
```

Upcoming-race prediction:

```text
http://localhost:8787/api/prediction/next-race
```

---

# 🌐 Start the Frontend

Open another terminal.

Navigate to the frontend:

```bash
cd frontend
```

Start a local web server:

```bash
python -m http.server 5500
```

Then open:

```text
http://localhost:5500
```

The frontend communicates with the backend API running on port `8787`.

---

# 🔄 Prediction Workflow

When the application is used for an upcoming race:

```text
Upcoming Race
      ↓
FastF1 Schedule
      ↓
Qualifying Completed?
      ↓
Historical Driver/Team/Circuit Features
      ↓
ML Model
      ↓
Predicted Finishing Positions
      ↓
Win Probability Layer
      ↓
REST API
      ↓
Interactive Dashboard
```

The backend remains the source of truth for prediction values.

The frontend does **not** independently calculate race predictions.

---

# 🛡️ Data Integrity

The feature-generation pipeline is designed around a prediction cutoff.

Historical features are generated from information available before the race being predicted.

This is important because using the target race's final result to generate its own prediction would introduce **data leakage**.

The project therefore separates:

```text
Historical information
        ↓
Features
        ↓
Prediction
        ↓
Actual race result
        ↓
Evaluation
```

rather than allowing future race results to influence the prediction.

---

# ⚠️ Important Notes

### Real Data

The repository does not rely on fabricated race results.

The project is designed to retrieve Formula 1 data through FastF1 when the user runs the data pipeline.

The repository may therefore contain empty data/model directories in a fresh clone until the training pipeline is executed.

---

### Win Probability

The current win probability calculation is a heuristic transformation of predicted finishing positions.

It should therefore be treated as an **estimated relative probability**, not as a statistically calibrated probability, until it has been properly calibrated and validated on real historical data.

---

### FastF1 Connectivity

FastF1 requires network access to retrieve session data.

If FastF1 cannot access the required Formula 1 data source in a particular environment, the live-data verification step may fail even though the project code itself is correctly configured.

---

# 🧪 Testing

The project includes automated tests covering core data, feature, model, API, and frontend behavior.

Run the backend test suite with:

```bash
pytest -v
```

For live-data verification:

```bash
python scripts/verify_live_fetch.py
```

---

# 📈 Future Improvements

Potential future improvements include:

* Calibrated win probabilities
* More advanced ranking models
* Additional machine-learning algorithms
* Weather features
* Tyre strategy information
* Pit-stop performance
* Safety-car probability
* Track-specific characteristics
* Driver/team upgrades
* Sprint-weekend handling improvements
* Automated model retraining
* Live race updates
* Historical prediction-vs-actual visualization
* More detailed model explainability
* Cloud deployment
* Automated data refresh

---

# 🏁 Project Goals

The long-term goal of **F1 Race Predictor** is to create a reliable, data-driven Formula 1 prediction platform that combines:

> **Real F1 Data + Feature Engineering + Machine Learning + Interactive Visualization**

to provide an accessible way to explore and understand Formula 1 race predictions.

---

## 👩‍💻 Author

**Sanskriti Maheshwari**

Built as a machine-learning and data-driven Formula 1 analytics project.

---

## ⭐ If You Like This Project

If you find the project interesting, consider giving the repository a ⭐ on GitHub.

Contributions, ideas, and improvements are welcome.

---

## 📄 License

Add your preferred open-source license here, such as MIT, before publishing the repository.

```

### One important recommendation

For your GitHub repository, I would **not** put phrases like “98% accurate,” “predicts the winner accurately,” or “real-time accurate predictions” in the README unless you have actually run the model on real historical data and have the corresponding evaluation results.

Your project is much more professional if the README clearly separates **what the system does** from **what has actually been validated**.
```
real
  historical seasons as unverified.
