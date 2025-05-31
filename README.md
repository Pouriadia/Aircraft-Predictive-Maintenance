
# Aircraft Predictive Maintenance

This project uses Recurrent Neural Networks (RNNs) to predict the **Remaining Useful Life (RUL)** of aircraft engines based on historical sensor readings. The aim is to anticipate potential failures and schedule maintenance efficiently, enhancing aircraft safety and reducing downtime.

---

## 📁 Project Structure

```
Aircraft-Predictive-Maintenance-main/
│
├── Aircraft_Data.ipynb      # Main Jupyter notebook with data processing, modeling, and evaluation
├── PM_train.txt             # Training dataset (sensor readings until failure)
├── PM_test.txt              # Test dataset (sensor readings, truncated before failure)
├── PM_truth.txt             # Ground truth for RUL of test dataset
└── README.md                # Project documentation (you're reading it!)
```

---

## 🧠 Problem Statement

Aircraft engines are equipped with sensors that generate time-series data over operational cycles. Each engine eventually fails, and we aim to **predict the number of cycles remaining until failure (RUL)** using this sensor data.

---

## 📊 Dataset Description

The dataset is sourced from NASA's C-MAPSS dataset. It consists of three parts:

- `PM_train.txt`: Sensor data for multiple engines up to the point of failure.
- `PM_test.txt`: Sensor data ending before failure.
- `PM_truth.txt`: Actual RUL values for each engine in the test set.

Each row in the train/test files includes:

- Engine ID
- Operational cycle
- 3 operational settings
- 21 sensor measurements (s1 to s21)

---

## 🔍 Workflow Overview

### 1. **Data Preprocessing**
- Load datasets and clean missing values.
- Rename columns for clarity.
- Sort records by engine ID and cycle.
- Calculate RUL for training data: `RUL = max_cycle - current_cycle`.
- Normalize both train and test datasets using `MinMaxScaler`.

### 2. **Exploratory Data Analysis (EDA)**
- Visualize trends in sensor values over time.
- Example: Sensor 1 tends to **increase**, while Sensor 6 tends to **decrease** as engines approach failure.
- Observation: Certain sensor patterns correlate with approaching failure.

### 3. **Label Generation**
- Define binary classification target based on a threshold RUL (e.g., 30 cycles).
- Useful for modeling binary classification and regression tasks.

### 4. **Sequence Generation**
- Convert sensor time-series into input sequences for RNN models.
- Sliding windows are used to create overlapping sequences of cycles.

---

## 🔁 Model Architecture

Several RNN-based deep learning models are trained:

### 🔹 Simple RNN (1 Feature)
- Uses a single sensor (e.g., `s1`) as input.
- Basic sequential model with `SimpleRNN`, `Dense`, and `Dropout` layers.

### 🔹 Simple RNN (25 Features)
- Uses all 25 features (3 settings + 21 sensors).
- Deeper model with increased input dimensionality.

### 🔹 Bidirectional RNN
- Uses bidirectional RNN layers for improved context understanding.
- Helps capture both forward and backward dependencies in the sequence.

---

## 📈 Evaluation Metrics

Model performance is evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **Confusion Matrix**

Training and validation losses/accuracies are plotted to monitor learning progress.

---

## 🔬 Results & Observations

- Sensor patterns like rising `s1` and falling `s6` are strong indicators of imminent failure.
- Models using all 25 features outperform those with a single feature.
- RNNs effectively capture temporal dependencies in sensor data.
- Proper normalization and sequence generation are key to model performance.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Pouriadia/Aircraft-Predictive-Maintenance.git
   cd Aircraft-Predictive-Maintenance
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the Jupyter notebook:
   ```bash
   jupyter notebook Aircraft_Data.ipynb
   ```

---

## 📌 Future Work

- Integrate more complex architectures like LSTM, GRU, or Transformer-based models.
- Add model explainability (e.g., SHAP or LIME).
- Explore ensemble modeling for improved RUL prediction.

---

## 📚 References

- NASA C-MAPSS Dataset: https://www.nasa.gov/cmapps
- TensorFlow Documentation: https://www.tensorflow.org/
- Original paper: Saxena, A., & Goebel, K. (2008). "Turbofan Engine Degradation Simulation Data Set".
