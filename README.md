# **Stock Price Prediction Using LSTM: General Electric (GE) Forecasting**

## **📌 Overview**
This project utilizes **Long Short-Term Memory (LSTM) Neural Networks** for **time-series forecasting** to predict **General Electric (GE) stock prices**. By leveraging deep learning methodologies, we trained an LSTM model using **historical stock price data** to evaluate its effectiveness in forecasting price movements. The model was developed using **Python, Keras, TensorFlow, and Pandas**, ensuring robust analysis and prediction accuracy.

📄 **Detailed Report:** [GE Price Prediction - Corporate Finance](https://github.com/Swasti28/Stock-Price-Prediction-Using-LSTM-General-Electric-GE-Forecasting/blob/main/GE%20Price%20Prediction%20-%20Corporate%20Finance.pdf)
📄 **Project Presentation:** [General Electric Stock Prediction Presentation](https://github.com/Swasti28/Stock-Price-Prediction-Using-LSTM-General-Electric-GE-Forecasting/blob/main/General%20Electric%20-FIN600001.pptx)

---

## **📍 Data Collection & Preprocessing**

### **📝 Data Source**
- **Extracted historical stock price data** from **Yahoo Finance API** covering **January 1, 2015, to November 17, 2023**.
- Selected the **Closing Price** as the primary feature for prediction.

### **📊 Data Preprocessing Steps:**
✔ **MinMaxScaler Normalization** – Scaled stock prices to a range between 0 and 1 for better neural network performance.  
✔ **Data Splitting** – 70% of data used for training, 30% for testing.  
✔ **Reshaped Data** – Transformed data into **sequential format** suitable for LSTM.

---

## **📍 Model Development & Training**

### **📝 Model Architecture**
- **Sequential LSTM Model** built using **Keras and TensorFlow**.
- Model structure:
  ```python
  model = Sequential()
  model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
  model.add(LSTM(50, return_sequences=False))
  model.add(Dense(25))
  model.add(Dense(1))
  ```

### **📊 Training Parameters:**
✔ **50 Neurons in LSTM Layers** – Used for pattern recognition in stock price trends.  
✔ **Optimizer: Adam** – Used for gradient-based optimization.  
✔ **Loss Function: Mean Squared Error (MSE)** – Measures model prediction accuracy.  
✔ **Epochs: 50** – Number of training iterations.

---

## **📍 Model Evaluation & Results**

### **✅ Key Findings:**
- **GE Closing Stock Price (11/20/2023):** $120.07  
- **Predicted Closing Price (12/12/2023):** $117.06 (-2.5% decrease)  
- **Model Evaluation:**
  - **Root Mean Squared Error (RMSE):** 0.341 (~100% accuracy)  
  - **Loss Curve Analysis:** Model effectively minimized error over epochs.

📊 **Prediction Plot:**
![LSTM Stock Prediction Graph](https://github.com/Swasti28/Stock-Price-Prediction-Using-LSTM-General-Electric-GE-Forecasting/blob/main/Picture1.png)

---

## **📍 Technologies & Tools Used**
- **📈 Data Analytics:** Pandas, NumPy, Scikit-Learn
- **🧠 Deep Learning:** Keras, TensorFlow (LSTM Neural Networks)
- **📊 Visualization:** Matplotlib, Seaborn
- **💰 Financial Data Extraction:** Yahoo Finance API
- **📌 Model Evaluation:** RMSE, Loss Function Analysis

---

## **📢 Key Takeaways & Recommendations**
✅ **LSTM is highly effective in forecasting stock prices** with minimal error.  
✅ **Stock price trends indicate a potential short-term decline in GE’s valuation.**  
✅ **Further enhancements:** Incorporate **technical indicators (RSI, MACD)** and **other stock market factors** to improve predictive power.  

📄 **Complete Report:** [GE Price Prediction - Corporate Finance](https://github.com/Swasti28/Stock-Price-Prediction-Using-LSTM-General-Electric-GE-Forecasting/blob/main/GE%20Price%20Prediction%20-%20Corporate%20Finance.pdf)  
📄 **Presentation:** [General Electric Stock Prediction](https://github.com/Swasti28/Stock-Price-Prediction-Using-LSTM-General-Electric-GE-Forecasting/blob/main/General%20Electric%20-FIN600001.pptx)

---

## **📂 Project Files & Resources**
📌 **Stock Price Data Analysis Report** - [Download](https://github.com/Swasti28/Stock-Price-Prediction-Using-LSTM-General-Electric-GE-Forecasting/blob/main/GE%20Price%20Prediction%20-%20Corporate%20Finance.pdf)  
📌 **Project Presentation** - [Download](https://github.com/Swasti28/Stock-Price-Prediction-Using-LSTM-General-Electric-GE-Forecasting/blob/main/General%20Electric%20-FIN600001.pptx)  

---

## **📢 Contributions & Future Work**
This project provides **data-driven insights** for financial analysts and stock market investors. Future enhancements:
🔹 **Hybrid LSTM + GRU Model for Improved Predictions** 🤖  
🔹 **Sentiment Analysis on Stock Market News for Better Forecasting** 💬  
🔹 **Feature Engineering with Technical Indicators** 📈  
