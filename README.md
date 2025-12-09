# Consumer-behavior-prediction-as-a-service-

An end-to-end machine learning system designed to forecast electronics consumer demand up to **3 months in advance**.  
The project combines time-series forecasting, machine learning, and cloud deployment with an interactive desktop GUI and scalable APIs.

---

## 🚀 Project Overview
Accurate demand forecasting in electronics retail is challenging due to seasonality, rapid product cycles, promotions, and volatile consumer behavior.  
This project addresses these challenges by integrating statistical models and machine learning to deliver reliable, production-ready forecasts.

---

## 🎯 Objectives
- Predict electronics demand 1–3 months ahead  
- Integrate real and synthetic retail datasets  
- Engineer temporal, seasonal, and external-signal features  
- Evaluate models using sMAPE, RMSE, MAE, and Direction-of-Change accuracy  
- Provide a desktop GUI and cloud-deployable API service  

---

## 🧠 Models Used
- Naïve Seasonal (baseline)
- Prophet
- SARIMAX
- LightGBM
- Hybrid (Prophet + LightGBM)

---

## 📊 Results
- Achieved **< 6% sMAPE** (target ≤ 12%)
- Direction-of-Change accuracy **> 85%**
- Consistent performance across brands and months

---

## 🧱 System Architecture
- Data Layer: Real + synthetic retail data  
- Processing Layer: Feature engineering & transformations  
- Modeling Layer: Forecasting & ensemble models  
- UI Layer: Desktop GUI with charts & exports  
- Cloud Layer: Dockerized deployment with REST APIs  

---

## 🛠️ Technology Stack
- **Programming:** Python  
- **Libraries:** pandas, scikit-learn, Prophet, SARIMAX, LightGBM  
- **Visualization:** Matplotlib  
- **GUI:** Tkinter  
- **Cloud:** Docker, AWS EC2  
- **CI/CD:** GitHub Actions  
- **Monitoring:** AWS CloudWatch  

---

## 📌 Future Enhancements
- Real-time data ingestion  
- Economic and pricing indicators  
- Web-based dashboard  
- Automated retraining pipelines  

---



