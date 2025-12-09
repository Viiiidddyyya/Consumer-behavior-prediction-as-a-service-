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
<img width="571" height="914" alt="image" src="https://github.com/user-attachments/assets/0eca1818-26bb-4961-bd3c-4a4a3c37ddd0" />

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
## Results and Visulizations
<img width="744" height="430" alt="image" src="https://github.com/user-attachments/assets/36c4ef2e-d9ab-4358-8b6e-7eefa77e69c0" />
<img width="925" height="430" alt="image" src="https://github.com/user-attachments/assets/68a5d884-b0ee-4d83-905e-a2ed8554adb2" />
<img width="911" height="412" alt="image" src="https://github.com/user-attachments/assets/01158c25-64a8-4c0d-82c8-452747343233" />
<img width="911" height="413" alt="image" src="https://github.com/user-attachments/assets/f33a1afe-6c14-4329-907f-c9c4d50a5cda" />

## Deployment
<img width="523" height="248" alt="image" src="https://github.com/user-attachments/assets/8dd02a32-4675-4086-a688-63b257da7426" />
<img width="484" height="493" alt="image" src="https://github.com/user-attachments/assets/0ddf322e-4e08-4799-a32f-8aad3a541ba0" />






