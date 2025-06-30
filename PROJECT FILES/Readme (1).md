
# Revolutionizing Liver Care: Predicting Liver Cirrhosis using Advanced Machine Learning Techniques

## 🧠 Project Overview

This project is developed as part of self-learning. It aims to leverage **Machine Learning** techniques to build a predictive system for the early detection of **Liver Cirrhosis** — a chronic liver disease caused by long-term damage and scarring of liver tissue.

The model is deployed using **Flask**, providing a real-time web interface where users can input clinical parameters and receive an immediate prediction on cirrhosis risk.


## 🎯 Objectives

- Predict liver cirrhosis using clinical data
- Build and evaluate multiple ML models
- Visualize data for better insight
- Deploy the best-performing model using Flask
- Create a user-friendly web interface for predictions

## 📊 Dataset

- **Source**: [UCI Machine Learning Repository – Indian Liver Patient Dataset](https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset))
- **Attributes**: Age, Gender, Bilirubin levels, Enzyme counts (ALP, SGPT, SGOT), Proteins, Albumin, A/G Ratio, Diagnosis


## 🧪 Machine Learning Algorithms Used

- Decision Tree Classifier
- Random Forest Classifier ✅ (Best Performing)
- K-Nearest Neighbors (KNN)
- XGBoost

## 🔍 Key Steps Performed

- **Data Cleaning & Preprocessing**
  - Handled missing values and encoded categorical data
  - Scaled features using `StandardScaler`
- **Exploratory Data Analysis (EDA)**
  - Used seaborn and matplotlib to analyze distributions and correlations
- **Model Training**
  - Trained using multiple classification algorithms
- **Model Evaluation**
  - Metrics used: Accuracy, Precision, Recall, F1-score
- **Model Saving**
  - Used `joblib` to export the trained model and scaler
- **Web App Deployment**
  - Built a Flask-based web UI for real-time predictions


## 🛠️ Tech Stack

- **Languages**: Python
- **Libraries**: 
  - `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `xgboost`
- **Framework**: Flask (for deployment)
- **Tools**: Jupyter Notebook, VS Code



## 📂 Project Structure
  liver_cirrhosis_project/
  │ 
  ├── training/ 
  │        └── model_training.ipynb 
  │ 
  ├── saved_models/ 
  │        ├── rf_acc_68.pkl │
  |        └── normalizer.pkl │ 
  ├── templates/ 
  |        ├── index.html │
  ├── static/ 
  |        └── style.css (optional) │ 
  ├── app.py



## 💻 How to Run This Project Locally

1. **Clone the repo**
```bash
git clone https://github.com/VamsiGunukula/liver-cirrhosis-predictor.git
cd liver-cirrhosis-predictor

2. Install required libraries

  -->pip install pandas numpy scikit-learn matplotlib seaborn joblib flask



3. Run Flask App
   --> python app.py


4. Open your browser and visit:
http://127.0.0.1:5000/


🚀 Future Enhancements

Integrate live data input from medical devices

Improve accuracy with deep learning models

Deploy on cloud (Heroku / Render / AWS)
 

🙋‍♂️ Author

  Naga Vamsi Gunukula
  CSE student at Krishna University
📧 vamsi20016@gmail.com
🔗 LinkedIn Profile:https://www.linkedin.com/in/naga-vamsi-gunukula-963253364


📌 License

This project is for educational and non-commercial use only.



🔖 Hashtags

#MachineLearning #HealthcareAI #FlaskApp #SmartInternz #PythonProject #AIinHealthcare #LiverDiseasePrediction #MLDeployment #InternshipProject


