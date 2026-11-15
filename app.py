
import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_files():
    model = joblib.load("heart_disease_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("feature_names.pkl")
    return model, scaler, features

model, scaler, features = load_files()

# ---------------- MODEL RESULTS ----------------
results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "KNN"
    ],
    "Accuracy": [0.7728, 0.6544, 0.7739, 0.7222],
    "Precision": [0.6858, 0.4335, 0.6927, 0.5613],
    "Recall": [0.4606, 0.4606, 0.4550, 0.3780],
    "F1 Score": [0.5510, 0.4466, 0.5493, 0.4518]
})

# ---------------- HEADER ----------------
st.title("❤️ Heart Disease Risk Prediction")
st.write("Machine Learning Based Heart Disease Prediction System")

st.warning(
    "Educational project only. This application is not a medical "
    "diagnosis tool and must not be used for medical decisions."
)

# ---------------- NAVIGATION ----------------
overview_tab, prediction_tab = st.tabs(
    ["📊 Project Dashboard", "🩺 Patient Prediction"]
)

# =================================================
# DASHBOARD
# =================================================
with overview_tab:

    st.header("Project Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", "9,000")
    c2.metric("Input Features", "10")
    c3.metric("Training Records", "7,200")
    c4.metric("Testing Records", "1,800")

    st.divider()

    st.subheader("Model Performance")

    a, b, c, d = st.columns(4)

    a.metric("Best Accuracy", "77.39%", "Random Forest")
    b.metric("Best Precision", "69.27%", "Random Forest")
    c.metric("Best Recall", "46.06%", "Logistic Regression / Decision Tree")
    d.metric("Best F1 Score", "55.10%", "Logistic Regression")

    st.divider()

    st.subheader("Model Comparison")

    st.write("Compare all four performance metrics")

    chart_data = results.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
    ]

    st.bar_chart(
    chart_data,
    height=450
     )

    
    import matplotlib.pyplot as plt
    
    # Get feature names and Logistic Regression coefficients
    coefficients = model.coef_[0]
    
    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": coefficients
    })
    
    # Sort features by coefficient
    coef_df = coef_df.sort_values(
        by="Coefficient",
        ascending=True
    )
    
    st.divider()
    st.subheader("Logistic Regression - Feature Coefficients")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.barh(
        coef_df["Feature"],
        coef_df["Coefficient"]
    )
    
    ax.axvline(0, linestyle="--", linewidth=1)
    
    ax.set_xlabel("Coefficient Value")
    ax.set_ylabel("Patient Features")
    ax.set_title("Feature Coefficient Analysis")
    
    plt.tight_layout()
    
    st.pyplot(fig)
    plt.close(fig)
    
    st.caption(
        "Positive coefficients increase the model's predicted log-odds "
        "for class 1; negative coefficients decrease them. "
        "These values do not show causation."
    )
    

    st.divider()

    st.subheader("PCA Visualization Information")

    pca_data = pd.DataFrame({
        "Component": ["PCA 1", "PCA 2"],
        "Explained Variance (%)": [23.32, 17.52]
    })

    st.bar_chart(
        pca_data.set_index("Component"),
        height=250
    )

    st.metric("Total Variance Explained", "40.85%")

    st.caption(
        "PCA reduces the original 10 features into two components "
        "for visualization."
    )

# =================================================
# PATIENT PREDICTION
# =================================================
with prediction_tab:

    st.header("Enter Patient Details")

    with st.form("patient_form"):

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age", min_value=1, max_value=120, value=45
            )

            sex = st.selectbox("Sex", ["Male", "Female"])

            systolic = st.number_input(
                "Resting Systolic BP", min_value=0.0, value=120.0
            )

            diastolic = st.number_input(
                "Resting Diastolic BP", min_value=0.0, value=80.0
            )

            cholesterol = st.number_input(
                "Total Cholesterol", min_value=0.0, value=190.0
            )

        with col2:
            hdl = st.number_input("HDL", min_value=0.0, value=50.0)

            ldl = st.number_input("LDL", min_value=0.0, value=110.0)

            triglycerides = st.number_input(
                "Triglycerides", min_value=0.0, value=130.0
            )

            fasting_sugar = st.number_input(
                "Fasting Blood Sugar", min_value=0.0, value=95.0
            )

            family_history = st.selectbox(
                "Family History", ["No", "Yes"]
            )

        submit = st.form_submit_button(
            "Predict Heart Disease",
            use_container_width=True
        )

    if submit:

        patient_data = {
            "age": age,
            "resting_bp_systolic": systolic,
            "resting_bp_diastolic": diastolic,
            "cholesterol_total": cholesterol,
            "hdl": hdl,
            "ldl": ldl,
            "triglycerides": triglycerides,
            "fasting_blood_sugar": fasting_sugar,
            "sex_Male": 1 if sex == "Male" else 0,
            "family_history_True": 1 if family_history == "Yes" else 0
        }

        patient_df = pd.DataFrame([patient_data])
        patient_df = patient_df[features]

        scaled_data = scaler.transform(patient_df)

        prediction = model.predict(scaled_data)[0]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("Model Prediction: Heart Disease Detected")
        else:
            st.success("Model Prediction: No Heart Disease Detected")

        probability = model.predict_proba(scaled_data)[0][1]

        st.write("Model-estimated probability for class 1:")
        st.progress(float(probability))
        st.write(f"{probability * 100:.2f}%")

        st.caption(
            "This probability is the model's output, not a clinically "
            "validated estimate of an individual's actual disease risk."
        )

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "Heart Disease Risk Prediction | Machine Learning Project | "
    "For Educational Purposes Only"
)
