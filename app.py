
import streamlit as st
import pandas as pd
import joblib

# --------------------------------
# 1. Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide"
)

# --------------------------------
# 2. Load Saved Machine Learning Files
# --------------------------------

@st.cache_resource
def load_files():
    model = joblib.load("heart_disease_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("feature_names.pkl")
    return model, scaler, features

model, scaler, features = load_files()

# --------------------------------
# 3. Project Title
# --------------------------------

st.title("❤️ Heart Disease Risk Prediction")
st.write(
    "A Machine Learning project to predict heart disease "
    "using patient health information."
)

st.info(
    "This application is an educational project. "
    "It is not a medical diagnosis tool."
)

st.divider()

# --------------------------------
# 4. Model Performance Data
# --------------------------------

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
    "F1-score": [0.5510, 0.4466, 0.5493, 0.4518]
})

# --------------------------------
# 5. Dashboard Tabs
# --------------------------------

tab1, tab2 = st.tabs([
    "📊 Project Dashboard",
    "❤️ Patient Prediction"
])

# =================================
# TAB 1: PROJECT DASHBOARD
# =================================

with tab1:

    st.header("Project Overview")

    # Dataset information

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", "9,000")
    col2.metric("Features", "10")
    col3.metric("Training Records", "7,200")
    col4.metric("Testing Records", "1,800")

    st.divider()

    # Model performance

    st.subheader("Model Performance")

    col1, col2 = st.columns(2)

    col1.metric(
        "Best Accuracy",
        "77.39%",
        "Random Forest"
    )

    col2.metric(
        "Best Precision",
        "69.27%",
        "Random Forest"
    )

    # Model comparison graph

    st.divider()

    st.subheader("Machine Learning Model Comparison")

    st.write(
        "Comparison of Accuracy, Precision, Recall "
        "and F1-score for all four models."
    )

    chart_data = results.set_index("Model")[
        ["Accuracy", "Precision", "Recall", "F1-score"]
    ]

    st.bar_chart(chart_data)

    # Complete model results

    st.divider()

    st.subheader("Complete Model Results")

    st.dataframe(
        results.set_index("Model").style.format("{:.2%}"),
        use_container_width=True
    )

    # Confusion matrix

    st.divider()

    st.subheader("Confusion Matrix - Logistic Regression")

    confusion = pd.DataFrame(
        [[1140, 115], [294, 251]],
        index=["Actual: No Disease", "Actual: Disease"],
        columns=["Predicted: No Disease", "Predicted: Disease"]
    )

    st.dataframe(
        confusion,
        use_container_width=True
    )

    st.caption(
        "The confusion matrix shows correct and incorrect "
        "predictions made by the Logistic Regression model."
    )

    # PCA visualization

    st.divider()

    st.subheader("PCA - Principal Component Analysis")

    st.write(
        "PCA reduces the original features into two principal "
        "components for visualization."
    )

    pca_data = pd.DataFrame({
        "Principal Component": ["PC1", "PC2"],
        "Explained Variance": [23.32, 17.52]
    })

    st.bar_chart(
        pca_data.set_index("Principal Component")
    )

    st.metric(
        "Total Explained Variance",
        "40.85%"
    )

    st.caption(
        "The first two principal components explain approximately "
        "40.85% of the total variance."
    )

# =================================
# TAB 2: PATIENT PREDICTION
# =================================

with tab2:

    st.header("Heart Disease Prediction")

    st.write(
        "Enter the patient information below to get a "
        "prediction from the trained Logistic Regression model."
    )

    # Patient input form

    with st.form("patient_form"):

        st.subheader("Patient Information")

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=45
            )

            sex = st.selectbox(
                "Sex",
                ["Male", "Female"]
            )

            systolic = st.number_input(
                "Resting Systolic Blood Pressure",
                min_value=0.0,
                value=120.0
            )

            diastolic = st.number_input(
                "Resting Diastolic Blood Pressure",
                min_value=0.0,
                value=80.0
            )

            cholesterol = st.number_input(
                "Total Cholesterol",
                min_value=0.0,
                value=190.0
            )

        with col2:

            hdl = st.number_input(
                "HDL",
                min_value=0.0,
                value=50.0
            )

            ldl = st.number_input(
                "LDL",
                min_value=0.0,
                value=110.0
            )

            triglycerides = st.number_input(
                "Triglycerides",
                min_value=0.0,
                value=130.0
            )

            fasting_sugar = st.number_input(
                "Fasting Blood Sugar",
                min_value=0.0,
                value=95.0
            )

            family_history = st.selectbox(
                "Family History of Heart Disease",
                ["No", "Yes"]
            )

        submit = st.form_submit_button(
            "Predict",
            use_container_width=True
        )

    # Prediction

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

        # Arrange features in training order

        patient_df = patient_df[features]

        # Apply the saved scaler

        patient_scaled = scaler.transform(patient_df)

        # Make prediction

        prediction = model.predict(patient_scaled)

        st.divider()

        st.subheader("Prediction Result")

        if prediction[0] == 1:

            st.error(
                "Model Prediction: Heart Disease Detected"
            )

        else:

            st.success(
                "Model Prediction: No Heart Disease Detected"
            )

        # Display model-estimated probability if available

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(patient_scaled)[0][1]

            st.write(
                "Model-estimated probability for class 1:"
            )

            st.progress(float(probability))

            st.metric(
                "Model Probability",
                f"{probability * 100:.2f}%"
            )

        st.warning(
            "This prediction is generated by a machine learning "
            "model. It is not a medically validated risk estimate "
            "and must not be used for medical decisions."
        )
