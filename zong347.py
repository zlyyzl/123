
import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import sqlite3
import hashlib
from PIL import Image
import pickle
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score, f1_score 
from pycaret import classification
from pycaret.classification import setup, get_logs,create_model, predict_model, tune_model, save_model,load_model
import shap
shap.initjs()
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import streamlit.components.v1 as components
import openpyxl
import os
import plotly.express as px

def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    conn = get_db_connection()
    try:
        conn.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                     (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def validate_user(username, password):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?",
                        (username, hash_password(password))).fetchone()
    conn.close()
    return user is not None

def login_page():
    st.title("User Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if validate_user(username, password):
                st.session_state['is_logged_in'] = True
                st.success("Login successful!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password.")

def register_page():
    st.title("User Registration")

    with st.form("register_form"):
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        register_button = st.form_submit_button("Register")

        if register_button:
            if register_user(new_username, new_password):
                st.success("Registration successful! Please log in.")
            else:
                st.error("That username is already taken. Please try another username.")
                
def prediction_page():
    import streamlit as st

    # Set the page title
    st.set_page_config(page_title="My Application", layout="centered")


    page = st.sidebar.selectbox("Select a Page", ["Home Page", "Prediction"])

    if page == "Home Page":
        st.title("Welcome to Functional outcome prediction App for patients with posterior circulation large vessel occlusion after mechanical thrombectomy")
    
        st.header("Summary")
        st.write("""
            This application aims to predict functional outcome in  patients with posterior circulation large vessel occlusion following mechanical thrombectomy，thus facilitates informed clinical judgment, supports personalized treatment and follow-up plans, and establishes realistic treatment expectations.
        """)

        st.header("Main Features")
        features = [
            "✔️ Implementation of preoperative, intraoperative, and postoperative prediction models to dynamically update predictions of functional outcomes.",
            "✔️ Support for batch predictions of functional outcomes for multiple patients.",
            "✔️ Ability to predict outcomes for patients with missing variable values.",
            "✔️ Facilitation of the interpretation of how the model provides personalized predictions for specific cases.",
            "✔️ Consideration of changing environments with automatic deployment of updated prediction models."
        ]
        for feature in features:
            st.write(feature)

        st.header("How to Use")
        st.markdown("""
            To the left, is a dropdown main menu for navigating to each page in the present App:<br><br>
            
            &bull; **Home Page:** We are here!<br>
            
            &bull; **Prediction:** Overview of the prediction section.<br>
            
            &bull; **Preoperative_number:** Manage preoperative predictions by inputting the necessary data.<br>
            
            &bull; **Preoperative_batch:** Process preoperative batch predictions by uploading a file.<br>
            
            &bull; **Perioperative_number:** Manage perioperative predictions by inputting the necessary data.<br>
            
            &bull; **Perioperative_batch:** Process perioperative batch predictions by uploading a file.<br>
            
            &bull; **Postoperative_number:** Manage postoperative predictions by inputting the necessary data.<br>
            
            &bull; **Postoperative_batch:** Process postoperative batch predictions by uploading a file.<br>
            
            """, unsafe_allow_html=True)

        pdf_file_path = r"123.pdf"
        if os.path.exists(pdf_file_path):
            st.markdown("Click here to download the manual for more detailed usage instructions:")
            with open(pdf_file_path, "rb") as f:
                st.download_button(label="User Manual.pdf",data=f, file_name="User Manual.pdf",mime="application/pdf" )
        else:
            st.error("指定的文件不存在，请检查文件路径。")
        
        st.header("Contact Us")
        st.write("""
            If you have any questions, please contact the support team:
            - Email: 2894683001@qq.com
            
        """)
        
        st.header("Useful Links")
        st.markdown(
        """
        An app designed to predict functional outcomes for patients with anterior circulation large vessel occlusion following mechanical thrombectomy:
         - [Visit our partner site](https://zhelvyao-123-60-anterior.streamlit.app/)
         """)

        st.markdown(
            f"""
            <style>
                .appview-container .main .block-container{{
                    max-width: {1150}px;
                    padding-top: {5}rem;
                    padding-right: {10}rem;
                    padding-left: {10}rem;
                    padding-bottom: {5}rem;
                }}
                .reportview-container .main {{
                    color: white;
                    background-color: black;
                }}
            </style>
            """,
            unsafe_allow_html=True
        )
    
        image = Image.open('it.tif')
        st.image(image, use_column_width=True)

    elif page == "Prediction":
        st.title('Functional outcome prediction App for patients with posterior circulation large vessel occlusion after mechanical thrombectomy')

        model = joblib.load('tuned_rf_pre_BUN.pkl')
        model2 = load_model('tuned_rf_pre_BUN_model')
        model3 = joblib.load('tuned_rf_intra_BUN.pkl')
        model4 = load_model('tuned_rf_intra_BUN_model')
        model5 = joblib.load('tuned_rf_post_BUN.pkl')
        model6 = load_model('tuned_rf_post_BUN_model')
    
        class DynamicWeightedForest:

            def __call__(self, X):
                return self.predict_proba(X)
                
            def __init__(self, base_trees):
                self.trees = base_trees
                self.tree_weights = np.ones(len(self.trees)) / len(self.trees)
    
            def predict_proba(self, X):
                weighted_votes = np.zeros((X.shape[0], 2))
                for i, tree in enumerate(self.trees):
                    proba = tree.predict_proba(X)
                    weighted_votes += self.tree_weights[i] * proba
                return weighted_votes / np.sum(self.tree_weights)

            def get_weighted_shap_values(self, X):
                shap_values_sum = np.zeros((X.shape[0], X.shape[1]))
                expected_value_sum = 0
            
                for tree, weight in zip(self.trees, self.tree_weights):
                    explainer = shap.TreeExplainer(tree)
                    shap_values_tree = explainer.shap_values(X)[1]  # 正类的 SHAP 值
                    expected_value_tree = explainer.expected_value[1]
            
                    shap_values_sum += weight * shap_values_tree
                    expected_value_sum += weight * expected_value_tree
            
                return shap_values_sum, expected_value_sum

    
            def update_weights(self, X, y):
                for i, tree in enumerate(self.trees):
                    predictions = tree.predict(X)
                    accuracy = np.mean(predictions == y)
                    self.tree_weights[i] = accuracy
                self.tree_weights /= np.sum(self.tree_weights)
    
            def add_tree(self, new_tree):
                self.trees.append(new_tree)
                self.tree_weights = np.append(self.tree_weights, [1.0])
                self.tree_weights /= np.sum(self.tree_weights)
    
            def save_model(self, model_name):
                joblib.dump(self, model_name)
    
            @staticmethod
            def load_model(model_name):
                if os.path.exists(model_name):
                    return joblib.load(model_name)
                return None
    
        def load_hospital_model(hospital_id):
            model_file = f'{hospital_id}_weighted_forest.pkl'
            if os.path.exists(model_file):
                return DynamicWeightedForest.load_model(model_file)
            else:
                initial_model = joblib.load('tuned_rf_pre_BUN.pkl')
                return DynamicWeightedForest(initial_model.estimators_)

        hospital_id = st.sidebar.selectbox("Select Hospital ID:", ["Hospital_A", "Hospital_B", "Hospital_C"])
        current_model = load_hospital_model(hospital_id)
    
        def st_shap(plot):
            shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
            components.html(shap_html)
    
        prediction_type = st.sidebar.selectbox(
            "How would you like to predict?",
            ("Preoperative_number", "Preoperative_batch", "Intraoperative_number", "Intraoperative_batch", 
             "Postoperative_number", "Postoperative_batch")
        )
    
        if prediction_type == "Preoperative_number":
            st.subheader("Preoperative Number Prediction")
            st.write("Please fill in the blanks with corresponding data.")
    
            features = {
                'NIHSS': st.number_input('NIHSS', min_value=4, max_value=38, value=10),
                'GCS': st.number_input('GCS', min_value=0, max_value=15, value=10),
                'pre_eGFR': st.number_input('pre_eGFR', min_value=10.00, max_value=250.00, value=111.5),
                'pre_glucose': st.number_input('pre_glucose', min_value=2.50, max_value=25.00, value=7.78),
                'PC_ASPECTS': st.number_input('PC_ASPECTS', min_value=0.0, max_value=10.0, value=8.0),
                'Age': st.number_input('Age', min_value=0, max_value=120, value=60),
                'pre_BUN': st.number_input('pre_BUN', min_value=0.00, max_value=30.00, value=10.20)
            }
    
            input_df = pd.DataFrame([features])
    
            if st.button('Predict'):
                try:
                    output = current_model.predict_proba(input_df)[:, 1]
                    explainer = shap.Explainer(current_model)  
                    shap_values, expected_value = current_model.get_weighted_shap_values(input_df)
    
                    st.write('Based on feature values, predicted probability of good functional outcome is: ' + str(output))
                    st_shap(shap.force_plot(expected_value, shap_values[0], input_df))
    
                    shap_df = pd.DataFrame({
                        'Feature': input_df.columns,
                        'SHAP Value': shap_values[0]
                    })
                    st.write("SHAP values for each feature:")
                    st.dataframe(shap_df)
    
                    # 渲染 Add Data for Learning 按钮
                    label = st.selectbox('Outcome for Learning', [0, 1])
                    if st.button('Add Data for Learning'):
                        st.write("Button clicked!")  # 验证点击事件是否触发
                        try:
                            new_tree = DecisionTreeClassifier(random_state=42)
                            st.write("Initialized new Decision Tree.")
                            new_tree.fit(input_df, [label])
                            st.write("Fitted new tree with input data.")
                            current_model.add_tree(new_tree)
                            st.write("Added new tree to the model.")
                            current_model.update_weights(input_df, [label])
                            st.write("Updated tree weights.")
                            current_model.save_model(f'{hospital_id}_weighted_forest.pkl')
                            st.success("New tree added and weights updated dynamically! Model saved successfully.")
                        except Exception as e:
                            st.error(f"Error during model update: {e}")

                    

                    
  
  
def main():
    initialize_database()  

    if 'is_logged_in' not in st.session_state:
        st.session_state['is_logged_in'] = False

    if st.session_state['is_logged_in']:
        prediction_page()  
    else:
        login_or_register = st.sidebar.selectbox("Select an action:", ("Login", "Register"), key="login_register_selectbox")
        if login_or_register == "Login":
            login_page()
        else:
            register_page()

if __name__ == "__main__":
    main()
