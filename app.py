import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("data_for_modeling.csv")
daily_data = pd.read_csv("daily_data.csv")

# Load the trained SARIMA model
model = joblib.load("sarima_model_SARIMAX.pkl")

# Define exog_cols globally
exog_cols = ['scheduled_pickups', 'actual_pickup_lag7', 'actual_pickup_lag14',
             'scheduled_pickup_lag7', 'scheduled_pickup_lag14', 'family_size',
             'special_occasion_Dhu_al_Qadah', 'special_occasion_Eid_al_Adha', 'special_occasion_Eid_al_Fitr',
             'special_occasion_Muharram', 'special_occasion_Rabi_al_Awwal', 'special_occasion_Rajab',
             'special_occasion_Ramadan', 'special_occasion_Shaban', 'season_Summer', 'season_Winter', 'season_Spring']

# Page 1: Dashboard
def dashboard():
    st.title("Food Hamper Pickup Prediction Dashboard")
    st.write("This dashboard helps predict the number of food hamper pickups based on historical data.")

# Page 2: Exploratory Data Analysis (EDA)
def exploratory_data_analysis():
    st.title("Exploratory Data Analysis")
    st.write("Exploring the dataset used for modeling.")
    st.write(data.head())
    
    # Example Visualization
    fig = px.histogram(data, x="actual_pickups", title="Distribution of Actual Pickups")
    st.plotly_chart(fig)

    # Distribution of actual pickups
    plt.figure(figsize=(10,5))
    sns.histplot(daily_data['actual_pickups'], bins=30, kde=True)
    plt.title('Distribution of Actual Hamper Pickups')
    plt.show()

    # Time series plot of actual pickups
    plt.figure(figsize=(12,5))
    plt.plot(daily_data['pickup_date'], daily_data['actual_pickups'], label='Actual Pickups')
    plt.xlabel('Date')
    plt.ylabel('Number of Pickups')
    plt.title('Daily Actual Hamper Pickups Over Time')
    plt.legend()
    plt.show()

# Page 3: Machine Learning Modeling
def machine_learning_modeling():
    st.title("Food Hamper Pickup Prediction")
    
    # Get user input
    input_features = get_user_input()
    st.subheader("User Input Parameters")
    st.write(input_features)
    
    # Save the input_features to session state
    st.session_state.input_features = input_features
    
    if st.button("Predict"):
        prediction = predict_hamper_pickups(input_features)
        st.subheader("Prediction")
        st.write(prediction)

    # Add button to generate the XAI report
    if st.button("Generate XAI Report"):
        xai_report()  # Now no need to pass input_features to this function

# Page 4: XAI Report
def xai_report():
    # Access input_features from session state
    input_features = st.session_state.input_features
    
    st.title("XAI Report for Demand Prediction")
    
    # Define a custom prediction function
    def predict_fn(input_data):
        # Make sure input_data is a DataFrame with one row
        input_data = pd.DataFrame(input_data, columns=input_features.columns)
        
        # Ensure the input features are in the right format (type and columns)
        input_data = input_data.reindex(columns=exog_cols, fill_value=0)
        input_data = input_data.astype(float)
        
        # Make sure we are passing only a single row of data (1, 17)
        if input_data.shape[0] != 1:
            input_data = input_data.iloc[:1]  # Ensure we only pass the first row
        
        # Predict using SARIMAX (exogenous variables)
        start = len(data)
        end = start
        prediction = model.predict(start=start, end=end, exog=input_data)
        
        return prediction
    
    # Use SHAP's KernelExplainer
    # Ensure the data used here also has the same shape as the input passed to the model (1 row, 17 columns)
    explainer = shap.KernelExplainer(predict_fn, data.drop(columns=["actual_pickups"]).iloc[:1])  # Only take the first row
    shap_values = explainer.shap_values(data.drop(columns=["actual_pickups"]).iloc[:1])  # Ensure consistent shape
    
    st.subheader("SHAP Feature Importance")
    
    # Generate SHAP Feature Importance Plot
    shap_summary_plot = generate_shap_summary_plots(shap_values, data.drop(columns=["actual_pickups"]).iloc[:1])
    
    # Render the plot
    st.plotly_chart(shap_summary_plot)

    st.write("SHAP analysis highlights the most influential features in predicting food hamper pickups.")
    
# Function to generate SHAP Feature Importance Plot (using Plotly)
def generate_shap_summary_plots(shap_values, X_shap):
    # Compute feature importances (Mean Absolute SHAP Values)
    shap_values_array = shap_values  # No need for .values as shap_values is already a numpy.ndarray
    mean_shap_values = np.mean(np.abs(shap_values_array), axis=0)
    feature_names = X_shap.columns

    # Create Plotly bar chart for feature importance
    fig = go.Figure(data=[go.Bar(
        x=mean_shap_values,
        y=feature_names,
        orientation='h',
        marker=dict(color='royalblue')
    )])

    # Update layout
    fig.update_layout(
        title="SHAP Feature Importance",
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="Feature",
        yaxis=dict(categoryorder='total ascending'),
        template="plotly_white"
    )

    return fig

# Function to get user input
def get_user_input():
    st.sidebar.header("User Input Parameters")
    
    pickup_date = st.sidebar.date_input("Pickup Date")
    actual_pickups = st.sidebar.number_input("Actual Pickups", min_value=0)
    actual_pickup_boxcox = st.sidebar.number_input("Actual Pickup Boxcox")
    scheduled_pickups = st.sidebar.number_input("Scheduled Pickups", min_value=0)
    actual_pickup_lag7 = st.sidebar.number_input("Actual Pickup Lag 7")
    actual_pickup_lag14 = st.sidebar.number_input("Actual Pickup Lag 14")
    scheduled_pickup_lag7 = st.sidebar.number_input("Scheduled Pickup Lag 7")
    scheduled_pickup_lag14 = st.sidebar.number_input("Scheduled Pickup Lag 14")
    family_size = st.sidebar.number_input("Family Size", min_value=0)
    
    special_occasions = {
        "Dhu al-Qadah": st.sidebar.checkbox("Special Occasion: Dhu al-Qadah"),
        "Eid al-Adha": st.sidebar.checkbox("Special Occasion: Eid al-Adha"),
        "Eid al-Fitr": st.sidebar.checkbox("Special Occasion: Eid al-Fitr"),
        "Muharram": st.sidebar.checkbox("Special Occasion: Muharram"),
        "Rabi al-Awwal": st.sidebar.checkbox("Special Occasion: Rabi al-Awwal"),
        "Rajab": st.sidebar.checkbox("Special Occasion: Rajab"),
        "Ramadan": st.sidebar.checkbox("Special Occasion: Ramadan"),
        "Shaban": st.sidebar.checkbox("Special Occasion: Shaban")
    }
    
    season = st.sidebar.radio("Season", ("Summer", "Winter", "Spring"))
    
    user_data = {
        "actual_pickups": [actual_pickups],
        "actual_pickup_boxcox": [actual_pickup_boxcox],
        "scheduled_pickups": [scheduled_pickups],
        "actual_pickup_lag7": [actual_pickup_lag7],
        "actual_pickup_lag14": [actual_pickup_lag14],
        "scheduled_pickup_lag7": [scheduled_pickup_lag7],
        "scheduled_pickup_lag14": [scheduled_pickup_lag14],
        "family_size": [family_size],
        **{key: [value] for key, value in special_occasions.items()},
        "season_Summer": [1 if season == "Summer" else 0],
        "season_Winter": [1 if season == "Winter" else 0],
        "season_Spring": [1 if season == "Spring" else 0]
    }
    
    return pd.DataFrame(user_data)

# Function to make predictions
def predict_hamper_pickups(input_features):
    # Ensure input features match expected columns
    input_features = input_features.reindex(columns=exog_cols, fill_value=0)
    input_features = input_features.astype(float)

    start = len(data)
    end = start
    prediction = model.predict(start=start, end=end, exog=input_features)
    
    return prediction

# Main App Logic
def main():
    st.sidebar.title("Food Drive App")
    app_page = st.sidebar.radio("Select a Page", ["Dashboard", "EDA", "ML Modeling", "XAI Report"])
    
    if app_page == "Dashboard":
        dashboard()
    elif app_page == "EDA":
        exploratory_data_analysis()
    elif app_page == "ML Modeling":
        machine_learning_modeling()
    elif app_page == "XAI Report":
        xai_report()  # No arguments required here

if __name__ == "__main__":
    main()
