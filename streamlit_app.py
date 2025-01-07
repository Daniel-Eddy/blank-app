import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Or import pickle if you saved your model with it

# Load the pre-trained Random Forest model
model = joblib.load('best_rf_model.pkl')  # Replace with your model's filename

# Streamlit app
st.title("PREDICTION OF MOTOR VEHICLE THEFT CRIME")

# Add additional information about the model and its training data
st.subheader("Model Information")
st.write("""
This model is a Random Forest classifier trained on data from the Los Angeles Police Department (LAPD) from 2020 to 2024. 
It predicts the likelihood of motor vehicle theft crimes based on historical crime data.
""")

# day_of_week
data = {'day_of_week': ["Monday", "Tuesday", "Wednesday",
                        "Thursday", "Friday", "Saturday", "Sunday"]}
df = pd.DataFrame(data)
df['day_group'] = df['day_of_week'].map({
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}).fillna(0).astype(int)
day_group_dict = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday"
}
selected_day_of_week = st.selectbox(
    "Select a Day of Week:",
    options=df['day_group'].unique(),
    format_func=lambda x: f"{day_group_dict.get(x, 'Unknown')}"
)

# area
data = {
    'area': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    'area_name': [
        "Central", "Rampart", "Southwest", "Hollenbeck", "Harbor", "Hollywood", "Wilshire",
        "West LA", "Van Nuys", "West Valley", "Northeast", "77th Street", "Newton",
        "Pacific", "N Hollywood", "Foothill", "Devonshire", "Southeast", "Mission",
        "Olympic", "Topanga"
    ]
}
df_area = pd.DataFrame(data)
area_dict = {row['area']: row['area_name'] for _, row in df_area.iterrows()}
selected_area = st.selectbox(
    "Select an area:",
    options=df_area['area'],
    format_func=lambda x: f"{area_dict[x]}"
)

# year_group
data = {'year': [2020, 2021, 2022, 2023, 2024, 2019]}
df = pd.DataFrame(data)
df['year_group'] = df['year'].map(
    {2020: 1, 2021: 2, 2022: 3, 2023: 4, 2024: 5}).fillna(0).astype(int)
year_group_dict = {
    1: "2020",
    2: "2021",
    3: "2022",
    4: "2023",
    5: "2024",
    0: "Other"
}
selected_year_group = st.selectbox(
    "Select a Year Group:",
    options=df['year_group'].unique(),
    format_func=lambda x: f"{year_group_dict.get(x, 'Unknown')}"
)

# month
month = st.slider("Month", min_value=1, max_value=12)

# day
day = st.slider("Day", min_value=1, max_value=31)

# premise_code
df = pd.read_excel('premise.xlsx')
distinct_premise_data = df[['premise_code', 'premise_description']
                           ].drop_duplicates().sort_values(by='premise_description')
selected_premise = st.selectbox(
    "Select a Premise:",
    options=df['premise_code'].unique(),
    format_func=lambda x: f"{df[df['premise_code'] == x]['premise_description'].values[0]}"
)

# Create a DataFrame for the input
input_data = pd.DataFrame({
    'day_of_week': [selected_day_of_week],
    'area': [selected_area],
    'year_group': [selected_year_group],
    'month': [month],
    'day': [day],
    'premise_code': [selected_premise]
})

# Handle categorical encoding if needed
# For example, encode 'area' and 'year_group' as per your model training

# Make a prediction
if st.button("Predict"):
    print(input_data)
    # Display user inputs
    # st.subheader("Entered Values:")
    # st.write(f"Day of the Week: {selected_day_of_week}")
    # st.write(f"Area: {selected_area}")
    # st.write(f"Year Group: {selected_year_group}")
    # st.write(f"Month: {month}")
    # st.write(f"Day: {day}")
    # st.write(f"Premise Code: {selected_premise}")
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]  # Probability of being a vehicle theft
    st.subheader(f"Prediction: {'Vehicle Theft' if prediction[0] == 1 else 'No Vehicle Theft'}")
    st.write(f"Probability of Vehicle Theft: {probability[0]:.2f}")

# Optional: Add feature importance visualization
# st.header("Feature Importance")
# st.markdown("The importance of each feature in making predictions:")
# # Replace `sorted_features` and `sorted_importances` with your feature importance data
# sorted_features = ['day_of_week', 'area',
#                    'year_group', 'month', 'day', 'premise_code']
# sorted_importances = [0.3, 0.2, 0.2, 0.15, 0.1, 0.05]
# st.bar_chart(pd.DataFrame({'Feature': sorted_features,
#              'Importance': sorted_importances}).set_index('Feature'))
