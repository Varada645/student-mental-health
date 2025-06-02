# Student Mental Health Evaluation Analysis Project

## Overview
The project aims to monitor student mental health by analyzing stress levels and identifying high-risk cases. It provides a web-based dashboard for real-time tracking and insights.

## Technologies Used

### Frontend
- **HTML/CSS**: For structuring and styling the dashboard.
- **JavaScript**: 
  - **Plotly**: For creating interactive visualizations.
  - **Chart.js**: For rendering dynamic charts.
  - **Fetch API**: For asynchronous data retrieval from backend endpoints.

### Backend
- **Python**: Core programming language for data processing and server logic.
- **Flask**: Lightweight web framework for serving the dashboard and API endpoints.
- **MongoDB**: NoSQL database for storing student mental health data.
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For implementing machine learning models.

## Models Implemented
1. **Logistic Regression**: Used for predicting high stress levels based on various predictors.
2. **Random Forest Classifier**: Another model for stress prediction, providing feature importance metrics.
3. **K-Means Clustering**: Used for clustering students based on their stress levels and other features.

## Key Files
- **app.py**: Main Flask application that sets up routes and handles API requests.
- **preprocess.py**: Loads and cleans the dataset, uploading it to MongoDB.
- **model.py**: Trains the logistic regression and random forest models, evaluates their performance, and updates the database with clustering results.
- **eda.py**: Generates exploratory data analysis plots and saves them as interactive HTML files in the static directory.

## Data Flow
1. **Data Loading**: The dataset is loaded from a CSV file and uploaded to MongoDB.
2. **Data Preprocessing**: Missing values are imputed, and categorical variables are converted to dummy variables.
3. **Model Training**: The models are trained on the preprocessed data, and performance metrics are calculated.
4. **Exploratory Data Analysis**: EDA plots are generated to visualize relationships between stress levels and various factors.

## Insights and Findings
- The project identifies high-stress students and provides insights into factors affecting stress levels, such as sleep quality and physical activity.
- The predictive models help in understanding the patterns and correlations in the data, guiding mental health initiatives.
- Recommendations for interventions can be made based on the analysis, such as promoting better sleep habits and physical activities among students.

## Conclusion
The "Student Mental Health Analysis" project successfully delivers a functional dashboard to monitor and analyze student stress, meeting its core objectives. It lays a solid foundation for understanding student mental health trends and can evolve into a robust support tool with targeted enhancements.
