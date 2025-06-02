# Student Mental Health Analysis Dashboard

A Flask-based web dashboard to monitor and predict student stress using machine learning and visualizations.

## Features
- **Monitor**: Real-time high-stress alerts (Stress_Level > 4).
- **Predict**: Random Forest (~85% accuracy) and Logistic Regression.
- **Cluster**: K-Means for student groups.
- **Visualize**: Course balance pie chart (33.3% Engineering), stress by course bar chart.
- **Correlations**: Table shows ~0.75 stress-anxiety link.
- **Recommendations**: Suggests sleep workshops.
- **UI**: Responsive, light blue-gray background.

## Setup
1. **Install MongoDB**:
   - Download from [MongoDB](https://www.mongodb.com/try/download/community).
   - Start: `mongod --dbpath C:\data\db`.
2. **Clone Repository**:
   ```bash
   git clone https://github.com/your-username/student-mental-health-dashboard.git
   cd student-mental-health-dashboard
