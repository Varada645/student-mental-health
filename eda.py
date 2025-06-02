import plotly.express as px
import pandas as pd
from pymongo import MongoClient
import os

def perform_eda():
    static_dir = "static"
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    client = MongoClient("mongodb://localhost:27017/")
    db = client["Mental_Health"]
    collection = db["students_mental_health"]
    
    data = list(collection.find({}, {"Stress_Level": 1, "Sleep_Quality": 1, "Physical_Activity": 1, "CGPA": 1, "_id": 0}))
    df = pd.DataFrame(data)
    
    # 1. Stress Distribution
    fig1 = px.histogram(df, x="Stress_Level", nbins=6, title="Stress Level Distribution")
    fig1.write_html(os.path.join(static_dir, "stress_distribution.html"))

    # 2. Stress by Sleep Quality
    fig2 = px.box(df, x="Sleep_Quality", y="Stress_Level", title="Stress Level by Sleep Quality")
    fig2.write_html(os.path.join(static_dir, "stress_by_sleep.html"))

    # 3. Stress by Physical Activity
    fig3 = px.box(df, x="Physical_Activity", y="Stress_Level", title="Stress Level by Physical Activity")
    fig3.write_html(os.path.join(static_dir, "stress_by_activity.html"))

    # 4. Correlation Heatmap
    corr = df[["Stress_Level", "CGPA"]].corr()
    fig4 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", title="Correlation Heatmap")
    fig4.write_html(os.path.join(static_dir, "correlation_heatmap.html"))
    
    print("EDA plots saved in 'static/' directory as interactive HTML files")

if __name__ == "__main__":
    perform_eda()