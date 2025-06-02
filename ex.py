from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017/")
db = client["Mental_Health"]
db["students_mental_health"].insert_one({"Course": "Test", "Stress_Level": 5, "Sleep_Quality": "Poor"})