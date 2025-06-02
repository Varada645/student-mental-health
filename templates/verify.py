from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017/")
db = client["Mental_Health"]
collection = db["students_mental_health"]
collection.insert_one({
    "Course": "TestCourse",
    "Stress_Level": 5,
    "Sleep_Quality": "Poor",
    "Physical_Activity": "Low",
    "Diet_Quality": "Poor",
    "CGPA": 3.0,
    "Gender": "Male",
    "Financial_Stress": 2
})