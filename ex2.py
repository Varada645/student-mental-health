from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017/")
db = client["Mental_Health"]
collection = db["students_mental_health"]
print(list(collection.find().limit(5)))