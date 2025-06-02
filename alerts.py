from pymongo import MongoClient
import time

def monitor_stress_alerts(poll_interval=5):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["Mental_Health"]
    collection = db["students_mental_health"]
    alerted_students = set()
    
    print("Monitoring for high stress levels (polling every {} seconds)...".format(poll_interval))
    while True:
        high_stress_students = collection.find({"Stress_Level": {"$gt": 4}})
        for student in high_stress_students:
            student_id = str(student["_id"])
            if student_id not in alerted_students:
                print(f"ALERT: Student {student_id} has high stress (Level: {student['Stress_Level']})")
                alerted_students.add(student_id)
        time.sleep(poll_interval)

if __name__ == "__main__":
    monitor_stress_alerts(poll_interval=5)                          