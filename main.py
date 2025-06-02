import threading
from preprocess import preprocess_and_upload_data
from eda import perform_eda
from alerts import monitor_stress_alerts
from app import app

def main():
    # Preprocess and upload data
    preprocess_and_upload_data()
    
    # Perform EDA
    perform_eda()
    
    # Start real-time monitoring in a separate thread
    alert_thread = threading.Thread(target=monitor_stress_alerts, args=(5,), daemon=True)
    alert_thread.start()
    
    # Run Flask app
    app.run(debug=True, host="0.0.0.0", port=5001, use_reloader=False)

if __name__ == "__main__":
    main()