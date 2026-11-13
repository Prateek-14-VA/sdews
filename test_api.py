import requests

# 1. Login to get the session cookie
session = requests.Session()
login_url = "http://127.0.0.1:5000/login"
login_data = {"username": "admin", "password": "admin123"}

print("Logging in...")
login_resp = session.post(login_url, data=login_data)
if "Welcome" not in login_resp.text and "Dashboard" not in login_resp.text:
    print(f"Login failed. Status: {login_resp.status_code}")
else:
    print("Login successful!")

# 2. Define the student features
features = {
    "Marital status": 1, "Application mode": 17, "Application order": 5, "Course": 171,
    "Daytime/evening attendance": 1, "Previous qualification": 1, "Previous qualification (grade)": 122,
    "Nacionality": 1, "Mother's qualification": 19, "Father's qualification": 12,
    "Mother's occupation": 5, "Father's occupation": 9, "Admission grade": 127,
    "Displaced": 0, "Educational special needs": 0, "Debtor": 0, "Tuition fees up to date": 1,
    "Gender": 1, "Scholarship holder": 0, "Age at enrollment": 20, "International": 0,
    "Curricular units 1st sem (credited)": 0, "Curricular units 1st sem (enrolled)": 6,
    "Curricular units 1st sem (evaluations)": 6, "Curricular units 1st sem (approved)": 6,
    "Curricular units 1st sem (grade)": 14, "Curricular units 1st sem (without evaluations)": 0
}

payload = {
    "stage": "sem",
    "features": features
}

# 3. Send the request to the API
print("\nSending prediction request...")
api_url = "http://127.0.0.1:5000/api/predict"
response = session.post(api_url, json=payload)

# 4. Print the result
print(f"Status Code: {response.status_code}")
print("Response JSON:")
print(response.json())