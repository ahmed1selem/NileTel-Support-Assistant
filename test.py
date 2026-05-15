import requests

# Put the exact URL that worked in your curl command here:
URL = "https://niletel111.app.n8n.cloud/webhook/029d940f-1645-461f-a933-86ce6cc83fe7"

print(f"Aiming at: {URL}")
print("Firing the webhook from Python...")

try:
    payload = {"query": "Sledgehammer Test", "answer": "Python is working"}
    response = requests.post(URL, json=payload, timeout=10)
    
    print("--- RESULTS ---")
    print(f"Status Code: {response.status_code}")
    print(f"Response Text: {response.text}")
    print("Check your Google Sheet right now!")

except Exception as e:
    print(f"CRITICAL FIREWALL/NETWORK ERROR: {e}")