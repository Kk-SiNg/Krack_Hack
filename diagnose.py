import requests 

API_KEY = "Enter_your_API_key_here"  # <-- paste your Groq key here

# Test with a tiny request
resp = requests.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 5
    }
)

print(f"Status: {resp.status_code}")
print(f"Response: {resp.text[:500]}")
print()

# Check rate limit headers
if resp.headers:
    for key, val in resp.headers.items():
        if "rate" in key.lower() or "limit" in key.lower() or "remaining" in key.lower():
            print(f"  {key}: {val}")