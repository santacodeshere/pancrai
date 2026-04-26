import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()
import os

key = os.getenv("GEMINI_API_KEY", "")
print(f"Key found: {bool(key)}")

# Try old package with correct model names
import google.generativeai as genai
genai.configure(api_key=key)

print("\nAvailable models:")
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(f"  {m.name}")

# Try with gemini-2.0-flash (newer free model)
print("\nTesting gemini-2.0-flash...")
try:
    model = genai.GenerativeModel("gemini-2.0-flash")
    r = model.generate_content("Say hello in one word.")
    print(f"Response: {r.text}")
    print("gemini-2.0-flash WORKS!")
except Exception as e:
    print(f"Error: {e}")

print("\nTesting gemini-1.5-flash-latest...")
try:
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    r = model.generate_content("Say hello in one word.")
    print(f"Response: {r.text}")
    print("gemini-1.5-flash-latest WORKS!")
except Exception as e:
    print(f"Error: {e}")
