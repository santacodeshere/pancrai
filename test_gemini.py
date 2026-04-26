import sys
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()
import os

key = os.getenv("GEMINI_API_KEY", "")
print(f"GEMINI_API_KEY found: {bool(key)}")
print(f"Key preview: {key[:20] if key else 'MISSING'}")
print(f"Key length: {len(key)}")

if not key:
    print("\nERROR: Key not found. Check your .env file.")
else:
    print("\nTesting Gemini connection...")
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content("Say hello in one word.")
        print(f"Gemini response: {response.text}")
        print("Gemini API working!")
    except Exception as e:
        print(f"Gemini error: {e}")
        print("\nTrying new google-genai package...")
        try:
            from google import genai as genai2
            client = genai2.Client(api_key=key)
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents="Say hello in one word."
            )
            print(f"New package response: {response.text}")
            print("New google-genai package works!")
        except Exception as e2:
            print(f"New package error: {e2}")
