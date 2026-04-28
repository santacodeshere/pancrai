import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()
import os

print("=== Environment Check ===")
print(f"GEMINI_API_KEY : {'SET — ' + os.getenv('GEMINI_API_KEY','')[:15] if os.getenv('GEMINI_API_KEY') else 'MISSING'}")
print(f"GROQ_API_KEY   : {'SET — ' + os.getenv('GROQ_API_KEY','')[:15] if os.getenv('GROQ_API_KEY') else 'MISSING'}")
print(f"ANTHROPIC_API_KEY: {'SET — ' + os.getenv('ANTHROPIC_API_KEY','')[:15] if os.getenv('ANTHROPIC_API_KEY') else 'MISSING'}")

print("\n=== Testing Groq ===")
groq_key = os.getenv('GROQ_API_KEY','')
if groq_key:
    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":"Say hello in 3 words"}],
            max_tokens=20
        )
        print(f"Groq OK: {r.choices[0].message.content}")
    except Exception as e:
        print(f"Groq error: {e}")
else:
    print("Groq key missing")

print("\n=== Testing Gemini ===")
gem_key = os.getenv('GEMINI_API_KEY','')
if gem_key:
    try:
        import google.generativeai as genai
        genai.configure(api_key=gem_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        r = model.generate_content("Say hello in 3 words")
        print(f"Gemini OK: {r.text}")
    except Exception as e:
        print(f"Gemini error: {e}")
else:
    print("Gemini key missing")
