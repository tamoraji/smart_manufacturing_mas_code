
import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')

def list_available_models():
    """
    Connects to the Google Gemini API and lists all available generative models
    that support the 'generateContent' method.
    """
    try:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            logging.error("GEMINI_API_KEY not found or not set in .env file.")
            return

        genai.configure(api_key=api_key)

        logging.info("Fetching available Gemini models...")
        print("\n--- Available Models for 'generateContent' ---")
        count = 0
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                print(model.name)
                count += 1
        print("------------------------------------------------\n")
        
        if count == 0:
            logging.warning("No models supporting 'generateContent' were found. Your API key may have restrictions.")
        else:
            logging.info(f"Found {count} available model(s). Please update 'llm_planner_agent.py' with one of the model names listed above.")

    except Exception as e:
        logging.error(f"An error occurred while trying to list models: {e}", exc_info=True)

if __name__ == "__main__":
    list_available_models()
