from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
import io
import requests

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# --- CONFIGURATION ---

OCR_API_KEY = "helloworld"  # OCR.space free default key
MODEL_NAME = "gemini-1.5-flash"
GEMINI_API_KEY = "AIzaSyBK5gc2fbQAOBP218EAplCHdssNf7C3hm8"

# Default values based on your dataset
DEFAULT_VALUES = {
    "N": 50.55,
    "P": 53.36,
    "K": 48.15,
    "temperature": 25.62,
    "humidity": 71.48,
    "pH": 6.47,
    "rainfall": 103.46
}

# LangChain + Gemini setup
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GEMINI_API_KEY,
    temperature=0.5
)

response_schemas = [
    ResponseSchema(name="N", description="Nitrogen value"),
    ResponseSchema(name="P", description="Phosphorus value"),
    ResponseSchema(name="K", description="Potassium value"),
    ResponseSchema(name="temperature", description="Temperature in Celsius"),
    ResponseSchema(name="humidity", description="Humidity in percentage"),
    ResponseSchema(name="pH", description="Soil pH level"),
    ResponseSchema(name="rainfall", description="Rainfall in mm")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# FastAPI app
app = FastAPI(
    title="Soil Report OCR Extractor",
    description="Extracts N, P, K, temperature, humidity, pH, and rainfall from soil report image."
)

# --- ROUTE ---

@app.post("/soil-ocr/")
async def analyze_soil_image(file: UploadFile = File(...)):
    try:
        # Step 1: Read and send image to OCR.space
        image_bytes = await file.read()
        response = requests.post(
            'https://api.ocr.space/parse/image',
            files={'filename': ('image.jpg', image_bytes)},
            data={'apikey': OCR_API_KEY, 'language': 'eng'}
        )
        result = response.json()

        if result.get("IsErroredOnProcessing"):
            raise ValueError("OCR API Error: " + result.get("ErrorMessage", ["Unknown error"])[0])

        extracted_text = result['ParsedResults'][0]['ParsedText']

        # Step 2: LLM prompt
        prompt = f"""
You are a soil report analyzer.

From the following raw soil report text, extract the following values:
- N (Nitrogen)
- P (Phosphorus)
- K (Potassium)
- temperature (in Celsius)
- humidity (in %)
- pH (soil pH level)
- rainfall (in mm)

Use this format:
{format_instructions}

Soil Report Text:
{extracted_text}
        """

        # Step 3: Invoke Gemini
        gemini_response = llm.invoke(prompt)
        structured_data = output_parser.parse(gemini_response.content)

        # Step 4: Fill missing fields with default values
        complete_data = {}
        for key, default in DEFAULT_VALUES.items():
            value = structured_data.get(key)
            if value is None or value == "":
                complete_data[key] = default
            else:
                try:
                    complete_data[key] = float(value)
                except ValueError:
                    complete_data[key] = default

        return {
            "status": "success",
            "raw_text": extracted_text,
            "parsed_data": complete_data
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

# --- MAIN ENTRYPOINT ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
