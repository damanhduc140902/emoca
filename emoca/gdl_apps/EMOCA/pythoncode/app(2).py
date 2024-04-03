from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
import shutil
import os
from PIL import Image
from data_preprocessing import load_data, expression_labels
from model_prediction import load_saved_models, predict_emotion

app = FastAPI()

# Assuming your models are stored in a 'models' directory with version '91'
MODEL_DIR = "models"
MODEL_VERSION = "91"

# Load models once when the server starts
models = load_saved_models(MODEL_DIR, MODEL_VERSION)

@app.post("/predict-emotion")
async def predict_emotion_route(file: UploadFile = File(...)):
    try:
        # Save uploaded file to disk
        input_folder = "temp_images"
        if not os.path.exists(input_folder):
            os.makedirs(input_folder)
        temp_file_path = os.path.join(input_folder, file.filename)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Resize the image to 224x224
        with Image.open(temp_file_path) as img:
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            img.save(temp_file_path)  # Overwrite the original image

        # Process the image and load data
        data = load_data(temp_file_path)  # Adjusted to send the file path directly
        
        # Predict emotion
        prediction = predict_emotion(data, models)
        predicted_emotion = expression_labels[int(prediction[0])]
        
        # Clean up: remove temporary files
        # shutil.rmtree(input_folder)
        
        # Return the prediction
        return {"emotion": predicted_emotion}
    except Exception as e:
        # In case of error, return the error message
        return JSONResponse(status_code=400, content={"message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
