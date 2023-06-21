from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
from datetime import datetime
import time

from classifier import NewsCategoryClassifier


class PredictRequest(BaseModel):
    source: str
    url: str
    title: str
    description: str

    def model_dump(self, *args, **kwargs) -> dict:
        super().model_dump(*args, **kwargs)


class PredictResponse(BaseModel):
    scores: dict
    label: str

    def model_dump(self, *args, **kwargs) -> dict:
        super().model_dump(*args, **kwargs)


MODEL_PATH = "../data/news_classifier.joblib"
LOGS_OUTPUT_PATH = "../data/logs.out"

app = FastAPI()


@app.on_event("startup")
def startup_event():
    """
    [TO BE IMPLEMENTED]
    1. Initialize an instance of `NewsCategoryClassifier`.
    2. Load the serialized trained model parameters (pointed to by `MODEL_PATH`) into the NewsCategoryClassifier you initialized.
    3. Open an output file to write logs, at the destimation specififed by `LOGS_OUTPUT_PATH`
        
    Access to the model instance and log file will be needed in /predict endpoint, make sure you
    store them as global variables
    """
    global classifier
    global log_handler_id

    log_handler_id = logger.add(LOGS_OUTPUT_PATH, enqueue=True)    

    classifier = NewsCategoryClassifier()
    classifier.load(MODEL_PATH)
    logger.info("Setup completed")


@app.on_event("shutdown")
def shutdown_event():
    # clean up
    """
    [TO BE IMPLEMENTED]
    1. Make sure to flush the log file and close any file pointers to avoid corruption
    2. Any other cleanups
    """
    classifier = None
    logger.info("Shutting down application")
    try:
        logger.remove(log_handler_id)
    except:
        print("Exception cleaning up the log file handle")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # get model prediction for the input request
    # construct the data to be logged
    # construct response
    """
    [TO BE IMPLEMENTED]
    1. run model inference and get model predictions for model inputs specified in `request`
    2. Log the following data to the log file (the data should be logged to the file that was opened in `startup_event`)
    {
        'timestamp': <YYYY:MM:DD HH:MM:SS> format, when the request was received,
        'request': dictionary representation of the input request,
        'prediction': dictionary representation of the response,
        'latency': time it took to serve the request, in millisec
    }
    3. Construct an instance of `PredictResponse` and return
    """
    request_dict = {
        'source': request.source,
        'url': request.url,
        'title': request.title,
        'description': request.description
    }

    start = time.time()
    
    pred_label = classifier.predict_label(request_dict)
    label_prob = classifier.predict_proba(request_dict)

    end = time.time()
    latency = end - start

    response = PredictResponse(scores={**label_prob}, label=pred_label)
    response_dict = {
        'scores': label_prob,
        'label': pred_label
    }

    log_etnry = {
        'timestamp': datetime.now().strftime('%Y:%m:%d %H:%M:%S'),
        'request': request_dict,
        'prediction': response_dict,
        'latency': latency
    }

    logger.info(log_etnry)
    return response


@app.get("/")
def read_root():
    return {"Hello": "World"}
