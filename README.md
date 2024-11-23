*Preparing:*

1. Create folder model and put the file dataset .h5, app.py, requirements.txt and Dockerfile

*Installation:*

Install dependencies:
1. pip install -r requirements.txt

2. If you encounter any error, try to install:
pip install tensorflow pillow fastapi uvicorn python-multipart opencv-python-headless imutils

3. Run the FastAPI web server:
uvicorn app:app --reload

4. Try it out:
http://127.0.0.1:8000

*Deployed*

1. gcloud builds submit --tag gcr.io/deploy-model1/fast-api

2. gcloud run deploy --image gcr.io/deploy-model1/fast-api --platform managed