# TravAI

## **Create and Activate Virtual Environment:**
```
python -m venv venv
```
### **Windows:**
```
venv\Scripts\activate
```
### **Mac/Linux:**
```
source venv/bin/activate
```

## **Dataset**
```
https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/#sample-review
```
### **Download both:**

Rhode Island:                                                                                                                                                                                                                   
	[reviews (1,777,094 reviews)](https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/review-Rhode_Island.json.gz)                                                                                                                
 	[metadata (15,941 businesses)](https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/meta-Rhode_Island.json.gz)

  **IF DOWNLOAD NAME IS NOT meta-Rhode_Island.json, PLEASE RENAME**

## Merge ReviewData + MetaData
```
python convert_and_merge.py
```

## **Install Dependencies:**
```
pip install -r requirements.txt
```

## **Run Program:** 

Start Backend: 
```
uvicorn backend.api:app --reload --port 8000
```
Start Front End:
```
cd frontend 
streamlit run web_ui.py
```
