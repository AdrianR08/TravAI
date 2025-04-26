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
