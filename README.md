# TravAI

Create and Activate Virtual Envirnment:

python -m venv venv

Windows:
venv\Scripts\activate

Mac/Linx:
source venv/bin/activate


pip install -r requirements.txt

Starting Program:

Start Backend: 
uvicorn backend.api:app --reload --port 8000



Start front End:
streamlit run web_ui.py
