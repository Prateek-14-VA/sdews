"""
SDEWS entry point.
Runs the Flask development server. Run with:
    python run.py
"""
from app import create_app
app=create_app("development")

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000, debug=True)