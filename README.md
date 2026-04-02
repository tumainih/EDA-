# AI-Powered Data Analysis Platform

This project provides a robust, zero-setup, **One-Click** interface for automated Data Cleaning, Exploratory Intelligence (EDA), Predictive Machine Learning Models, and Prescriptive Recommendations.

## Local Installation

1. Clone or download this directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the engine:
   ```bash
   streamlit run app.py
   ```

## Cloud / Docker Usage

For immediate portability across platforms like **AWS EC2**, **Heroku**, or **Google Cloud Run**, build and run the provided Docker image natively:

1. Build the container:
   ```bash
   docker build -t ai-data-analytics .
   ```
2. Run the container:
   ```bash
   docker run -p 8501:8501 ai-data-analytics
   ```
3. Access identically across operating systems by navigating to `http://localhost:8501` securely.
