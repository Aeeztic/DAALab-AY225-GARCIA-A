# CARVAJAL, Christian Ezekiel L. & GARCIA, Aahron Jamez

# Student Placement Analytics - Final Project

A full-stack analytics and prediction platform for student placement outcomes, featuring a FastAPI backend, React + Vite frontend, and deployment on Render.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Step-by-Step Setup](#step-by-step-setup)
- [Deployment (Render)](#deployment-render)
- [API Endpoints](#api-endpoints)
- [Frontend Usage](#frontend-usage)
- [Requirements](#requirements)
- [Authors](#authors)

---

## Project Overview
This project provides:
- Data ingestion and processing (CSV → Parquet)
- Uses `data/students.parquet` as the main analytics and prediction data source (not the CSV)
- Machine learning model for placement prediction
- REST API for analytics and predictions
- React frontend dashboard
- Cloud deployment via Render

---

## Features
- Upload and process large student datasets (up to 1M rows)
- Predict placement for single students or batch uploads
- Visualize statistics (placement rate, CGPA, internships, etc.)
- Filter, sort, and paginate student records
- The frontend never loads the entire dataset into the browser or as a JSON file; all data is accessed efficiently via paginated API calls.
- Deployed and accessible via Render

---

## Step-by-Step Setup

### 1. Clone the Repository
```
git clone <your-repo-url>
cd FINALS-LAB/project
```

### 2. Install Python Dependencies
Ensure you have Python 3.10+ and pip installed.
```
pip install -r requirements.txt
```

### 3. Prepare the Data
- Place your raw CSV file at `data/raw/students.csv` (used only for initial conversion).
- The backend and analytics use `data/students.parquet` as the main data file.
- To generate `students.parquet` from your CSV:
```
python convert_to_parquet.py
```
- (Optional) Verify row counts:
```
python check_data.py
```

### 4. Run the Backend API (Locally)
```
uvicorn src.api.main:app --host 0.0.0.0 --port 10000
```

### 5. Run the Frontend (Locally)
```
cd frontend
npm install
npm run dev
```

---

## Deployment (Render)

The backend and frontend are deployed on [Render](https://render.com/).

**Render Deployment Settings:**
- **Repository:** Connected to GitHub repo at `https://github.com/aezekiel/DAALab-ay225-GARCIA-A`
- **Branch:** Deploys from `main`
- **Root Directory:** Set to `FINALS-LAB/project`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `bash start.sh`
- **Auto-Deploy:** Enabled on every commit to the main branch
- **Region:** Singapore (Southeast Asia)
- **Instance Type:** Free (0.5 CPU, 512 MB RAM)

All dependencies are listed in `requirements.txt` and loaded automatically by Render.

Example Render endpoints (replace with your actual Render URLs):
  - Backend API: `https://your-backend-service.onrender.com`
  - Frontend: `https://your-frontend-service.onrender.com`

---

## API Endpoints

### Endpoints Used by the Frontend (from index.html)
- `GET /stats/overview` — Fetches dataset-level statistics
- `GET /stats/dataset` — Fetches full schema and analytics
- `GET /students` — Fetches paginated/filterable student records
- `POST /predict` — Runs placement prediction for a single student

### Additional Backend Endpoints
- `POST /batch-predict` — Batch prediction from uploaded CSV file (not directly used in frontend)
- `GET /health` — Health check endpoint

---

## Frontend Usage
- Built with React + Vite (see `frontend/`)
- Main dashboard: statistics, charts, and student table
- Connects to backend API endpoints (see above)
- To build for production:
```
npm run build
```

---

## Requirements

All backend dependencies are listed in `requirements.txt`:

```
alembic==1.18.4
annotated-doc==0.0.4
annotated-types==0.7.0
anyio==4.13.0
beautifulsoup4==4.14.3
bs4==0.0.2
certifi==2026.2.25
charset-normalizer==3.4.7
click==8.3.2
cloudpickle==3.1.2
colorama==0.4.6
colorlog==6.10.1
contourpy==1.3.3
cycler==0.12.1
fastapi==0.136.0
fonttools==4.62.1
greenlet==3.4.0
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
idna==3.12
joblib==1.5.3
kiwisolver==1.5.0
llvmlite==0.47.0
Mako==1.3.11
markdown-it-py==4.0.0
MarkupSafe==3.0.3
matplotlib==3.10.0
mdurl==0.1.2
networkx==3.6.1
numba==0.65.0
numpy==1.24.4
optuna==4.8.0
packaging==26.1
pandas==2.2.2
parse_pip_search==0.0.2
Pillow==12.2.0
polars==1.40.0
polars-runtime-32==1.40.0
pydantic==2.13.3
pydantic_core==2.46.3
Pygments==2.20.0
pypasing==3.3.2
python-dateutil==2.9.0.post0
python-multipart==0.0.26
PyYAML==6.0.3
requests==2.33.1
rich==15.0.0
scikit-learn==1.8.0
scipy==1.17.1
shap==0.51.0
six==1.17.0
slicer==0.0.8
soupsieve==2.8.3
SQLAlchemy==2.0.49
starlette==1.0.0
threadpoolctl==3.6.0
tqdm==4.67.3
typing-inspection==0.4.2
typing_extensions==4.15.0
tzdata==2026.1
urllib3==2.6.3
uvicorn==0.45.0
xgboost==3.3.3
```

Frontend dependencies are managed via npm (see `frontend/package.json`).

---

## Authors

---

## Contribution Breakdown & Implementation Details

### Student 1 · Repo Owner (GARCIA, Aahron Jamez)
**Data Engine & API Integration**

- Created the GitHub repository and managed all version control.
- Implemented the dataset loading logic and API integration to fetch data from the backend, ensuring `window.DS` is populated dynamically from the API (not from static files).
- Built the dynamic HTML table rendering logic for student records, supporting pagination and efficient updates.
- Developed filtering and sorting features for the table, allowing users to filter by name and sort by any column.
- Added a reset function to restore the original table view after filtering or sorting.
- Designed and implemented the summary cards (mean score, top scorer, pass rate, average study hours) using live data from the backend.

### Student 2 · Collaborator (CARVAJAL, Christian Ezekiel L.)
**Model Inference & Prediction**

- Implemented the bar chart visualization for the top 10 scores using Chart.js, dynamically updating from API data.
- Built the scatter plot for attendance vs. score, leveraging real-time data from the backend.
- Developed the doughnut chart for grade distribution, ensuring accurate representation of the dataset.
- Coded statistical functions for standard deviation, variance, Pearson correlation, and linear regression, all computed on-the-fly from API responses.
- Implemented the rendering of analysis tables, populating all statistical summaries and breakdowns.
- Authored the narrative insights section, providing written findings and interpretations based on the analyzed data.

---

## Notes
- For any issues, check logs on Render or run locally for debugging.
- Ensure your data files are correctly placed and formatted.
- All endpoints and features are documented above for easy reference.
- A cron job is set up to call the API every 5 minutes to prevent the Render service from sleeping.


