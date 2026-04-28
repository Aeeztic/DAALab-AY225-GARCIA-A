# CARVAJAL, Christian Ezekiel L. & GARCIA, Aahron Jamez
# Student Placement Analytics - Final Project

**🔴 Live Application (GitHub Pages):** [https://aeeztic.github.io/DAALab-AY225-GARCIA-A/](https://aeeztic.github.io/DAALab-AY225-GARCIA-A/)

A full-stack analytics and prediction platform for student placement outcomes, featuring a highly-optimized FastAPI backend (with DuckDB), a dynamic React + Vite frontend, and continuous deployment on Render.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture & Technical Features](#architecture--technical-features)
3. [Dashboard Guide & Expected Inputs](#dashboard-guide--expected-inputs)
    - [Global Server-Side Search Guide](#global-server-side-search-guide)
    - [Placement Prediction Engine Guide](#placement-prediction-engine-guide)
    - [Visualizations & Analytics Output](#visualizations--analytics-output)
4. [Step-by-Step Setup](#step-by-step-setup)
5. [Deployment (Render)](#deployment-render)
6. [API Endpoints](#api-endpoints)
7. [Project File Structure](#project-file-structure)
8. [Contribution Breakdown & Implementation Details](#contribution-breakdown--implementation-details)

---

## Project Overview

This project provides a robust solution for tracking, analyzing, and predicting student placement outcomes in university settings. The system processes a massive dataset of 1,000,000 students, transforming raw CSV data into a highly compressed Parquet format (`data/students.parquet`). 

By leveraging DuckDB, the backend can query this million-row dataset entirely from disk with a RAM footprint well under 100MB, preventing out-of-memory crashes on resource-constrained deployment environments.

---

## Architecture & Technical Features

- **Massive Scale Ingestion:** Capable of processing up to 1,000,000 student records instantly.
- **Memory-Efficient Data Engine:** Uses DuckDB to query `students.parquet` dynamically. The frontend *never* loads the entire dataset into the browser, relying purely on paginated REST API calls.
- **Machine Learning Integration:** Uses an XGBoost predictive model for single-student and batch predictions.
- **Dynamic Frontend:** Built with vanilla JS patterns on Vite to ensure minimal bundle overhead, utilizing Chart.js for data visualization.
- **Cloud Continuous Deployment:** Deployed securely on Render via continuous branch integration.

---

## Dashboard Guide & Expected Inputs

To fully test the application's capabilities, please review the following boundaries and UI behaviors. These are mapped directly to the actual values present in our 1,000,000-row dataset.

### Global Server-Side Search Guide

The dashboard includes a powerful, server-side filtering mechanism to sort through the database.
- **True Server-Side Search:** The "Search Database" field does *not* just search the visible page. It generates a query parameter (`?search=...`) that uses a SQL `ILIKE` condition on the backend, accurately searching the entire 1,000,000-row DuckDB file.
- **Expected Search Inputs:** 
  - **Branches:** Type specific branch acronyms: `CSE`, `IT`, `ECE`, `CE`, `EE`, or `ME`.
  - **Gender:** Type `Male` or `Female`.
- **Explicit Submit UI:** To prevent the application from spamming the server and creating a jumpy user experience, the filters (Search Database, Min CGPA, Min Internships) operate on an **Explicit Submit** model. 
  - Typing in the input fields will *not* trigger a reload.
  - You must either press the **"Enter"** key or click the primary **"Apply Filters"** button located next to the "Reset View" button. 
  - Dropdown options (like Page Size and Sort View) will still immediately reload the table.

### Placement Prediction Engine Guide

The Predictive Engine form (located on the right side of the dashboard or accessible by clicking any table row) uses an XGBoost ML model to calculate a student's probability of being placed. 

**Allowed Input Bounds (Based on actual dataset minimums/maximums):**
- **CGPA (Required):** Must be a decimal between `0.0` and `10.0` (The dataset specifically ranges from `5.0` to `10.0`).
- **Internships (Required):** Must be an integer `0` or greater (Dataset goes up to `11`).
- **Branch (Required):** Selectable dropdown mapping to `CSE`, `IT`, `ECE`, `CE`, `EE`, or `ME`.
- **Gender (Required):** Selectable dropdown for `Male` or `Female`.
- **Communication Skill:** Integer on a scale of `0` to `9`.
- **Problem Solving:** Integer on a scale of `0` to `9`.
- **Age (Optional):** Integer (defaults to `21`). Dataset ranges from `20` to `25`.
- **Projects (Optional):** Integer (defaults to `2`). Dataset ranges from `0` to `15`.

*Note: Clicking any row on the paginated data table will automatically grab that student's actual values, fill out the Prediction form, and scroll you down to run an instant prediction!*

### Visualizations & Analytics Output

After running a prediction or loading the main dashboard, the following analytics are generated:
- **SHAP Value Waterfall Chart:** Upon successfully running a prediction, the dashboard renders a SHapley Additive exPlanations (SHAP) chart. This crucial visualization mathematically breaks down *why* the model made its decision. Green bars demonstrate specific traits (like a high CGPA) that pushed the placement probability higher, while red bars demonstrate traits (like low internships) that hurt their chances.
- **Top 10 Aptitude Scores:** A dynamic horizontal bar chart highlighting the highest raw aptitude scores within the current view.
- **Attendance vs. Score Scatter Plot:** Evaluates the correlation between a student's attendance percentage and their test scores.
- **Placement Distribution Doughnut:** A concise chart representing the overall percentage of students marked as 'Placed' vs 'Not Placed'.

---

## Step-by-Step Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Aeeztic/DAALab-AY225-GARCIA-A.git
cd FINALS-LAB/project
```

### 2. Install Python Dependencies
Ensure you have Python 3.10+ and pip installed.
```bash
pip install -r requirements.txt
```

### 3. Prepare the Data
- Place your raw CSV file at `data/raw/students.csv` (used only for initial conversion).
- The backend and analytics use `data/students.parquet` as the main data file.
- To generate `students.parquet` from your CSV:
```bash
python scripts/convert_to_parquet.py
```
*(Optional)* Verify row counts and database integrity:
```bash
python scripts/check_data.py
```

### 4. Run the Backend API (Locally)
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 10000
```

### 5. Run the Frontend (Locally)
```bash
npm install
npm run dev
```

---

## Deployment (Render)

The backend and frontend are deployed entirely on the [Render](https://render.com/) cloud ecosystem.

**Render Deployment Settings:**
- **Repository:** Connected to GitHub repo at `https://github.com/Aeeztic/DAALab-AY225-GARCIA-A/tree/main/FINALS-LAB/project`
- **Branch:** Deploys from `main`
- **Root Directory:** Set to `FINALS-LAB/project`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `bash start.sh` (Starts Uvicorn Server)
- **Region:** Singapore (Southeast Asia)
- **Instance Type:** Free (0.5 CPU, 512 MB RAM limit — Memory crashes are avoided via our DuckDB architecture).

All dependencies are loaded automatically. A CRON job is implemented to call the API every 5 minutes to prevent the Render instance from spinning down.

---

## API Endpoints

### Core Dashboard Endpoints
- `GET /stats/overview` — Fetches global dataset-level metrics.
- `GET /stats/dataset` — Fetches full schema constraints and dynamic field ranges.
- `GET /students` — Core endpoint handling server-side pagination, sorting, and ILIKE database searching.
- `POST /predict` — Takes a JSON payload of a student's stats and returns XGBoost placement probability.

### System Endpoints
- `POST /batch-predict` — Batch prediction from an uploaded `.csv` file.
- `GET /health` — Instance health check.

---

## Project File Structure

```text
/ (Root Repository Directory)
+-- index.html                 <-- GitHub Pages Live Entrypoint (Self-Contained)
+-- README.md
+-- FINALS-LAB/
    +-- project/               <-- Backend & Original Frontend Source
        +-- package.json
        +-- package-lock.json
        +-- check_data.py
        +-- convert_to_parquet.py
        +-- requirements.txt
        +-- start.sh
        +-- data/
        |   +-- processed/  <--- ML
        |   |   +-- feature_metadata.json
        |   |   +-- test.parquet
        |   |   +-- train.parquet
        |   |   +-- val.parquet
        |   +-- raw/
        |   |   +-- students.csv
        |   +-- students.parquet
        +-- frontend/
        |   +-- src/           <-- React Components
        |   +-- public/
        |   +-- dist/
        |   +-- vite.config.js
        +-- src/
            +-- api/           <-- FastAPI Routes
            +-- config/
            +-- data/
            +-- ml/            <-- XGBoost Models & SHAP logic
```

---

## Contribution Breakdown & Implementation Details

### Student 1 · Repo Owner (GARCIA, Aahron Jamez)
**Data Engine, API Integration, & UI Enhancements**
- Created the GitHub repository and managed all version control.
- Refactored the data ingestion logic from memory-heavy Pandas eagerly loading to a RAM-efficient **DuckDB** integration directly querying Parquet files.
- Integrated the FastAPI backend with the frontend table, ensuring `window.DS` populated via paginated chunks instead of static local files.
- Built the complex frontend state and "Explicit Submit" filters (Min CGPA, Min Internships, Database ILIKE Search), wiring up debounce removal and keydown event logic.
- Implemented responsive, modern styling across the dashboard (branding to *Design, Analysis, & Algorithm*).

### Student 2 · Collaborator (CARVAJAL, Chrstian Ezekiel L.)
**Model Inference, Statistical Analysis, & Visualizations**
- Handled the machine learning normalization and validation pipelines via Pydantic mapping `src/ml/predict.py`.
- Developed the Chart.js visual engine for the dashboard, implementing the bar chart (top 10 scores), scatter plot (attendance vs. score), and the doughnut chart (grade distribution).
- Developed the highly complex SHAP value waterfall chart interpretation for the predictive form, dynamically creating visual explainability metrics (green/red bars) for every inference.
- Coded statistical mathematical formulas (Standard deviation, Pearson correlation, Linear regression) calculated directly on the API payloads.
- Authored narrative insight generators that automatically explain trends to the end user.