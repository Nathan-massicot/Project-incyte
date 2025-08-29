📑 Project Incyte – Dossier Optimization (DPO)

🔍 Overview

Project Incyte (DPO – Dossier Optimization) is a regulatory data-driven application designed to optimize the preparation, tracking, and visualization of Marketing Authorization Application (MAA) and other regulatory submissions.
The project combines structured CSV datasets, automated pipelines, and interactive dashboards to streamline dossier planning and workload management across functions.

The tool is built with Python + Streamlit, using pandas, plotly, and custom CSV mappings to provide:
	•	📊 Regulatory Gantt charts (submission roadmap, critical path, milestones)
	•	👥 Role mapping across Regulatory Operations, CMC, Strategy, EMA, Management, and others
	•	📂 Modular task aggregation (Module 1–5, clinical, non-clinical, CMC, labeling, etc.)
	•	⚙️ Automation scripts for CSV cleaning, re-indexing, and role attribution
	•	✅ FTE workload & milestone tracking with export capabilities

🖥️ Features

1. Regulatory Timeline Dashboard
	•	Gantt chart visualization of tasks, grouped by roles, modules, or functions
	•	Fixed color scheme per role:
	•	Reg Ops = 🔴 Red
	•	Management = 🔵 Light Blue
	•	Other function = 🟣 Violet
	•	CMC = 🟠 Orange
	•	Lead = 🟢 Green
	•	Strategist = 🟤 Brown
	•	EMA = 🟡 Yellow
	•	Labeling = 🔵 Dark Blue

2. CSV Automation
	•	Reindexing Task_ID and Step_No consistently
	•	Role assignment from Role_to_task_jalons.csv (multi-role support)
	•	Task cleaning (removal of ISS/ISE, integrated summaries, etc.)
	•	Dispatch day correction (start dates vs. finish dates)

3. Module Aggregation
	•	Mapping of tasks to CTD modules (1–5)
	•	Differentiation inside Module 2:
	•	CMC
	•	Clinical
	•	Non-clinical
	•	Other tasks flagged as “Other”

4. FTE & Milestone Tracking
	•	Compute workload (effort days × allocated FTE)
	•	Track critical milestones (e.g., Management Decision = 1, others = 0)
	•	Export reports in CSV or Excel



Installation
	1.	Clone the repo:git clone https://github.com/<your_repo>/project-incyte-dpo.git
cd project-incyte-dpo
	2.	Create a virtual environment and install dependencies:python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
3. Run the Streamlit app:streamlit run POMv4.py

You will see an interactive dashboard with:
	•	Gantt chart of regulatory tasks
	•	Filters by module, role, or function
	•	Milestones & workload analysis




👥 Roles & Responsibilities
	•	Regulatory Operations (Reg Ops) → eCTD compilation, technical validation, submissions
	•	Global Regulatory Lead (GRL) → Overall submission strategy
	•	Regulatory Strategist (Global/EU) → Timeline definition, EMA/CHMP interactions
	•	Labeling → Core Data Sheet, SmPC, labeling documents
	•	Management → Go/no-go decisions
	•	Other function → Cross-functional contributors (stats, medical writing, QA, etc.)
	•	Regulatory CMC → Module 3, quality sections
	•	EMA → External authority interactions

⸻

🗂️ Data Sources
	•	Internal task mappings (DayDataMaav5.csv)
	•	Role mapping (Role_to_task_jalons.csv)
	•	Task → CTD module mapping (Task-Module_mapping__refined.csv)
	•	EMA/FDA submission process guidance (public regulatory guidelines)




