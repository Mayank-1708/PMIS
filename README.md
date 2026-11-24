# AI-Based Recommendation and Allocation System for PMIS

An intelligent web-based internship recommendation and allocation platform designed for the **Prime Minister Internship Scheme (PMIS)**.

The platform aims to simplify the internship application process by connecting internship seekers, companies and administrators through a centralized system.

It provides dedicated dashboards and functionalities for different users, helping streamline internship discovery, candidate management and allocation activities.

## Project Overview

The Prime Minister Internship Scheme aims to provide internship opportunities to young individuals and improve their industry exposure.

This project focuses on developing a centralized platform that facilitates interaction between:

- **Students / Internship Seekers:** Explore opportunities, manage profiles and track internship applications.
- **Companies:** Manage internship-related activities and view candidate profiles.
- **Administrators:** Monitor platform activities and manage internship allocation operations.

## Key Features

### 1. Student / Seeker Module
- Student registration and login.
- Dedicated student dashboard.
- Profile management.
- Internship-related information and application tracking.
- View internship application status.

### 2. Company Module
- Company registration and login.
- Dedicated company dashboard.
- View candidate profiles.
- Manage candidate-related activities.

### 3. Admin Module
- Dedicated administrator dashboard.
- View and manage internship seekers.
- Access candidate profiles.
- View and manage internship matches.
- Monitor platform activities.

## Technology Stack

| Category | Technologies |
|---|---|
| Backend | Python, Flask |
| Frontend | HTML, CSS, JavaScript |
| Database | SQLite |
| Development Environment | Visual Studio Code |

## Project Structure

```text
PMIS-Smart-Allocation/
│
├── static/
│   ├── script.js
│   └── style.css
│
├── templates/
│   ├── admin_dashboard.html
│   ├── admin_matches.html
│   ├── admin_view_profile.html
│   ├── admin_view_seeker.html
│   ├── company_dashboard.html
│   ├── company_view_candidate.html
│   ├── index.html
│   ├── login.html
│   ├── seeker_dashboard.html
│   └── signup.html
│
├── app.py
├── db.py
├── database.db
├── requirements.txt
└── README.md
```

## Installation and Setup

Follow these steps to run the project locally.

### Prerequisites

- Python 3.x
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/PMIS-Smart-Allocation.git
```

### Step 2: Navigate to the Project Directory

```bash
cd PMIS-Smart-Allocation
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
python app.py
```

Open the local URL displayed in your terminal to access the application.

*Note: The database and application configuration must be compatible with the included demo database.*

## Database

The project uses SQLite for database management.

The repository includes a demonstration database (`database.db`) containing synthetic data, provided it has been checked for sensitive information.

## Future Enhancements

- Enhanced internship recommendation capabilities.
- Improved candidate-company matching.
- Advanced analytics and reporting.
- Improved user experience and dashboard functionalities.
- Additional transparency in internship allocation.

## Project Objective

The primary objective is to develop a centralized and accessible internship management platform that simplifies the interaction between internship seekers, companies and administrators while supporting an organized internship allocation process.

---

**Project:** AI-Based Recommendation and Allocation System for Prime Minister Internship Scheme (PMIS)

**Repository:** PMIS-Smart-Allocation
