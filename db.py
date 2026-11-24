import os
import sqlite3

DB_NAME = os.path.join(os.path.dirname(__file__), "database.db")


def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def _column_exists(cursor, table_name, column_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    return any(row[1] == column_name for row in cursor.fetchall())


def _ensure_column(cursor, table_name, column_name, definition):
    if not _column_exists(cursor, table_name, column_name):
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


def create_tables():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS seeker_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE,
            name TEXT,
            phone TEXT,
            age INTEGER,
            marks10 REAL,
            marks12 REAL,
            gpa REAL,
            qualification TEXT,
            degree TEXT,
            branch TEXT,
            college_name TEXT,
            college_type TEXT,
            skills TEXT,
            experience_level TEXT,
            sector_interest TEXT,
            preferred_mode TEXT,
            preferred_location TEXT,
            caste_category TEXT,
            rural_urban TEXT,
            aspirational_district TEXT,
            lor_available TEXT,
            first_time_intern TEXT,
            pwd TEXT,
            income_bracket TEXT,
            employed_fulltime TEXT,
            student_fulltime TEXT,
            distance_online_student TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS internships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_user_id INTEGER,
            company_name TEXT,
            role_title TEXT,
            sector TEXT,
            required_skills TEXT,
            seats INTEGER,
            stipend TEXT,
            mode TEXT,
            location TEXT,
            min_gpa REAL,
            status TEXT DEFAULT 'active',
            FOREIGN KEY(company_user_id) REFERENCES users(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            seeker_user_id INTEGER,
            internship_id INTEGER,
            match_score REAL,
            match_reason TEXT,
            status TEXT DEFAULT 'pending',
            interview_status TEXT DEFAULT 'not_scheduled',
            interview_date TEXT,
            company_verification TEXT DEFAULT 'pending',
            company_feedback TEXT,
            admin_decision_note TEXT,
            allocation_batch TEXT,
            created_at TEXT,
            updated_at TEXT,
            verified_at TEXT,
            offer_sent_at TEXT,
            FOREIGN KEY(seeker_user_id) REFERENCES users(id),
            FOREIGN KEY(internship_id) REFERENCES internships(id)
        )
        """
    )

    _ensure_column(cur, "applications", "interview_status", "TEXT DEFAULT 'not_scheduled'")
    _ensure_column(cur, "applications", "interview_date", "TEXT")
    _ensure_column(cur, "applications", "company_verification", "TEXT DEFAULT 'pending'")
    _ensure_column(cur, "applications", "company_feedback", "TEXT")
    _ensure_column(cur, "applications", "admin_decision_note", "TEXT")
    _ensure_column(cur, "applications", "allocation_batch", "TEXT")
    _ensure_column(cur, "applications", "created_at", "TEXT")
    _ensure_column(cur, "applications", "updated_at", "TEXT")
    _ensure_column(cur, "applications", "verified_at", "TEXT")
    _ensure_column(cur, "applications", "offer_sent_at", "TEXT")
    _ensure_column(cur, "applications", "skill_similarity", "REAL DEFAULT 0")
    _ensure_column(cur, "applications", "semantic_similarity", "REAL DEFAULT 0")
    _ensure_column(cur, "applications", "success_probability", "REAL DEFAULT 0")
    _ensure_column(cur, "applications", "offer_acceptance_probability", "REAL DEFAULT 0")
    _ensure_column(cur, "applications", "opportunity_gap_score", "REAL DEFAULT 0")
    _ensure_column(cur, "applications", "policy_priority_score", "REAL DEFAULT 0")
    _ensure_column(cur, "applications", "ai_rank_score", "REAL DEFAULT 0")
    _ensure_column(cur, "applications", "explanation", "TEXT")
    _ensure_column(cur, "applications", "score_breakdown", "TEXT")
    _ensure_column(cur, "applications", "student_preference_rank", "INTEGER DEFAULT 1")

    conn.commit()
    conn.close()


def initialize_database():
    create_tables()


if __name__ == "__main__":
    initialize_database()
    print("Database created successfully.")
