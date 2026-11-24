import csv
import json
import pickle
import re
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, flash, redirect, render_template, request, send_from_directory, session, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.security import check_password_hash, generate_password_hash

from db import get_connection, initialize_database
from research_engine.feature_engineering import add_rule_predictions
from research_engine.policy_allocator import policy_priority_score
from research_engine.explainability import local_explanation
from research_engine.stable_matching import modified_gale_shapley
from research_engine.ml_models import predict_probability

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency for local setup
    SentenceTransformer = None

SKILL_SYNONYMS = {
    "ai": "artificial intelligence",
    "artificial-intelligence": "artificial intelligence",
    "ml": "machine learning",
    "machine learning engineer": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "js": "javascript",
    "reactjs": "react",
    "react js": "react",
    "nodejs": "node.js",
    "node js": "node.js",
    "expressjs": "express",
    "py": "python",
    "sql server": "sql",
    "mysql": "sql",
    "postgresql": "sql",
    "postgres": "sql",
    "excel analytics": "excel",
    "ms excel": "excel",
    "powerbi": "power bi",
    "tableau dashboards": "tableau",
    "data analytics": "data analysis",
    "data analyst": "data analysis",
    "oop": "oops",
    "object oriented programming": "oops",
    "c plus plus": "c++",
    "communication skills": "communication",
    "leadership skills": "leadership",
}

SKILL_ONTOLOGY = {
    "python": {"programming", "data analysis", "automation"},
    "java": {"programming", "backend"},
    "javascript": {"programming", "frontend"},
    "react": {"frontend", "web development"},
    "node.js": {"backend", "web development"},
    "sql": {"databases", "data analysis"},
    "excel": {"data analysis", "reporting"},
    "power bi": {"dashboarding", "data visualization"},
    "tableau": {"dashboarding", "data visualization"},
    "machine learning": {"artificial intelligence", "data science"},
    "deep learning": {"machine learning", "artificial intelligence"},
    "natural language processing": {"machine learning", "artificial intelligence"},
    "computer vision": {"machine learning", "artificial intelligence"},
    "pandas": {"python", "data analysis"},
    "numpy": {"python", "data analysis"},
    "tensorflow": {"deep learning", "machine learning"},
    "pytorch": {"deep learning", "machine learning"},
    "hadoop": {"big data", "data engineering"},
    "spark": {"big data", "data engineering"},
    "communication": {"soft skills"},
    "leadership": {"soft skills"},
    "problem solving": {"soft skills"},
}

STATUS_META = {
    "pending": {"label": "Pending Admin Review", "class": "pending"},
    "allocated": {"label": "Allocated by Admin", "class": "live"},
    "interview_scheduled": {"label": "Interview Scheduled", "class": "warn"},
    "verified": {"label": "Verified by Company", "class": "done"},
    "successful": {"label": "Final Offer Sent", "class": "success"},
    "accepted": {"label": "Offer Accepted", "class": "accepted"},
    "rejected": {"label": "Rejected / Closed", "class": "danger"},
}

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_embedding_model = None
PROJECT_ROOT = Path(__file__).resolve().parent
RESEARCH_OUTPUTS = PROJECT_ROOT / "research_outputs"
RESEARCH_PIPELINE = RESEARCH_OUTPUTS / "pipeline"
RESEARCH_CHARTS = RESEARCH_OUTPUTS / "charts"
COMPLETION_MODEL_PATH = RESEARCH_PIPELINE / "models" / "best_completion_model.pkl"
SUCCESS_MODEL_PATH = RESEARCH_PIPELINE / "models" / "best_internship_success_model.pkl"
OPPORTUNITY_GAP_MODEL_PATH = RESEARCH_PIPELINE / "models" / "best_opportunity_gap_model.pkl"
OFFER_ACCEPTANCE_MODEL_PATH = RESEARCH_PIPELINE / "models" / "best_offer_acceptance_model.pkl"
_completion_model_payload = None
_prediction_model_payloads = {}


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_skill_token(token):
    cleaned = re.sub(r"[^a-zA-Z0-9#+. ]+", " ", (token or "").strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return SKILL_SYNONYMS.get(cleaned, cleaned)


def extract_skill_terms(skill_text):
    if not skill_text:
        return []
    tokens = re.split(r"[,/\n;|]+", skill_text)
    normalized = []
    for token in tokens:
        skill = normalize_skill_token(token)
        if skill:
            normalized.append(skill)
    return normalized


def expand_skill_set(skill_text):
    expanded = set()
    for skill in extract_skill_terms(skill_text):
        expanded.add(skill)
        for related in SKILL_ONTOLOGY.get(skill, set()):
            expanded.add(related)
    return expanded


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None and SentenceTransformer is not None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_completion_model_payload():
    global _completion_model_payload
    if _completion_model_payload is None and COMPLETION_MODEL_PATH.exists():
        try:
            with open(COMPLETION_MODEL_PATH, "rb") as handle:
                _completion_model_payload = pickle.load(handle)
        except Exception:
            _completion_model_payload = False
    return _completion_model_payload if _completion_model_payload is not False else None


def load_prediction_model(path):
    key = str(path)
    if key not in _prediction_model_payloads:
        if not path.exists():
            _prediction_model_payloads[key] = None
        else:
            try:
                with open(path, "rb") as handle:
                    _prediction_model_payloads[key] = pickle.load(handle)
            except Exception:
                _prediction_model_payloads[key] = None
    return _prediction_model_payloads[key]


def embedding_similarity(candidate_text, internship_text):
    model = get_embedding_model()
    if model is None:
        return None
    embeddings = model.encode([candidate_text, internship_text], normalize_embeddings=True)
    return float(embeddings[0] @ embeddings[1])


def skill_similarity_tfidf(seeker_skills, required_skills):
    seeker_set = expand_skill_set(seeker_skills)
    required_set = expand_skill_set(required_skills)
    if not seeker_set or not required_set:
        return 0.0, []

    seeker_text = " ".join(sorted(seeker_set))
    required_text = " ".join(sorted(required_set))
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([seeker_text, required_text])
    semantic_sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])

    matched = sorted(seeker_set & required_set)
    overlap_sim = len(matched) / max(len(required_set), 1)
    combined = (0.7 * semantic_sim) + (0.3 * overlap_sim)
    return min(combined, 1.0), matched


def income_to_group(income_bracket):
    value = (income_bracket or "").lower()
    if "below" in value or "1l" in value or "1l-3l" in value:
        return "Low"
    if "3l-5l" in value:
        return "Middle"
    return "High"


def build_research_feature_row(seeker, internship, tfidf_similarity, semantic_similarity, matched_skills, student_preference_rank=1):
    mode_fit = 1.0 if (seeker["preferred_mode"] or "").lower() == (internship["mode"] or "").lower() else 0.45
    loc_score, _ = compute_location_score(seeker["preferred_location"], internship["location"])
    location_fit = loc_score / 8
    sector_fit = 1.0 if (seeker["sector_interest"] or "").lower() == (internship["sector"] or "").lower() else 0.45
    required_count = max(len(expand_skill_set(internship["required_skills"])), 1)
    skill_overlap = len(matched_skills) / required_count
    first_time = (seeker["first_time_intern"] or "").lower() == "yes"

    return {
        "student_id": str(seeker["user_id"]),
        "internship_id": str(internship["id"]),
        "skill_similarity": round(max(tfidf_similarity, skill_overlap), 6),
        "semantic_similarity": round(semantic_similarity, 6),
        "location_fit": round(location_fit, 6),
        "mode_fit": round(mode_fit, 6),
        "sector_fit": round(sector_fit, 6),
        "stipend_fit": 0.70,
        "student_preference_rank": max(int(student_preference_rank or 1), 1),
        "cgpa": safe_float(seeker["gpa"], 0),
        "project_count": 1 if (seeker["experience_level"] or "").lower() in {"beginner", "intermediate"} else 0,
        "communication_score": 0.70 if (seeker["lor_available"] or "").lower() == "yes" else 0.62,
        "rural_urban": seeker["rural_urban"] or "",
        "first_generation": "Yes" if (seeker["college_type"] or "").lower() in {"government college", "state university"} else "No",
        "previous_internship": "No" if first_time else "Yes",
        "income_group": income_to_group(seeker["income_bracket"]),
        "aspirational_district": seeker["aspirational_district"] or "",
        "candidate_category": seeker["caste_category"] or "",
    }


def compute_research_match(seeker, internship, student_preference_rank=1):
    tfidf_similarity, matched_skills = skill_similarity_tfidf(seeker["skills"], internship["required_skills"])
    candidate_profile_text = " ".join(
        filter(
            None,
            [
                seeker["skills"],
                seeker["branch"],
                seeker["sector_interest"],
                seeker["experience_level"],
                seeker["preferred_location"],
                seeker["preferred_mode"],
            ],
        )
    )
    internship_profile_text = " ".join(
        filter(
            None,
            [
                internship["role_title"],
                internship["required_skills"],
                internship["sector"],
                internship["location"],
                internship["mode"],
            ],
        )
    )
    semantic_similarity = embedding_similarity(candidate_profile_text, internship_profile_text)
    semantic_similarity = tfidf_similarity if semantic_similarity is None else semantic_similarity

    feature_row = build_research_feature_row(
        seeker,
        internship,
        tfidf_similarity,
        semantic_similarity,
        matched_skills,
        student_preference_rank,
    )
    enriched = add_rule_predictions(feature_row)
    model_payload = load_prediction_model(SUCCESS_MODEL_PATH) or get_completion_model_payload()
    if model_payload:
        try:
            enriched["ml_completion_probability"] = round(predict_probability(model_payload, enriched), 6)
            enriched["selected_model"] = model_payload.get("model_name", "Saved ML model")
        except Exception:
            enriched["ml_completion_probability"] = enriched["success_probability_rule"]
            enriched["selected_model"] = "Rule fallback"
    else:
        enriched["ml_completion_probability"] = enriched["success_probability_rule"]
        enriched["selected_model"] = "Rule fallback"
    opportunity_model = load_prediction_model(OPPORTUNITY_GAP_MODEL_PATH)
    if opportunity_model:
        try:
            enriched["opportunity_gap_score"] = round(predict_probability(opportunity_model, enriched), 6)
            enriched["opportunity_gap_model"] = opportunity_model.get("model_name", "Saved ML model")
        except Exception:
            enriched["opportunity_gap_model"] = "Rule fallback"
    else:
        enriched["opportunity_gap_model"] = "Rule fallback"
    acceptance_model = load_prediction_model(OFFER_ACCEPTANCE_MODEL_PATH)
    if acceptance_model:
        try:
            enriched["offer_acceptance_probability_rule"] = round(predict_probability(acceptance_model, enriched), 6)
            enriched["offer_acceptance_model"] = acceptance_model.get("model_name", "Saved ML model")
        except Exception:
            enriched["offer_acceptance_model"] = "Rule fallback"
    else:
        enriched["offer_acceptance_model"] = "Rule fallback"
    enriched["policy_priority_score"] = policy_priority_score(enriched)
    enriched["ai_rank_score"] = round(
        (
            0.34 * enriched["ml_completion_probability"]
            + 0.22 * enriched["offer_acceptance_probability_rule"]
            + 0.18 * enriched["semantic_similarity"]
            + 0.14 * enriched["skill_similarity"]
            + 0.12 * enriched["policy_priority_score"]
        ),
        6,
    )
    enriched["matched_skills"] = matched_skills
    enriched["explanations"] = local_explanation({
        **enriched,
        "success_probability": enriched["ml_completion_probability"],
        "offer_acceptance_probability": enriched["offer_acceptance_probability_rule"],
    })
    return enriched


def status_badge(status):
    return STATUS_META.get(status, {"label": status.replace("_", " ").title(), "class": "pending"})


def read_csv_rows(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_csv_one(path):
    rows = read_csv_rows(path)
    return rows[0] if rows else {}


def load_research_snapshot():
    return {
        "summary": read_csv_one(RESEARCH_PIPELINE / "summary.csv"),
        "model_comparison": read_csv_rows(RESEARCH_PIPELINE / "model_comparison.csv"),
        "allocation_comparison": read_csv_rows(RESEARCH_PIPELINE / "allocation_comparison.csv"),
        "quota_report": read_csv_one(RESEARCH_PIPELINE / "policy_quota_report.csv"),
        "feature_importance": read_csv_rows(RESEARCH_PIPELINE / "global_feature_importance.csv")[:8],
        "multi_size": read_csv_rows(RESEARCH_OUTPUTS / "multi_size_experiment_summary.csv"),
        "charts_available": RESEARCH_CHARTS.exists(),
    }


def candidate_is_eligible(seeker):
    reasons = []
    age = safe_float(seeker["age"], 0)
    if age < 21 or age > 24:
        reasons.append("Age is outside the PM Internship Scheme range.")
    if (seeker["employed_fulltime"] or "").lower() == "yes":
        reasons.append("Full-time employed candidates are not eligible.")
    return len(reasons) == 0, reasons


def compute_location_score(candidate_location, internship_location):
    candidate = (candidate_location or "").strip().lower()
    internship = (internship_location or "").strip().lower()
    if not candidate or not internship:
        return 3, "Location preference not available, so a neutral score is used."
    if candidate == internship:
        return 8, "Preferred location matched exactly."
    candidate_parts = {part.strip() for part in re.split(r"[,/-]+", candidate) if part.strip()}
    internship_parts = {part.strip() for part in re.split(r"[,/-]+", internship) if part.strip()}
    if candidate_parts & internship_parts:
        return 6, "Preferred location partially matched at city/state level."
    return 3, "Location does not match exactly, but candidate remains eligible."


def compute_match_score(seeker, internship, student_preference_rank=1):
    score = 0.0
    reasons = []
    breakdown = {}

    research = compute_research_match(seeker, internship, student_preference_rank)
    matched_skills = research["matched_skills"]
    overlap_bonus = len(matched_skills) / max(len(expand_skill_set(internship["required_skills"])), 1)
    similarity = min(
        (0.25 * research["skill_similarity"])
        + (0.35 * research["semantic_similarity"])
        + (0.20 * research["ml_completion_probability"])
        + (0.10 * research["offer_acceptance_probability_rule"])
        + (0.10 * research["policy_priority_score"]),
        1.0,
    )
    skill_score = round(similarity * 40, 2)
    score += skill_score
    breakdown["Skill Fit"] = f"{skill_score}/40"
    breakdown["Skill Graph Similarity"] = f"{round(research['skill_similarity'] * 100, 1)}%"
    breakdown["Embedding Similarity"] = f"{round(research['semantic_similarity'] * 100, 1)}%"
    breakdown["Knowledge Overlap"] = f"{round(overlap_bonus * 100, 1)}%"
    breakdown["ML Success Probability"] = f"{round(research['ml_completion_probability'] * 100, 1)}%"
    breakdown["Offer Acceptance Probability"] = f"{round(research['offer_acceptance_probability_rule'] * 100, 1)}%"
    breakdown["Opportunity Gap Score"] = f"{round(research['opportunity_gap_score'] * 100, 1)}%"
    breakdown["Policy Priority"] = f"{round(research['policy_priority_score'] * 100, 1)}%"
    breakdown["Student Preference Rank"] = f"#{max(int(student_preference_rank or 1), 1)}"
    if matched_skills:
        reasons.append(f"Matched or related skills found: {', '.join(matched_skills[:6])}")
    else:
        reasons.append("Few direct skill overlaps, relying on academic and preference fit.")
    reasons.extend(research["explanations"])

    seeker_gpa = safe_float(seeker["gpa"], 0)
    min_gpa = safe_float(internship["min_gpa"], 0)
    gpa_score = 15 if seeker_gpa >= min_gpa else 5
    breakdown["GPA Fit"] = f"{gpa_score}/15"
    score += gpa_score
    reasons.append("Meets GPA requirement." if seeker_gpa >= min_gpa else "Below preferred GPA, but still considered.")

    avg_marks = (safe_float(seeker["marks10"], 0) + safe_float(seeker["marks12"], 0)) / 2
    if avg_marks >= 80:
        marks_score = 8
        reasons.append("Strong 10th and 12th academic performance.")
    elif avg_marks >= 60:
        marks_score = 6
        reasons.append("Solid school-level academic record.")
    else:
        marks_score = 3
        reasons.append("Basic academic eligibility retained.")
    breakdown["Academic Strength"] = f"{marks_score}/8"
    score += marks_score

    experience_level = (seeker["experience_level"] or "").lower()
    breakdown["Experience"] = "Captured for company review only"

    mode_match = (seeker["preferred_mode"] or "").lower() == (internship["mode"] or "").lower()
    mode_score = 8 if mode_match else 3
    breakdown["Mode Preference"] = f"{mode_score}/8"
    score += mode_score
    reasons.append("Preferred internship mode matched." if mode_match else "Mode preference is flexible for this role.")

    loc_score, location_reason = compute_location_score(seeker["preferred_location"], internship["location"])
    breakdown["Location Fit"] = f"{loc_score}/8"
    score += loc_score
    reasons.append(location_reason)

    sector_match = (seeker["sector_interest"] or "").lower() == (internship["sector"] or "").lower()
    sector_score = 8 if sector_match else 3
    breakdown["Sector Interest"] = f"{sector_score}/8"
    score += sector_score
    reasons.append("Sector interest aligned with the role." if sector_match else "Sector interest is nearby but not an exact match.")

    first_time_score = 6 if (seeker["first_time_intern"] or "").lower() == "yes" else 0
    lor_score = 1 if (seeker["lor_available"] or "").lower() == "yes" else 0
    rural_score = 2 if (seeker["rural_urban"] or "").lower() == "rural" else 0
    aspirational_score = 2 if (seeker["aspirational_district"] or "").lower() == "yes" else 0
    caste_score = 2 if (seeker["caste_category"] or "").upper() in {"SC", "ST", "OBC", "EWS"} else 0

    breakdown["First-Time Candidate"] = f"+{first_time_score}/6"
    breakdown["LOR Bonus"] = f"+{lor_score}/1"
    breakdown["Rural Inclusion"] = f"+{rural_score}/2"
    breakdown["Aspirational District"] = f"+{aspirational_score}/2"
    breakdown["Social Inclusion"] = f"+{caste_score}/2"
    score += first_time_score + lor_score + rural_score + aspirational_score + caste_score

    if first_time_score:
        reasons.append("First-time internship priority applied as per scheme preference.")
    if lor_score:
        reasons.append("LOR adds extra confidence to the application profile.")
    if rural_score:
        reasons.append("Rural representation bonus applied.")
    if aspirational_score:
        reasons.append("Aspirational district inclusion bonus applied.")
    if caste_score:
        reasons.append("Social category inclusion bonus applied.")

    final_score = min(round(score, 2), 100.0)
    breakdown["Final Score"] = f"{final_score}/100"
    return final_score, reasons, breakdown


def get_ranked_candidates(cur, internship_id):
    cur.execute("SELECT * FROM internships WHERE id=?", (internship_id,))
    internship = cur.fetchone()
    if not internship:
        return None, []

    cur.execute(
        """
        SELECT a.id AS application_id, a.status AS application_status, a.match_score AS stored_score,
               a.interview_status, a.company_verification, a.student_preference_rank, s.*
        FROM applications a
        JOIN seeker_profiles s ON a.seeker_user_id = s.user_id
        WHERE a.internship_id = ? AND a.status = 'pending'
        """,
        (internship_id,),
    )
    seekers = cur.fetchall()
    ranked = []
    for seeker in seekers:
        eligible, eligibility_notes = candidate_is_eligible(seeker)
        if not eligible:
            continue
        preference_rank = seeker["student_preference_rank"] or 1
        score, reasons, breakdown = compute_match_score(seeker, internship, preference_rank)
        research = compute_research_match(seeker, internship, preference_rank)
        student_preference_weight = 1 / max(int(preference_rank), 1)
        ranked.append({
            "application_id": seeker["application_id"],
            "user_id": seeker["user_id"],
            "name": seeker["name"],
            "phone": seeker["phone"],
            "skills": seeker["skills"],
            "score": score,
            "skill_similarity": research["skill_similarity"],
            "semantic_similarity": research["semantic_similarity"],
            "success_probability": research["ml_completion_probability"],
            "offer_acceptance_probability": research["offer_acceptance_probability_rule"],
            "opportunity_gap_score": research["opportunity_gap_score"],
            "policy_priority_score": research["policy_priority_score"],
            "ai_rank_score": research["ai_rank_score"],
            "student_preference_rank": preference_rank,
            "student_preference_weight": student_preference_weight,
            "reasons": reasons,
            "breakdown": breakdown,
            "eligibility_notes": eligibility_notes,
        })
    ranked.sort(
        key=lambda item: (
            item["ai_rank_score"],
            item["score"],
            item["student_preference_weight"],
            item["policy_priority_score"],
        ),
        reverse=True,
    )
    return internship, ranked


def candidate_has_live_pipeline(cur, seeker_user_id, exclude_internship_id=None):
    query = """
        SELECT COUNT(*)
        FROM applications
        WHERE seeker_user_id=?
          AND status IN ('allocated','interview_scheduled','verified','successful','accepted')
    """
    params = [seeker_user_id]
    if exclude_internship_id is not None:
        query += " AND internship_id != ?"
        params.append(exclude_internship_id)
    cur.execute(query, tuple(params))
    return cur.fetchone()[0] > 0


def normalize_student_preferences(cur, seeker_user_id):
    cur.execute(
        """
        SELECT id, student_preference_rank, created_at
        FROM applications
        WHERE seeker_user_id=?
        ORDER BY COALESCE(student_preference_rank, 999), datetime(COALESCE(created_at, updated_at)), id
        """,
        (seeker_user_id,),
    )
    for index, row in enumerate(cur.fetchall(), start=1):
        if row["student_preference_rank"] != index:
            cur.execute("UPDATE applications SET student_preference_rank=? WHERE id=?", (index, row["id"]))


def refresh_application_scoring(cur, application_id):
    cur.execute(
        """
        SELECT a.id, a.student_preference_rank, s.*, i.id AS internship_id
        FROM applications a
        JOIN seeker_profiles s ON a.seeker_user_id = s.user_id
        JOIN internships i ON a.internship_id = i.id
        WHERE a.id=?
        """,
        (application_id,),
    )
    joined = cur.fetchone()
    if not joined:
        return
    cur.execute("SELECT * FROM internships WHERE id=?", (joined["internship_id"],))
    internship = cur.fetchone()
    preference_rank = joined["student_preference_rank"] or 1
    score, reasons, breakdown = compute_match_score(joined, internship, preference_rank)
    research = compute_research_match(joined, internship, preference_rank)
    cur.execute(
        """
        UPDATE applications
        SET match_score=?, match_reason=?, skill_similarity=?, semantic_similarity=?,
            success_probability=?, offer_acceptance_probability=?, opportunity_gap_score=?,
            policy_priority_score=?, ai_rank_score=?, explanation=?, score_breakdown=?,
            updated_at=?
        WHERE id=?
        """,
        (
            score,
            "; ".join(reasons[:4]),
            research["skill_similarity"],
            research["semantic_similarity"],
            research["ml_completion_probability"],
            research["offer_acceptance_probability_rule"],
            research["opportunity_gap_score"],
            research["policy_priority_score"],
            research["ai_rank_score"],
            json.dumps(research["explanations"]),
            json.dumps(breakdown),
            now_text(),
            application_id,
        ),
    )


def allocate_candidate(cur, internship_id, seeker_user_id, admin_note="", allocation_batch=None):
    cur.execute("SELECT seats FROM internships WHERE id=?", (internship_id,))
    internship = cur.fetchone()
    if not internship:
        return False, "Internship not found."
    if internship["seats"] <= 0:
        return False, "No seats available."
    cur.execute("SELECT status, student_preference_rank FROM applications WHERE internship_id=? AND seeker_user_id=?", (internship_id, seeker_user_id))
    application = cur.fetchone()
    if not application or application["status"] != "pending":
        return False, "Candidate is no longer pending for this internship."
    cur.execute("SELECT * FROM seeker_profiles WHERE user_id=?", (seeker_user_id,))
    seeker = cur.fetchone()
    cur.execute("SELECT * FROM internships WHERE id=?", (internship_id,))
    full_internship = cur.fetchone()
    preference_rank = application["student_preference_rank"] if "student_preference_rank" in application.keys() else 1
    score, reasons, breakdown = compute_match_score(seeker, full_internship, preference_rank)
    research = compute_research_match(seeker, full_internship, preference_rank)

    timestamp = now_text()
    cur.execute(
        """
        UPDATE applications
        SET status='allocated', interview_status='not_scheduled', company_verification='pending',
            admin_decision_note=?, allocation_batch=?, updated_at=?,
            match_score=?, match_reason=?, skill_similarity=?, semantic_similarity=?,
            success_probability=?, offer_acceptance_probability=?, opportunity_gap_score=?,
            policy_priority_score=?, ai_rank_score=?, explanation=?, score_breakdown=?
        WHERE internship_id=? AND seeker_user_id=?
        """,
        (
            admin_note,
            allocation_batch,
            timestamp,
            score,
            "; ".join(reasons[:4]),
            research["skill_similarity"],
            research["semantic_similarity"],
            research["ml_completion_probability"],
            research["offer_acceptance_probability_rule"],
            research["opportunity_gap_score"],
            research["policy_priority_score"],
            research["ai_rank_score"],
            json.dumps(research["explanations"]),
            json.dumps(breakdown),
            internship_id,
            seeker_user_id,
        ),
    )
    cur.execute("UPDATE internships SET seats = seats - 1, status = 'active' WHERE id=? AND seats > 0", (internship_id,))
    return True, "Candidate allocated."


def build_gale_shapley_pool(cur, internship_filter=None):
    query = """
        SELECT a.id AS application_id, a.student_preference_rank, s.*, i.id AS target_internship_id
        FROM applications a
        JOIN seeker_profiles s ON a.seeker_user_id = s.user_id
        JOIN internships i ON a.internship_id = i.id
        WHERE a.status='pending'
          AND i.seats > 0
          AND i.status IN ('active','reapproval')
    """
    params = []
    if internship_filter is not None:
        query += " AND i.id=?"
        params.append(internship_filter)
    cur.execute(query, tuple(params))

    rows = []
    for seeker in cur.fetchall():
        cur.execute("SELECT * FROM internships WHERE id=?", (seeker["target_internship_id"],))
        internship = cur.fetchone()
        eligible, _ = candidate_is_eligible(seeker)
        if not eligible:
            continue
        preference_rank = seeker["student_preference_rank"] or 999
        score, reasons, breakdown = compute_match_score(seeker, internship, preference_rank)
        research = compute_research_match(seeker, internship, preference_rank)
        rows.append({
            "student_id": str(seeker["user_id"]),
            "internship_id": str(internship["id"]),
            "seeker_user_id": seeker["user_id"],
            "student_preference_rank": int(preference_rank),
            "ai_rank_score": research["ai_rank_score"],
            "match_score": score,
            "policy_priority_score": research["policy_priority_score"],
            "skill_similarity": research["skill_similarity"],
            "previous_internship": "No" if (seeker["first_time_intern"] or "").lower() == "yes" else "Yes",
            "rural_urban": seeker["rural_urban"] or "",
            "aspirational_district": seeker["aspirational_district"] or "",
            "income_group": income_to_group(seeker["income_bracket"]),
            "candidate_category": seeker["caste_category"] or "",
            "opportunity_gap_score": research["opportunity_gap_score"],
            "reasons": reasons,
            "breakdown": breakdown,
        })
    return rows


def gale_shapley_matches_for_pending(cur, internship_filter=None):
    rows = build_gale_shapley_pool(cur, internship_filter=internship_filter)
    if not rows:
        return [], []

    capacities = {}
    for row in rows:
        internship_id = int(row["internship_id"])
        if internship_id not in capacities:
            cur.execute("SELECT seats FROM internships WHERE id=?", (internship_id,))
            capacities[str(internship_id)] = max(0, int(cur.fetchone()["seats"]))

    by_student = {}
    by_internship = {}
    row_lookup = {}
    applicant_profiles = {}
    eligibility = {}
    for row in rows:
        by_student.setdefault(row["student_id"], []).append(row)
        by_internship.setdefault(row["internship_id"], []).append(row)
        row_lookup[(row["student_id"], row["internship_id"])] = row
        applicant_profiles[(row["student_id"], row["internship_id"])] = row
        eligibility[(row["student_id"], row["internship_id"])] = row["skill_similarity"] >= 0.05

    student_preferences = {
        student: [
            row["internship_id"]
            for row in sorted(
                items,
                key=lambda item: (
                    item["student_preference_rank"],
                    -item["ai_rank_score"],
                    -item["match_score"],
                ),
            )
        ]
        for student, items in by_student.items()
    }
    company_preferences = {
        internship: [
            row["student_id"]
            for row in sorted(
                items,
                key=lambda item: (
                    item["ai_rank_score"],
                    item["match_score"],
                    item["policy_priority_score"],
                    item["skill_similarity"],
                ),
                reverse=True,
            )
        ]
        for internship, items in by_internship.items()
    }

    matches = modified_gale_shapley(
        student_preferences,
        company_preferences,
        capacities,
        eligibility,
        applicant_profiles=applicant_profiles,
    )
    matched_rows = [row_lookup[(student, internship)] for student, internship in matches.items()]
    matched_rows.sort(key=lambda item: (int(item["internship_id"]), item["student_preference_rank"]))
    return matched_rows, []


def bulk_allocate_for_internship(cur, internship_id=None, allocation_batch=None):
    matched_rows, skipped = gale_shapley_matches_for_pending(cur, internship_filter=internship_id)
    allocated_count = 0
    for row in matched_rows:
        success, message = allocate_candidate(
            cur,
            int(row["internship_id"]),
            row["seeker_user_id"],
            admin_note="Provisionally allocated using Modified Gale-Shapley with student preference and AI company ranking.",
            allocation_batch=allocation_batch,
        )
        if success:
            allocated_count += 1
        else:
            skipped.append(f'{row["student_id"]}: {message}')
    return allocated_count, skipped


def require_role(role):
    return "user_id" in session and session.get("role") == role


initialize_database()
app = Flask(__name__)
app.secret_key = "hackathon_secret_key"


@app.context_processor
def inject_status_meta():
    return {"status_meta": STATUS_META}

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/admin/research_chart/<path:filename>")
def admin_research_chart(filename):
    if not require_role("admin"):
        return redirect(url_for("login"))
    allowed = {path.name for path in RESEARCH_CHARTS.glob("*.svg")}
    if filename not in allowed:
        abort(404)
    return send_from_directory(RESEARCH_CHARTS, filename)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        role = request.form.get("role")
        email = request.form.get("email")
        password = request.form.get("password")
        if not role or not email or not password:
            flash("All signup fields are required.", "error")
            return redirect(url_for("signup"))

        hashed = generate_password_hash(password)
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO users (role, email, password) VALUES (?, ?, ?)",
                (role, email.strip().lower(), hashed),
            )
            conn.commit()
            flash("Account created successfully. Please log in.", "success")
            return redirect(url_for("login"))
        except Exception:
            flash("Email already exists. Try another email.", "error")
            return redirect(url_for("signup"))
        finally:
            conn.close()

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        role = request.form.get("role")
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password")

        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email=? AND role=?", (email, role))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["role"] = user["role"]
            session["email"] = user["email"]
            flash("Login successful.", "success")
            return redirect(url_for(f"{role}_dashboard"))

        flash("Invalid login credentials.", "error")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("home"))


@app.route("/seeker/dashboard")
def seeker_dashboard():
    if not require_role("seeker"):
        return redirect(url_for("login"))

    user_id = session["user_id"]
    conn = get_connection()
    cur = conn.cursor()
    active_tab = request.args.get("tab", "profile")

    cur.execute("SELECT * FROM seeker_profiles WHERE user_id=?", (user_id,))
    seeker = cur.fetchone()

    recommendations = []
    if seeker:
        cur.execute("SELECT * FROM internships WHERE status IN ('active','reapproval') AND seats > 0 ORDER BY id DESC")
        internships = cur.fetchall()
        for index, internship in enumerate(internships, start=1):
            score, reasons, breakdown = compute_match_score(seeker, internship)
            recommendations.append({
                "internship_id": internship["id"],
                "role_title": internship["role_title"],
                "company_name": internship["company_name"],
                "sector": internship["sector"],
                "mode": internship["mode"],
                "location": internship["location"],
                "seats": internship["seats"],
                "score": score,
                "suggested_preference_rank": index,
                "reasons": reasons,
                "breakdown": breakdown,
            })
        recommendations.sort(key=lambda item: item["score"], reverse=True)
        for index, recommendation in enumerate(recommendations, start=1):
            recommendation["suggested_preference_rank"] = index

    cur.execute(
        """
        SELECT a.id, a.status, a.match_score, a.internship_id, a.interview_status, a.interview_date,
               a.company_verification, a.company_feedback, a.updated_at, a.offer_sent_at,
               a.student_preference_rank,
               i.role_title, i.company_name, i.location, i.mode, i.sector
        FROM applications a
        JOIN internships i ON a.internship_id = i.id
        WHERE a.seeker_user_id = ?
        ORDER BY COALESCE(a.student_preference_rank, 999), a.id DESC
        """,
        (user_id,),
    )
    applications = []
    applied_internship_ids = set()
    for row in cur.fetchall():
        app = dict(row)
        app["status_badge"] = status_badge(app["status"])
        applications.append(app)
        applied_internship_ids.add(app["internship_id"])

    conn.close()
    return render_template(
        "seeker_dashboard.html",
        seeker=seeker,
        recommendations=recommendations,
        applications=applications,
        has_profile=bool(seeker),
        active_tab=active_tab,
        applied_internship_ids=applied_internship_ids,
    )


@app.route("/seeker/profile", methods=["POST"])
def save_seeker_profile():
    if not require_role("seeker"):
        flash("Please login as seeker.", "error")
        return redirect(url_for("login"))

    user_id = session["user_id"]
    form = request.form

    try:
        age = int(form.get("age", "0"))
    except ValueError:
        flash("Invalid age value.", "error")
        return redirect(url_for("seeker_dashboard", tab="profile"))

    if age < 21 or age > 24:
        flash("Age must be between 21 and 24.", "error")
        return redirect(url_for("seeker_dashboard", tab="profile"))

    if form.get("degree") in {"M.Tech", "M.Sc", "MBA", "MCA"}:
        flash("Masters students are not allowed as per the scheme.", "error")
        return redirect(url_for("seeker_dashboard", tab="profile"))

    payload = (
        user_id,
        form.get("name"),
        form.get("phone"),
        age,
        form.get("marks10"),
        form.get("marks12"),
        form.get("gpa"),
        form.get("qualification"),
        form.get("degree"),
        form.get("branch"),
        form.get("college_name"),
        form.get("college_type"),
        form.get("skills"),
        form.get("experience_level"),
        form.get("sector_interest"),
        form.get("preferred_mode"),
        form.get("preferred_location"),
        form.get("caste_category"),
        form.get("rural_urban"),
        form.get("aspirational_district"),
        form.get("lor_available"),
        form.get("first_time_intern"),
        form.get("pwd"),
        form.get("income_bracket"),
        form.get("employed_fulltime"),
        form.get("student_fulltime"),
        form.get("distance_online_student"),
    )

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO seeker_profiles (
            user_id, name, phone, age, marks10, marks12, gpa,
            qualification, degree, branch, college_name, college_type,
            skills, experience_level, sector_interest, preferred_mode,
            preferred_location, caste_category, rural_urban, aspirational_district,
            lor_available, first_time_intern, pwd, income_bracket,
            employed_fulltime, student_fulltime, distance_online_student
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            name=excluded.name, phone=excluded.phone, age=excluded.age,
            marks10=excluded.marks10, marks12=excluded.marks12, gpa=excluded.gpa,
            qualification=excluded.qualification, degree=excluded.degree, branch=excluded.branch,
            college_name=excluded.college_name, college_type=excluded.college_type,
            skills=excluded.skills, experience_level=excluded.experience_level,
            sector_interest=excluded.sector_interest, preferred_mode=excluded.preferred_mode,
            preferred_location=excluded.preferred_location, caste_category=excluded.caste_category,
            rural_urban=excluded.rural_urban, aspirational_district=excluded.aspirational_district,
            lor_available=excluded.lor_available, first_time_intern=excluded.first_time_intern,
            pwd=excluded.pwd, income_bracket=excluded.income_bracket,
            employed_fulltime=excluded.employed_fulltime, student_fulltime=excluded.student_fulltime,
            distance_online_student=excluded.distance_online_student
        """,
        payload,
    )
    conn.commit()
    conn.close()

    flash("Profile saved successfully. Matching internships refreshed.", "success")
    return redirect(url_for("seeker_dashboard", tab="internships"))

@app.route("/company/dashboard")
def company_dashboard():
    if not require_role("company"):
        return redirect(url_for("login"))

    company_user_id = session["user_id"]
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM internships WHERE company_user_id = ? ORDER BY id DESC", (company_user_id,))
    internships = cur.fetchall()

    internship_cards = []
    totals = {"posted": 0, "waiting_admin": 0, "interview": 0, "verified": 0, "offers": 0, "accepted": 0}
    for internship in internships:
        totals["posted"] += 1
        cur.execute(
            "SELECT status, COUNT(*) AS count FROM applications WHERE internship_id = ? GROUP BY status",
            (internship["id"],),
        )
        status_counts = {row["status"]: row["count"] for row in cur.fetchall()}

        cur.execute(
            """
            SELECT a.*, s.name, s.phone, s.skills
            FROM applications a
            JOIN seeker_profiles s ON a.seeker_user_id = s.user_id
            WHERE a.internship_id=?
              AND a.status IN ('allocated','interview_scheduled','verified','successful','accepted')
            ORDER BY CASE a.status
                WHEN 'accepted' THEN 5
                WHEN 'successful' THEN 4
                WHEN 'verified' THEN 3
                WHEN 'interview_scheduled' THEN 2
                WHEN 'allocated' THEN 1
                ELSE 0
            END DESC, a.match_score DESC
            """,
            (internship["id"],),
        )
        live_candidates = cur.fetchall()

        if status_counts.get("pending", 0):
            totals["waiting_admin"] += 1
        if status_counts.get("interview_scheduled", 0):
            totals["interview"] += status_counts["interview_scheduled"]
        if status_counts.get("verified", 0):
            totals["verified"] += status_counts["verified"]
        if status_counts.get("successful", 0):
            totals["offers"] += status_counts["successful"]
        if status_counts.get("accepted", 0):
            totals["accepted"] += status_counts["accepted"]

        internship_cards.append({
            "internship": internship,
            "status_counts": status_counts,
            "live_candidates": live_candidates,
            "app_count": sum(status_counts.values()),
        })

    conn.close()
    return render_template("company_dashboard.html", internships=internship_cards, totals=totals)


@app.route("/company/post", methods=["POST"])
def company_post_internship():
    if not require_role("company"):
        flash("Please login as company.", "error")
        return redirect(url_for("login"))

    form = request.form
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO internships (
            company_user_id, company_name, role_title, sector, required_skills,
            seats, stipend, mode, location, min_gpa, status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
        """,
        (
            session["user_id"],
            form.get("company_name") or session.get("email"),
            form.get("role"),
            form.get("sector"),
            form.get("skills"),
            form.get("seats"),
            form.get("stipend"),
            form.get("mode"),
            form.get("location"),
            form.get("min_gpa"),
        ),
    )
    conn.commit()
    conn.close()

    flash("Internship posted successfully.", "success")
    return redirect(url_for("company_dashboard"))


@app.route("/admin/dashboard")
def admin_dashboard():
    if not require_role("admin"):
        return redirect(url_for("login"))

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT i.*,
               (SELECT COUNT(*) FROM applications a WHERE a.internship_id = i.id) AS app_count,
               (SELECT COUNT(*) FROM applications a WHERE a.internship_id = i.id AND a.status='pending') AS pending_count,
               (SELECT COUNT(*) FROM applications a WHERE a.internship_id = i.id AND a.status='allocated') AS allocated_count,
               (SELECT COUNT(*) FROM applications a WHERE a.internship_id = i.id AND a.status='interview_scheduled') AS interview_count,
               (SELECT COUNT(*) FROM applications a WHERE a.internship_id = i.id AND a.status='verified') AS verified_count,
               (SELECT COUNT(*) FROM applications a WHERE a.internship_id = i.id AND a.status='successful') AS offer_count,
               (SELECT COUNT(*) FROM applications a WHERE a.internship_id = i.id AND a.status='accepted') AS accepted_count
        FROM internships i
        ORDER BY i.id DESC
        """
    )
    internships = cur.fetchall()

    stats = {}
    for status in STATUS_META:
        cur.execute("SELECT COUNT(*) FROM applications WHERE status=?", (status,))
        stats[status] = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM internships WHERE status='reapproval'")
    reapproval_needed = cur.fetchone()[0]
    cur.execute(
        """
        SELECT COUNT(*)
        FROM applications a
        JOIN seeker_profiles s ON a.seeker_user_id = s.user_id
        WHERE a.status IN ('allocated','interview_scheduled','verified','successful','accepted')
          AND lower(COALESCE(s.rural_urban,''))='rural'
        """
    )
    rural_live = cur.fetchone()[0]
    cur.execute(
        """
        SELECT COUNT(*)
        FROM applications a
        JOIN seeker_profiles s ON a.seeker_user_id = s.user_id
        WHERE a.status IN ('allocated','interview_scheduled','verified','successful','accepted')
          AND lower(COALESCE(s.aspirational_district,''))='yes'
        """
    )
    aspirational_live = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM internships WHERE seats > 0")
    open_internships = cur.fetchone()[0]
    research_snapshot = load_research_snapshot()

    conn.close()
    return render_template(
        "admin_dashboard.html",
        internships=internships,
        stats=stats,
        reapproval_needed=reapproval_needed,
        rural_live=rural_live,
        aspirational_live=aspirational_live,
        open_internships=open_internships,
        research=research_snapshot,
    )


@app.route("/seeker/apply", methods=["POST"])
def seeker_apply():
    if not require_role("seeker"):
        flash("Please login as seeker to apply.", "error")
        return redirect(url_for("login"))

    seeker_user_id = session["user_id"]
    internship_id = int(request.form.get("internship_id"))

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM seeker_profiles WHERE user_id=?", (seeker_user_id,))
    seeker = cur.fetchone()
    if not seeker:
        conn.close()
        flash("Please complete your profile before applying.", "error")
        return redirect(url_for("seeker_dashboard", tab="profile"))

    cur.execute("SELECT * FROM internships WHERE id=?", (internship_id,))
    internship = cur.fetchone()
    if not internship:
        conn.close()
        flash("Internship not found.", "error")
        return redirect(url_for("seeker_dashboard", tab="internships"))

    cur.execute("SELECT COUNT(*) FROM applications WHERE seeker_user_id=? AND internship_id=?", (seeker_user_id, internship_id))
    if cur.fetchone()[0]:
        conn.close()
        flash("You already applied for this internship.", "error")
        return redirect(url_for("seeker_dashboard", tab="internships"))

    cur.execute("SELECT COUNT(*) FROM applications WHERE seeker_user_id=?", (seeker_user_id,))
    application_count = cur.fetchone()[0]
    preference_rank = application_count + 1

    score, reasons, breakdown = compute_match_score(seeker, internship, preference_rank)
    research = compute_research_match(seeker, internship, preference_rank)
    timestamp = now_text()
    cur.execute(
        """
        INSERT INTO applications (
            seeker_user_id, internship_id, match_score, match_reason, status,
            created_at, updated_at, interview_status, company_verification,
            skill_similarity, semantic_similarity, success_probability,
            offer_acceptance_probability, opportunity_gap_score, policy_priority_score,
            ai_rank_score, explanation, score_breakdown, student_preference_rank
        )
        VALUES (?, ?, ?, ?, 'pending', ?, ?, 'not_scheduled', 'pending',
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            seeker_user_id,
            internship_id,
            score,
            "; ".join(reasons[:4]),
            timestamp,
            timestamp,
            research["skill_similarity"],
            research["semantic_similarity"],
            research["ml_completion_probability"],
            research["offer_acceptance_probability_rule"],
            research["opportunity_gap_score"],
            research["policy_priority_score"],
            research["ai_rank_score"],
            json.dumps(research["explanations"]),
            json.dumps(breakdown),
            preference_rank,
        ),
    )
    normalize_student_preferences(cur, seeker_user_id)
    conn.commit()
    conn.close()

    flash(f"Application submitted successfully. It was added as preference #{preference_rank}. You can reorder it from Application Status.", "success")
    return redirect(url_for("seeker_dashboard", tab="status"))


@app.route("/seeker/preferences", methods=["POST"])
def seeker_update_preferences():
    if not require_role("seeker"):
        flash("Please login as seeker.", "error")
        return redirect(url_for("login"))

    seeker_user_id = session["user_id"]
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("BEGIN IMMEDIATE")
    try:
        cur.execute(
            """
            SELECT id
            FROM applications
            WHERE seeker_user_id=?
              AND status NOT IN ('accepted','rejected')
            """,
            (seeker_user_id,),
        )
        application_ids = [row["id"] for row in cur.fetchall()]
        ranked_updates = []
        for application_id in application_ids:
            try:
                rank = int(request.form.get(f"rank_{application_id}", "999"))
            except ValueError:
                rank = 999
            ranked_updates.append((max(rank, 1), application_id))

        ranked_updates.sort(key=lambda item: (item[0], item[1]))
        for index, (_, application_id) in enumerate(ranked_updates, start=1):
            cur.execute(
                "UPDATE applications SET student_preference_rank=?, updated_at=? WHERE id=?",
                (index, now_text(), application_id),
            )
            refresh_application_scoring(cur, application_id)
        normalize_student_preferences(cur, seeker_user_id)
        conn.commit()
    finally:
        conn.close()

    flash("Preference list updated. Recommendation and allocation scores were refreshed.", "success")
    return redirect(url_for("seeker_dashboard", tab="status"))


@app.route("/admin/internship/<int:internship_id>/matches")
def admin_view_matches(internship_id):
    if not require_role("admin"):
        return redirect(url_for("login"))

    conn = get_connection()
    cur = conn.cursor()
    internship, ranked = get_ranked_candidates(cur, internship_id)
    if not internship:
        conn.close()
        flash("Internship not found.", "error")
        return redirect(url_for("admin_dashboard"))

    cur.execute("SELECT COUNT(*) FROM applications WHERE internship_id=?", (internship_id,))
    total_applications = cur.fetchone()[0]
    cur.execute(
        "SELECT COUNT(*) FROM applications WHERE internship_id=? AND status IN ('allocated','interview_scheduled','verified','successful','accepted')",
        (internship_id,),
    )
    live_pipeline_count = cur.fetchone()[0]
    conn.close()

    return render_template(
        "admin_matches.html",
        internship=internship,
        ranked=ranked,
        total_applications=total_applications,
        live_pipeline_count=live_pipeline_count,
    )


@app.route("/admin/approve", methods=["POST"])
def admin_approve_allocation():
    if not require_role("admin"):
        flash("Please login as admin.", "error")
        return redirect(url_for("login"))

    internship_id = int(request.form.get("internship_id"))
    seeker_user_id = int(request.form.get("seeker_user_id"))
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("BEGIN IMMEDIATE")
    try:
        success, message = allocate_candidate(
            cur,
            internship_id,
            seeker_user_id,
            admin_note="Manually approved by admin after rank review.",
            allocation_batch=f"single-{uuid.uuid4().hex[:8]}",
        )
        if not success:
            conn.rollback()
            flash(message, "error")
            return redirect(url_for("admin_view_matches", internship_id=internship_id))
        conn.commit()
        flash("Candidate approved and allocated successfully.", "success")
    finally:
        conn.close()

    return redirect(url_for("admin_view_matches", internship_id=internship_id))


@app.route("/admin/bulk_approve", methods=["POST"])
def admin_bulk_approve():
    if not require_role("admin"):
        flash("Please login as admin.", "error")
        return redirect(url_for("login"))

    internship_id = int(request.form.get("internship_id"))
    batch_id = f"bulk-{uuid.uuid4().hex[:8]}"
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("BEGIN IMMEDIATE")
    try:
        allocated_count, skipped = bulk_allocate_for_internship(cur, internship_id, allocation_batch=batch_id)
        conn.commit()
        if allocated_count:
            flash(f"Bulk approved {allocated_count} candidate(s).", "success")
        else:
            flash("No candidates were bulk approved.", "error")
        if skipped:
            flash("Skipped: " + " | ".join(skipped[:3]), "error")
    finally:
        conn.close()

    return redirect(url_for("admin_view_matches", internship_id=internship_id))


@app.route("/admin/bulk_approve_all", methods=["POST"])
def admin_bulk_approve_all():
    if not require_role("admin"):
        flash("Please login as admin.", "error")
        return redirect(url_for("login"))

    batch_id = f"system-{uuid.uuid4().hex[:8]}"
    total_allocated = 0
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("BEGIN IMMEDIATE")
    try:
        total_allocated, _ = bulk_allocate_for_internship(cur, None, allocation_batch=batch_id)
        conn.commit()
    finally:
        conn.close()

    if total_allocated:
        flash(f"Bulk allocation completed. {total_allocated} candidate(s) allocated.", "success")
    else:
        flash("No bulk allocations were made. Check pending applicants and seat availability.", "error")
    return redirect(url_for("admin_dashboard"))


@app.route("/company/view_candidate/<int:internship_id>")
def company_view_candidate(internship_id):
    if not require_role("company"):
        return redirect(url_for("login"))

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM internships WHERE id=? AND company_user_id=?", (internship_id, session["user_id"]))
    internship = cur.fetchone()
    if not internship:
        conn.close()
        flash("Internship not found for this company.", "error")
        return redirect(url_for("company_dashboard"))

    cur.execute(
        """
        SELECT a.*, s.name, s.age, s.phone, s.skills, s.gpa, s.degree, s.branch,
               s.sector_interest, s.preferred_mode, s.preferred_location, s.rural_urban,
               s.caste_category, s.aspirational_district
        FROM applications a
        JOIN seeker_profiles s ON a.seeker_user_id = s.user_id
        WHERE a.internship_id=?
          AND a.status IN ('allocated','interview_scheduled','verified','successful','accepted')
        ORDER BY CASE a.status
            WHEN 'accepted' THEN 5
            WHEN 'successful' THEN 4
            WHEN 'verified' THEN 3
            WHEN 'interview_scheduled' THEN 2
            WHEN 'allocated' THEN 1
            ELSE 0
        END DESC, a.match_score DESC
        """,
        (internship_id,),
    )
    candidates = []
    for row in cur.fetchall():
        item = dict(row)
        item["status_badge"] = status_badge(item["status"])
        candidates.append(item)

    conn.close()
    if not candidates:
        flash("No allocated candidate pipeline is active yet.", "error")
        return redirect(url_for("company_dashboard"))

    return render_template("company_view_candidate.html", internship=internship, candidates=candidates)

@app.route("/company/schedule_interview", methods=["POST"])
def company_schedule_interview():
    if not require_role("company"):
        flash("Please login as company.", "error")
        return redirect(url_for("login"))

    internship_id = int(request.form.get("internship_id"))
    seeker_user_id = int(request.form.get("seeker_user_id"))
    interview_date = request.form.get("interview_date") or "To be announced"
    notes = request.form.get("company_feedback") or "Interview scheduled for profile verification."

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE applications
        SET status='interview_scheduled',
            interview_status='scheduled',
            interview_date=?,
            company_feedback=?,
            updated_at=?
        WHERE internship_id=? AND seeker_user_id=? AND status='allocated'
        """,
        (interview_date, notes, now_text(), internship_id, seeker_user_id),
    )
    conn.commit()
    conn.close()

    flash("Interview call scheduled for the candidate.", "success")
    return redirect(url_for("company_view_candidate", internship_id=internship_id))


@app.route("/company/verify_candidate", methods=["POST"])
def company_verify_candidate():
    if not require_role("company"):
        flash("Please login as company.", "error")
        return redirect(url_for("login"))

    internship_id = int(request.form.get("internship_id"))
    seeker_user_id = int(request.form.get("seeker_user_id"))
    feedback = request.form.get("company_feedback") or "Candidate verified successfully after interview."

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE applications
        SET status='verified',
            interview_status='completed',
            company_verification='verified',
            company_feedback=?,
            verified_at=?,
            updated_at=?
        WHERE internship_id=? AND seeker_user_id=? AND status IN ('allocated','interview_scheduled')
        """,
        (feedback, now_text(), now_text(), internship_id, seeker_user_id),
    )
    conn.commit()
    conn.close()

    flash("Candidate verified by company. Final offer can now be sent.", "success")
    return redirect(url_for("company_view_candidate", internship_id=internship_id))


@app.route("/company/reject_candidate", methods=["POST"])
def company_reject_candidate():
    if not require_role("company"):
        flash("Please login as company.", "error")
        return redirect(url_for("login"))

    internship_id = int(request.form.get("internship_id"))
    seeker_user_id = int(request.form.get("seeker_user_id"))
    feedback = request.form.get("company_feedback") or "Rejected after interview verification."

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("BEGIN IMMEDIATE")
    try:
        cur.execute(
            """
            UPDATE applications
            SET status='rejected',
                interview_status='completed',
                company_verification='rejected',
                company_feedback=?,
                updated_at=?
            WHERE internship_id=? AND seeker_user_id=?
              AND status IN ('allocated','interview_scheduled','verified')
            """,
            (feedback, now_text(), internship_id, seeker_user_id),
        )
        reallocated_count = 0
        if cur.rowcount:
            cur.execute("UPDATE internships SET seats = seats + 1, status = 'reapproval' WHERE id=?", (internship_id,))
            reallocated_count, _ = bulk_allocate_for_internship(
                cur,
                internship_id,
                allocation_batch=f"realloc-{uuid.uuid4().hex[:8]}",
            )
        conn.commit()
    finally:
        conn.close()

    if reallocated_count:
        flash("Candidate rejected by company. The released seat was dynamically reallocated.", "success")
    else:
        flash("Candidate rejected by company. Seat reopened for reapproval.", "success")
    return redirect(url_for("company_view_candidate", internship_id=internship_id))


@app.route("/company/give_offer", methods=["POST"])
def company_give_offer():
    if not require_role("company"):
        flash("Please login as company.", "error")
        return redirect(url_for("login"))

    internship_id = int(request.form.get("internship_id"))
    seeker_user_id = int(request.form.get("seeker_user_id"))

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE applications
        SET status='successful',
            company_verification='verified',
            offer_sent_at=?,
            updated_at=?
        WHERE internship_id=? AND seeker_user_id=? AND status='verified'
        """,
        (now_text(), now_text(), internship_id, seeker_user_id),
    )
    conn.commit()
    conn.close()

    flash("Final offer sent after company verification.", "success")
    return redirect(url_for("company_view_candidate", internship_id=internship_id))


@app.route("/company/send_verified_offers", methods=["POST"])
def company_send_verified_offers():
    if not require_role("company"):
        flash("Please login as company.", "error")
        return redirect(url_for("login"))

    internship_id = int(request.form.get("internship_id"))
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE applications
        SET status='successful',
            company_verification='verified',
            offer_sent_at=?,
            updated_at=?
        WHERE internship_id=? AND status='verified'
        """,
        (now_text(), now_text(), internship_id),
    )
    updated = cur.rowcount
    conn.commit()
    conn.close()

    if updated:
        flash(f"Sent final offers to {updated} verified candidate(s).", "success")
    else:
        flash("No verified candidates are waiting for an offer.", "error")
    return redirect(url_for("company_view_candidate", internship_id=internship_id))


@app.route("/seeker/accept_offer", methods=["POST"])
def seeker_accept_offer():
    if not require_role("seeker"):
        flash("Please login as seeker.", "error")
        return redirect(url_for("login"))

    seeker_user_id = session["user_id"]
    internship_id = int(request.form.get("internship_id"))
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("BEGIN IMMEDIATE")

    try:
        cur.execute("SELECT COUNT(*) FROM applications WHERE seeker_user_id=? AND status='accepted'", (seeker_user_id,))
        if cur.fetchone()[0]:
            conn.rollback()
            flash("You already accepted one offer.", "error")
            return redirect(url_for("seeker_dashboard", tab="status"))

        cur.execute(
            """
            UPDATE applications
            SET status='accepted', updated_at=?
            WHERE seeker_user_id=? AND internship_id=? AND status='successful'
            """,
            (now_text(), seeker_user_id, internship_id),
        )
        if not cur.rowcount:
            conn.rollback()
            flash("Only final verified offers can be accepted.", "error")
            return redirect(url_for("seeker_dashboard", tab="status"))

        cur.execute(
            """
            SELECT internship_id
            FROM applications
            WHERE seeker_user_id=? AND internship_id != ?
              AND status IN ('allocated','interview_scheduled','verified','successful')
            """,
            (seeker_user_id, internship_id),
        )
        other_live = cur.fetchall()

        cur.execute(
            """
            UPDATE applications
            SET status='rejected', updated_at=?
            WHERE seeker_user_id=? AND internship_id != ?
              AND status IN ('allocated','interview_scheduled','verified','successful','pending')
            """,
            (now_text(), seeker_user_id, internship_id),
        )

        reallocated_total = 0
        for row in other_live:
            cur.execute("UPDATE internships SET seats = seats + 1, status = 'reapproval' WHERE id=?", (row["internship_id"],))
            reallocated_count, _ = bulk_allocate_for_internship(
                cur,
                row["internship_id"],
                allocation_batch=f"choice-realloc-{uuid.uuid4().hex[:8]}",
            )
            reallocated_total += reallocated_count

        conn.commit()
    finally:
        conn.close()

    if reallocated_total:
        flash(f"Offer accepted successfully. Other released seat(s) triggered {reallocated_total} dynamic reallocation(s).", "success")
    else:
        flash("Offer accepted successfully. Other active applications have been closed.", "success")
    return redirect(url_for("seeker_dashboard", tab="status"))


@app.route("/admin/view_seeker/<int:seeker_user_id>/internship/<int:internship_id>")
def admin_view_seeker(seeker_user_id, internship_id):
    if not require_role("admin"):
        return redirect(url_for("login"))

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM internships WHERE id=?", (internship_id,))
    internship = cur.fetchone()
    cur.execute("SELECT * FROM seeker_profiles WHERE user_id=?", (seeker_user_id,))
    seeker = cur.fetchone()
    cur.execute(
        """
        SELECT status, match_score, match_reason, interview_status, interview_date,
               company_verification, company_feedback, admin_decision_note, offer_sent_at
        FROM applications
        WHERE seeker_user_id=? AND internship_id=?
        LIMIT 1
        """,
        (seeker_user_id, internship_id),
    )
    application = cur.fetchone()
    conn.close()

    if not internship or not seeker:
        flash("Seeker or internship not found.", "error")
        return redirect(url_for("admin_dashboard"))

    return render_template("admin_view_seeker.html", seeker=seeker, internship=internship, application=application)


if __name__ == "__main__":
    app.run(debug=True)
