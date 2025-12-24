import os
import shutil

from cs50 import SQL
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash

# SQLite connection string for the same DB your Flask app uses
DB_URL = "sqlite:///resumemash.db"

# Path to the ZIP file containing all resumes you want to bulk import
ZIP_PATH = "bulk_resumes.zip"

# Temporary directory where we’ll unpack the ZIP contents
SOURCE_DIR = "bulk_resumes"

# Directory where the Flask app expects to find resume PDFs
UPLOAD_DIR = "uploads"


def extract_text_and_title(path):
    """
    Given a path to a PDF file, extract:

      - full_text: all text from the PDF (concatenated across pages)
      - title:     the PDF's metadata title, if present

    Returns:
      (full_text, title_str)
    """
    full_text = ""
    title = ""

    try:
        reader = PdfReader(path)

        # Extract text from all pages
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        full_text = "\n".join(text_parts).strip()

        # Extract metadata title if available
        meta = getattr(reader, "metadata", None)
        if meta:
            # PyPDF2 3.x style: meta.title or fallback meta["/Title"]
            if getattr(meta, "title", None):
                title = str(meta.title)
            elif "/Title" in meta:
                title = str(meta["/Title"])
    except Exception as e:
        print(f"Error reading {path}: {e}")
        full_text = ""

    # If we couldn't get any text, store a placeholder
    if not full_text:
        full_text = "(No text could be extracted from this PDF.)"

    return full_text, (title or "")


def guess_name_from_text(text, fallback_filename):
    """
    Try to guess first and last name from the first non-empty line in the resume.

    Strategy:
      1. Look at the first line of real text (ignoring blanks).
      2. Split on spaces/pipes, keep only alphabetic tokens.
      3. If that looks name-like, use it as first + last.
      4. Otherwise, fall back to guessing from the filename.

    Returns:
      (first_name, last_name)
    """
    if text:
        # Take all non-empty lines
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            # First line is often the candidate’s name
            first_line = lines[0]
            # Replace '|' separators and split into words, keep only alphabetic
            parts = [p for p in first_line.replace("|", " ").split() if p.isalpha()]

            # If it looks short enough to be a human name, use it
            if 1 <= len(parts) <= 4:
                first = parts[0].title()
                if len(parts) >= 2:
                    last = " ".join(parts[1:]).title()
                else:
                    last = "Candidate"
                return first, last

    # Fallback: derive something from the filename
    base = os.path.splitext(fallback_filename)[0]
    base = base.replace("_", " ").replace("-", " ")
    parts = base.split()
    if len(parts) >= 2:
        first = parts[0].title()
        last = " ".join(parts[1:]).title()
    else:
        first = base.title() if base else "Unknown"
        last = "Candidate"
    return first, last


def guess_job_field(text, filename, pdf_title):
    """
    Heuristic "AI" that guesses job_field from keywords.

    It inspects:
      - resume text
      - the filename
      - the PDF metadata title

    and assigns one of:
      'software', 'data', 'finance', 'consulting',
      'marketing', 'product', or 'general'
    """
    # Combine text, filename, and title into one blob and lowercase everything
    combined = " ".join(
        [
            text or "",
            filename or "",
            pdf_title or "",
        ]
    ).lower()

    if not combined.strip():
        # Nothing to work with, so default to 'general'
        return "general"

    # Initialize score buckets for each field
    scores = {
        "software": 0,
        "data": 0,
        "finance": 0,
        "consulting": 0,
        "marketing": 0,
        "product": 0,
    }

    # Helper to increment scores when we see specific keywords
    def bump(field, kws, weight=2):
        for kw in kws:
            if kw in combined:
                scores[field] += weight

    # Software / Engineering keywords
    bump(
        "software",
        [
            "software engineer", "software developer", "developer", "programmer",
            "python", "java", "c++", "c#", "javascript", "typescript", "react",
            "node", "api", "backend", "front end", "frontend",
            "computer science", "cs major", "git", "github",
        ],
    )

    # Data / Analytics keywords
    bump(
        "data",
        [
            "data scientist", "data science", "data analyst", "analytics",
            "machine learning", "ml engineer", "pandas", "numpy", "sql",
            "statistics", "regression", "tableau", "power bi", "ga4",
        ],
    )

    # Finance keywords
    bump(
        "finance",
        [
            "investment banking", "investment banker", "private equity",
            "hedge fund", "trading", "trader", "financial analyst",
            "equity research", "valuation", "dcf", "discounted cash flow",
            "leveraged buyout", "lbo", "m&a", "capital markets",
        ],
    )

    # Consulting keywords
    bump(
        "consulting",
        [
            "consultant", "consulting", "strategy consultant", "management consulting",
            "mckinsey", "bain", "bcg", "case interview", "client engagement",
        ],
    )

    # Marketing keywords
    bump(
        "marketing",
        [
            "marketing", "social media", "seo", "sem", "campaign", "digital ads",
            "content creator", "brand", "branding", "advertising", "copywriting",
        ],
    )

    # Product Management keywords
    bump(
        "product",
        [
            "product manager", "product management", "product owner",
            "product roadmap", "user stories", "requirements gathering",
            "feature prioritization", "a/b test", "ab test", "user research",
        ],
    )

    # Choose the field with the highest score
    best_field = max(scores, key=scores.get)
    if scores[best_field] == 0:
        # No field had any hits → default to 'general'
        return "general"
    return best_field


def main():
    """
    Bulk import pipeline:

      1. Connect to DB
      2. Unpack ZIP of resume PDFs into a temp folder
      3. For each PDF:
          - extract text + metadata title
          - guess candidate name
          - guess job_field
          - create a candidate user account
          - copy the PDF into uploads/
          - insert a resume row tied to that user and field
    """
    db = SQL(DB_URL)

    # Make sure uploads folder exists so Flask can serve the PDFs
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Check that the ZIP file actually exists
    if not os.path.isfile(ZIP_PATH):
        print(f"ZIP file '{ZIP_PATH}' not found.")
        return

    # Clean out old SOURCE_DIR if it exists, then recreate it fresh
    if os.path.isdir(SOURCE_DIR):
        shutil.rmtree(SOURCE_DIR)
    os.makedirs(SOURCE_DIR, exist_ok=True)

    print(f"Unpacking {ZIP_PATH} into {SOURCE_DIR}...")
    # Unpack everything in the ZIP into SOURCE_DIR
    shutil.unpack_archive(ZIP_PATH, SOURCE_DIR)

    # Get a sorted list of files in the unpacked directory
    files = sorted(os.listdir(SOURCE_DIR))
    imported = 0

    for name in files:
        # Skip non-PDFs
        if not name.lower().endswith(".pdf"):
            continue

        src_path = os.path.join(SOURCE_DIR, name)
        if not os.path.isfile(src_path):
            continue

        print(f"Processing {name}...")

        # Use a safe filename (no weird characters) when working locally
        safe_name = secure_filename(name)
        safe_src_path = os.path.join(SOURCE_DIR, safe_name)
        if safe_src_path != src_path:
            # If secure_filename changed the name, copy into new path
            shutil.copy(src_path, safe_src_path)
        else:
            # Otherwise just use the original
            safe_src_path = src_path

        # Extract text + PDF metadata title
        text, pdf_title = extract_text_and_title(safe_src_path)

        # Guess candidate first and last name
        first_name, last_name = guess_name_from_text(text, name)

        # Guess job field from text + filename + title
        job_field = guess_job_field(text, name, pdf_title)

        # Build a base username from the sanitized filename
        base_username = os.path.splitext(safe_name)[0].replace(" ", "").lower()
        username = base_username

        # Ensure username is unique in the DB by adding suffixes if needed
        suffix = 1
        while db.execute("SELECT id FROM users WHERE username = ?", username):
            suffix += 1
            username = f"{base_username}_{suffix}"

        # Auto-generate email/phone/role/password for bulk-created candidates
        email = f"{username}@example.com"
        phone = "000-000-0000"
        role = "candidate"
        password_hash = generate_password_hash("placeholder-password")

        # Insert a new candidate user
        db.execute(
            """
            INSERT INTO users (username, hash, role, first_name, last_name, email, phone)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            username,
            password_hash,
            role,
            first_name,
            last_name,
            email,
            phone,
        )

        # Fetch the new user’s id
        user_row = db.execute("SELECT id FROM users WHERE username = ?", username)[0]
        user_id = user_row["id"]

        # Copy the original PDF into uploads/ so the Flask app can embed it
        dest_filename = secure_filename(name)
        dest_path = os.path.join(UPLOAD_DIR, dest_filename)
        shutil.copy(src_path, dest_path)

        # Insert resume row with text + job_field
        db.execute(
            """
            INSERT INTO resumes (user_id, filename, text, job_field)
            VALUES (?, ?, ?, ?)
            """,
            user_id,
            dest_filename,
            text,
            job_field,
        )

        imported += 1
        print(
            f"Imported resume for {first_name} {last_name} "
            f"as user '{username}' (field: {job_field})."
        )

    print(f"Done. Imported {imported} resumes.")


if __name__ == "__main__":
    main()
