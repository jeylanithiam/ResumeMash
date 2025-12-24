from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
from cs50 import SQL
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash
from functools import wraps
import os
import random

from ml_model import train_model, score_text


UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev"  # session signing key (fine for demo)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload size

# Make sure uploads folder exists so we can save PDFs there
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Database setup

# Use CS50’s SQL helper with a SQLite database file
db = SQL("sqlite:///resumemash.db")

# Users table:
# - username + password hash for auth
# - role = "candidate" or "recruiter"
# - basic profile info for candidates / recruiters
db.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        hash TEXT NOT NULL,
        role TEXT NOT NULL CHECK (role IN ('candidate', 'recruiter')),
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        email TEXT NOT NULL,
        phone TEXT NOT NULL
    )
""")

# Resumes table:
# - stores which user uploaded it
# - filename for the PDF
# - extracted text content
# - job_field: what type of role this resume is targeting
db.execute("""
    CREATE TABLE IF NOT EXISTS resumes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        filename TEXT NOT NULL,
        text TEXT NOT NULL,
        job_field TEXT NOT NULL DEFAULT 'unspecified',
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

# If resumes table already existed without job_field, add it
try:
    db.execute("SELECT job_field FROM resumes LIMIT 1")
except Exception:
    db.execute("ALTER TABLE resumes ADD COLUMN job_field TEXT NOT NULL DEFAULT 'unspecified'")

# Swipes table:
# - one row per recruiter swipe
# - label: 1 = like / Mash, 0 = pass
# - used as training data for ML model
db.execute("""
    CREATE TABLE IF NOT EXISTS swipes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        resume_id INTEGER NOT NULL,
        user_id INTEGER,
        label INTEGER NOT NULL, -- 1 = like, 0 = pass
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")


# Helpers

def allowed_file(filename):
    """
    Return True if filename has a valid extension (currently: only PDF).
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def login_required(f):
    """
    Decorator: protect routes so only logged-in users can access them.

    If user is not logged in, redirect to /login and flash a message.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in first.")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


# Routes

@app.route("/")
def index():
    """
    Landing page.

    - If logged in as candidate: shows upload resume.
    - If logged in as recruiter: shows start swiping.
    - If not logged in: shows login / sign up buttons.
    """
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """
    Registration page.

    GET: show form.
    POST: validate input, create new user, log them in.
    """
    if request.method == "POST":
        # Grab form fields
        username = request.form.get("username")
        password = request.form.get("password")
        confirmation = request.form.get("confirmation")
        role = request.form.get("role")  # "candidate" or "recruiter"
        first_name = request.form.get("first_name")
        last_name = request.form.get("last_name")
        email = request.form.get("email")
        phone = request.form.get("phone")

        # Basic validation: make sure everything is filled
        if not all([username, password, confirmation, role, first_name, last_name, email, phone]):
            flash("Please fill out all fields.")
            return redirect(url_for("register"))

        # Password confirmation check
        if password != confirmation:
            flash("Passwords do not match.")
            return redirect(url_for("register"))

        # Role must be one of the allowed types
        if role not in ("candidate", "recruiter"):
            flash("Invalid role selected.")
            return redirect(url_for("register"))

        # Check for duplicate username
        rows = db.execute("SELECT id FROM users WHERE username = ?", username)
        if rows:
            flash("Username already taken.")
            return redirect(url_for("register"))

        # All good -> insert new user with hashed password
        hash_ = generate_password_hash(password)
        db.execute(
            """
            INSERT INTO users (username, hash, role, first_name, last_name, email, phone)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            username,
            hash_,
            role,
            first_name,
            last_name,
            email,
            phone,
        )

        # Immediately log the user in and store their info in the session
        user = db.execute("SELECT id, role FROM users WHERE username = ?", username)[0]
        session["user_id"] = user["id"]
        session["username"] = username
        session["role"] = user["role"]

        flash("Account created. You are now logged in.")
        return redirect(url_for("index"))

    # GET: show the registration form
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """
    Log in page.

    GET: show form.
    POST: check username + password, set session, redirect to home.
    """
    # Clear any existing session so we start clean
    session.clear()

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # Basic check that both fields are provided
        if not username or not password:
            flash("Please provide username and password.")
            return redirect(url_for("login"))

        # Look up the user
        rows = db.execute("SELECT * FROM users WHERE username = ?", username)
        # Ensure exactly one match and that password hash matches
        if len(rows) != 1 or not check_password_hash(rows[0]["hash"], password):
            flash("Invalid username and/or password.")
            return redirect(url_for("login"))

        # Success: populate session
        user = rows[0]
        session["user_id"] = user["id"]
        session["username"] = user["username"]
        session["role"] = user["role"]

        flash("Logged in successfully.")
        return redirect(url_for("index"))

    # GET: show login form
    return render_template("login.html")


@app.route("/logout")
def logout():
    """
    Log out route: clears the session and sends user back home.
    """
    session.clear()
    flash("You have been logged out.")
    return redirect(url_for("index"))


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    """
    Candidate-only route to upload a resume.

    - Requires role == "candidate".
    - Accepts exactly one PDF file.
    - Requires a job_field from the dropdown.
    - Extracts text using PyPDF2.
    - Does a simple duplicate check (same user, text, and field).
    - Inserts resume and redirects candidate to /feedback.
    """
    # Only candidates should upload resumes
    if session.get("role") != "candidate":
        flash("Only candidates can upload resumes.")
        return redirect(url_for("index"))

    if request.method == "POST":
        # Ensure the form actually includes a file
        if "resume" not in request.files:
            flash("No file part in the request.")
            return redirect(url_for("upload"))

        resume_file = request.files["resume"]

        # User clicked submit without choosing a file
        if resume_file.filename == "":
            flash("Please choose a file before submitting.")
            return redirect(url_for("upload"))

        # Job field from dropdown – required
        job_field = request.form.get("job_field")
        if not job_field:
            flash("Please select what kind of job you're targeting.")
            return redirect(url_for("upload"))

        # Reject non-PDF files
        if not allowed_file(resume_file.filename):
            flash("Only PDF files are allowed.")
            return redirect(url_for("upload"))

        # Save file to uploads/ with a secure filename
        filename = secure_filename(resume_file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        resume_file.save(filepath)

        # Extract text from the PDF
        try:
            reader = PdfReader(filepath)
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            full_text = "\n".join(text_parts).strip()
        except Exception as e:
            print("PDF parse error:", e)
            full_text = ""

        # Fallback if for some reason nothing can be extracted
        if not full_text:
            full_text = "(No text could be extracted from this PDF.)"

        user_id = session["user_id"]

        # Duplicate check:
        # Same user, same extracted text, and same job_field → treat as duplicate.
        existing = db.execute(
            "SELECT id FROM resumes WHERE user_id = ? AND text = ? AND job_field = ?",
            user_id,
            full_text,
            job_field,
        )
        if existing:
            flash("You already uploaded this resume for this job field. Try uploading an updated version instead.")
            return redirect(url_for("feedback"))

        # Insert new resume with job_field
        db.execute(
            "INSERT INTO resumes (user_id, filename, text, job_field) VALUES (?, ?, ?, ?)",
            user_id,
            filename,
            full_text,
            job_field,
        )

        flash("Resume uploaded!")
        return redirect(url_for("feedback"))

    # GET: show upload form
    return render_template("upload.html")


@app.route("/swipe", methods=["GET", "POST"])
@login_required
def swipe():
    """
    Recruiter-only route to swipe on resumes.

    - Requires role == "recruiter".
    - Uses a session-level job_field (set on /swipe/select).
    - Builds a randomized order of resume IDs per field and stores it in session.
    - On each POST, records a swipe and possibly retrains the ML model.
    """
    # Only recruiters should swipe resumes
    if session.get("role") != "recruiter":
        flash("Only recruiters can swipe resumes.")
        return redirect(url_for("index"))

    # Make sure a job field has been chosen on /swipe/select
    job_field = session.get("swipe_field")
    if not job_field:
        flash("Please choose a job field first.")
        return redirect(url_for("swipe_select"))

    # If we don't yet have a randomized order for this session, create one
    if "swipe_order" not in session:
        rows = db.execute(
            "SELECT id FROM resumes WHERE job_field = ?",
            job_field,
        )
        if not rows:
            flash("No resumes available yet for this field.")
            return redirect(url_for("swipe_select"))

        # Build a list of resume IDs and shuffle them
        order = [row["id"] for row in rows]
        random.shuffle(order)
        session["swipe_order"] = order
        session["swipe_index"] = 0

    order = session["swipe_order"]
    index = session.get("swipe_index", 0)

    if request.method == "POST":
        # Button choice from the form
        choice = request.form.get("choice")      # "pass" or "like"
        resume_id = int(request.form.get("resume_id"))

        # Map button choice to label: 1 = Mash / Like, 0 = Pass
        label = 1 if choice in ("mash", "like") else 0

        recruiter_id = session["user_id"]

        # Prevent duplicate swipes for the same resume by the same recruiter
        existing_swipe = db.execute(
            "SELECT id FROM swipes WHERE resume_id = ? AND user_id = ?",
            resume_id,
            recruiter_id,
        )
        if not existing_swipe:
            # Record this swipe
            db.execute(
                "INSERT INTO swipes (resume_id, user_id, label) VALUES (?, ?, ?)",
                resume_id,
                recruiter_id,
                label,
            )

            # Dynamic retraining:
            # Every 10 swipes *in this field*, retrain that field's model
            count = db.execute(
                """
                SELECT COUNT(*) AS n
                FROM swipes
                JOIN resumes ON swipes.resume_id = resumes.id
                WHERE resumes.job_field = ?
                """,
                job_field,
            )[0]["n"]

            if count % 10 == 0:
                used = train_model(db, job_field)
                print(
                    f"[ML] Retrained model for field '{job_field}' on {used} swipes (field total: {count})."
                )

        # Move to next resume in the randomized order
        index += 1
        session["swipe_index"] = index

    # If we've gone through all resumes in this order, show the "done" page
    if index >= len(order):
        return render_template("swipe_done.html")

    # Get the current resume by ID from the randomized order
    current_resume_id = order[index]
    resume = db.execute(
        """
        SELECT resumes.id,
               resumes.filename,
               resumes.job_field,
               users.first_name,
               users.last_name
        FROM resumes
        JOIN users ON resumes.user_id = users.id
        WHERE resumes.id = ?
        """,
        current_resume_id,
    )[0]

    # Render swipe.html with current resume info
    return render_template("swipe.html", resume=resume)


@app.route("/swipe/select", methods=["GET", "POST"])
@login_required
def swipe_select():
    """
    Page where recruiters pick which job field they want to swipe on.

    - Stores chosen field in session["swipe_field"].
    - Resets any existing swipe order/index for that recruiter session.
    """
    if session.get("role") != "recruiter":
        flash("Only recruiters can swipe resumes.")
        return redirect(url_for("index"))

    # List of job_field options shown in the dropdown
    fields = [
        ("software", "Software / Engineering"),
        ("data", "Data / Analytics"),
        ("finance", "Finance"),
        ("consulting", "Consulting"),
        ("marketing", "Marketing"),
        ("product", "Product Management"),
        ("general", "General / Other"),
    ]

    if request.method == "POST":
        chosen = request.form.get("job_field")
        if not chosen:
            flash("Please select a field.")
            return redirect(url_for("swipe_select"))

        # Store chosen field in session
        session["swipe_field"] = chosen

        # Reset swipe order + index for this newly selected field
        session.pop("swipe_order", None)
        session.pop("swipe_index", None)
        return redirect(url_for("swipe"))

    # GET: show dropdown form
    return render_template("swipe_select.html", fields=fields)


@app.route("/swipe/reset", methods=["POST"])
@login_required
def swipe_reset():
    """
    Developer/admin helper route to reset swipe order for the current recruiter.

    - Clears swipe_index and swipe_order from the session.
    - Next visit to /swipe will rebuild a fresh randomized order.
    """
    if session.get("role") != "recruiter":
        flash("Only recruiters can reset swipes.")
        return redirect(url_for("index"))

    session.pop("swipe_index", None)
    session.pop("swipe_order", None)
    return redirect(url_for("swipe"))


@app.route("/uploads/<path:filename>")
@login_required
def uploaded_file(filename):
    """
    Protected route to serve uploaded PDF files.

    Rules:
      - File must exist in resumes table.
      - Recruiters can view any resume.
      - Candidates can only view resumes they uploaded themselves.
    """
    # Make sure this filename exists in the resumes table
    rows = db.execute("SELECT id, user_id FROM resumes WHERE filename = ? LIMIT 1", filename)
    if not rows:
        flash("File not found.")
        return redirect(url_for("index"))

    owner_id = rows[0]["user_id"]
    current_id = session.get("user_id")
    role = session.get("role")

    # Candidates can only view their own resume; recruiters can view all
    if role != "recruiter" and current_id != owner_id:
        flash("You do not have permission to view this resume.")
        return redirect(url_for("index"))

    # Actually serve the file from the uploads directory
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/feedback")
@login_required
def feedback():
    """
    Candidate-only route that shows AI feedback for their most recent resume.

    - Finds the latest resume for this candidate.
    - Uses score_text() to get a probability for "Mash" for that job_field.
    - Converts probability to a percentage and bucketed feedback message.
    - Renders feedback.html with:
        - resume preview
        - score percentage (if any)
        - feedback text
    """
    # Only candidates should see feedback on their own resume
    if session.get("role") != "candidate":
        flash("Only candidates can view resume feedback.")
        return redirect(url_for("index"))

    user_id = session["user_id"]

    # Get this candidate's most recently uploaded resume (latest uploaded_at, then id)
    rows = db.execute(
        """
        SELECT id, user_id, filename, text, job_field, uploaded_at
        FROM resumes
        WHERE user_id = ?
        ORDER BY uploaded_at DESC, id DESC
        LIMIT 1
        """,
        user_id,
    )

    if not rows:
        flash("Upload a resume first to get feedback.")
        return redirect(url_for("upload"))

    resume = rows[0]

    # Run ML model to score this resume text for its job_field
    raw_score = score_text(resume["text"], resume["job_field"])
    score_pct = None
    feedback_message = None

    if raw_score is None:
        # No trained model yet for this field, so we can't give a numeric score
        feedback_message = (
            "Our AI doesn't have enough recruiter swipe data yet to give "
            "reliable feedback. Ask recruiters to swipe more resumes, then try again."
        )
    else:
        # Convert probability (0–1) to percentage
        score_pct = round(raw_score * 100)

        # Basic bucketed feedback based on score range
        if score_pct >= 80:
            feedback_message = (
                "Your resume currently scores very well. Recruiters whose swipes "
                "trained this model tend to like resumes like yours."
            )
        elif score_pct >= 50:
            feedback_message = (
                "Your resume is in a solid range, but there’s room to improve. "
                "Tighten bullets, quantify impact, and make sure your strongest "
                "experiences and skills are front and center."
            )
        else:
            feedback_message = (
                "Right now, the model predicts your resume might not perform as well "
                "with recruiters. Focus on clearer structure, stronger action verbs, "
                "and concrete numbers that show what you actually achieved."
            )

    # Render template with both the resume and the AI outputs
    return render_template(
        "feedback.html",
        resume=resume,
        score_pct=score_pct,          # integer percentage, e.g. 78
        feedback_message=feedback_message,
    )


if __name__ == "__main__":
    # Enable debug=True for development so changes auto-reload
    app.run(debug=True)
