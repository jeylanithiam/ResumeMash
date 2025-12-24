import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Directory to store per-field models on disk.
# Each job_field gets its own model file inside this folder.
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def _model_path(job_field: str) -> str:
    """
    Build a safe path for the model file for a given job_field.

    Example:
        job_field = "software"
        -> "models/model_software.pkl"
    """
    # Replace characters that could break filenames.
    safe = job_field.replace("/", "_").replace(" ", "_")
    return os.path.join(MODEL_DIR, f"model_{safe}.pkl")


def train_model(db, job_field):
    """
    Train a logistic regression model on all swipes for a given job_field.

    Inputs:
        db        - CS50 SQL database connection
        job_field - string like "software", "finance", etc.

    Training data:
        X = resume text (from resumes.text)
        y = swipe label (from swipes.label, 0 = Pass, 1 = Mash)

    Side effect:
        Saves vectorizer + model as a single pickle file:
            MODEL_DIR/model_<job_field>.pkl

    Returns:
        number of swipe samples used for training (int)
    """
    # Pull all swipes for this field, joined with the resume text.
    rows = db.execute(
        """
        SELECT resumes.text AS text, swipes.label AS label
        FROM swipes
        JOIN resumes ON swipes.resume_id = resumes.id
        WHERE resumes.job_field = ?
        """,
        job_field,
    )

    if not rows:
        # Nothing to train on for this field yet.
        return 0

    # Split query results into features and labels.
    texts = [row["text"] for row in rows]
    labels = [row["label"] for row in rows]

    # Guard: need at least 2 different classes (both 0 and 1) or LogisticRegression
    # will crash with "needs samples of at least 2 classes".
    if len(set(labels)) < 2:
        print(
            f"[ML] Not training model for '{job_field}' yet: "
            f"only one class present in {len(labels)} samples."
        )
        # If a model already exists on disk, we keep it.
        # We just skip updating it until we have more balanced data.
        return 0

    # Convert text into numeric features using TF-IDF.
    # - stop_words="english" removes common English words
    # - max_features limits vocabulary size to keep model small/simple
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
    )
    X = vectorizer.fit_transform(texts)

    # Logistic regression classifier.
    # - max_iter increased so training converges on small datasets.
    # - class_weight="balanced" helps when Pass vs Mash counts are imbalanced.
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )
    model.fit(X, labels)

    # Bundle both the vectorizer and the trained model together so we can
    # reload them later with a single file read.
    bundle = {"vectorizer": vectorizer, "model": model}
    path = _model_path(job_field)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)

    # Return how many training samples we had for this field.
    return len(texts)


def _load_model_bundle(job_field):
    """
    Load the saved vectorizer + model for a given job_field, if it exists.

    Returns:
        dict with keys "vectorizer" and "model", or None if no file yet.
    """
    path = _model_path(job_field)
    if not os.path.exists(path):
        # No model trained for this field yet.
        return None

    with open(path, "rb") as f:
        return pickle.load(f)


def score_text(text, job_field):
    """
    Given resume text and a job_field, run the trained model (if any)
    and return the probability that the model predicts "Mash" (label 1).

    Inputs:
        text      - plain text of a single resume
        job_field - which per-field model to use, e.g. "software"

    Returns:
        float in [0, 1] = P(label == 1 | text, job_field),
        or None if no trained model exists for this field yet.
    """
    bundle = _load_model_bundle(job_field)
    if bundle is None:
        # No model file on disk for this field.
        return None

    vectorizer = bundle["vectorizer"]
    model = bundle["model"]

    # Transform the single resume string into the same TF-IDF space
    # that the model was trained on.
    X = vectorizer.transform([text])

    # predict_proba returns an array of [P(class 0), P(class 1)].
    # Index 1 is the probability of "Mash".
    prob = model.predict_proba(X)[0][1]
    return prob
