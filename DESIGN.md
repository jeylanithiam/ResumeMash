OVERVIEW:

ResumeMash is built as a Flask + SQLite web app with a small machine learning component using scikit-learn. At a high level, I wanted to accomplish three main things: user accounts with roles, a resume upload + working swipe component, and an AI model that can score resumes based on the swipes. For the project, Flask handles the routing, templating, and sessions, SQLite handles the data, and scikit-learn provides the model that scores the resumes.

Everything runs from app.py. The models are stored as pickle files in a models/ directory. Resumes are stored as actual PDF files in uploads/, and their extracted text lives in the database. HTML templates are in templates/ and are styled by a CSS sheet at static/css/styles.css.

DATABASE DESIGN:

I used a single SQLite database, resumemash.db, with three tables: users, resumes, and swipes. I chose SQLite because it’s what we used in class and it’s the default for small Flask apps.

The users table stores:
	•	id – primary key
	•	username
	•	hash – password hash
	•	role – "candidate" or "recruiter"
	•	first_name
	•	last_name
	•	email
	•	phone

The resumes table stores:
	•	id – primary key
	•	user_id – foreign key pointing to users.id
	•	filename – the PDF filename stored in uploads/
	•	text – extracted from the PDF using PyPDF2
	•	job_field – the target field (software, data, finance, etc.)
	•	uploaded_at – timestamp

I decided to store the PDF text directly in the database since it is much easier than having to re-parse the PDF each time I want to run my logistic regression model.

The swipes table stores:
	•	id – primary key
	•	resume_id – which resume was swiped
	•	user_id – which recruiter swiped
	•	label – 1 for Mash and 0 for Pass
	•	created_at – timestamp

FLASK DESIGN/ROUTING:

All routing and application logic is inside app.py. I used Flask’s session support and a small login_required decorator like we did in Finance. The key design choice here is role-based access instead of having to build a whole separate app for recruiters and candidates.

The main routes are:
	•	/ – the landing page.
	•	/register – handles the GET and POST requests for account creation. It validates fields, checks for duplicate usernames, hashes passwords, and sets up the session.
	•	/login and /logout – log the user in and out.
	•	/upload – candidate-only route for uploading resumes. It validates the file, extracts text, and inserts into the resumes table with the chosen job_field.
	•	/feedback – candidate-only view that pulls the most recent resume for the logged-in candidate, calls the ML scoring function, and renders the result.
	•	/swipe/select – recruiter-only view where the recruiter chooses a job field to swipe. It also sets session["swipe_field"].
	•	/swipe/reset – recruiter-only helper route that resets swipe_order and swipe_index in the session so the recruiter can start over.
	•	/uploads/<path:filename> – this route checks the database first to ensure the filename corresponds to a real resume, then checks that either the current user is the resume’s owner or the current user is a recruiter. If so, it returns the file from the uploads/ directory.

I decided to store the swipe order and index in the session to avoid a more complex server-side solution. It also gave me an easy way to randomize the swipe sequence: I query all resume IDs for that job field, shuffle them with Python’s random.shuffle, and then just walk that list until I run out. When I hit the end, I render a separate swipe_done.html.

TEMPLATE AND STYLING CHOICES:

All pages inherit from layout.html, which holds the <head>, the Bootstrap CDN links, the custom CSS, the navbar, the flash messages, the main content wrapper, and a sticky footer. I wanted the UI to feel like a polished product, so I spent time on a custom dark theme in static/css/styles.css. I defined a number of CSS variables (for example, --rm-bg, --rm-surface, --rm-pass, --rm-fail) to control the color palette, and then built reusable classes like .rm-card, .rm-page-header, .rm-swipe-actions, and .rm-footer.

One specific design choice was how to embed the PDFs. Using <embed> produced a lot of UI chrome (sidebars, toolbars) that I didn’t like for a clean swipe experience. I switched to an <iframe> that points to the PDF URL with query parameters (#toolbar=0&navpanes=0&scrollbar=1) to hide most of the extra viewer UI and keep the focus on the resume itself. I also wrapped the iframe in a styled container with a subtle border and rounded corners so it visually matches the rest of the card.

For buttons, I used Bootstrap’s classes as a base but added custom .btn-pass and .btn-fail classes for the swipe actions. These are bright green and red with “soft” backgrounds and a heavier font weight. The idea was to signal clearly that these are primary actions but still keep them consistent with the dark background. I also implemented an auto-dismissing flash message system with a small bit of JavaScript inside layout.html: a setTimeout fades out the alert div after a few seconds and then removes it from the DOM entirely. This keeps notifications visible long enough to be useful but stops them from cluttering the UI.

The layout is flex-based at the body level: body is set as a column flex container, .main-content is given flex: 1 so it stretches, and the footer has flex-shrink: 0. This combination ensures the footer stays at the bottom of the viewport on short pages but moves down naturally as content grows on long pages. That’s a deliberate UX choice to avoid “floating” footers in the middle of the page.

MACHINE LEARNING DESIGN:

For the AI piece, I wanted something that felt real but was still manageable. I chose scikit-learn’s LogisticRegression model with a TfidfVectorizer on top of plain resume text. Conceptually, each training example is one entry in swipes joined with its corresponding row in resumes, with:
	•	X = TF-IDF features of the resume text
	•	y = 1 if a recruiter clicked on “Mash”, 0 if they clicked “Pass”

I split the model by job field instead of having a single global classifier. So for each distinct job_field (for example, "finance", "software"), I have a separate model file in the models/ directory named model_<field>.pkl. The train_model(db, job_field) function queries all swipes for that field, extracts the texts and labels, and trains logistic regression on the TF-IDF features. It also has a crucial guard: if there aren’t at least two classes present in the labels (for example, all swipes so far are “Mash” and none are “Pass”), the function prints a message and refuses to train. This avoids scikit-learn’s “only one class present” error and also avoids generating misleading models based on zero variation.

I also chose to retrain in batches of 10 swipes per field, not on every single swipe. If I retrained on every swipe, I’d be constantly re-evaluating text and hitting scikit-learn on almost every recruiter click, which is overkill for the size of this project and might feel laggy. On the other hand, never retraining would defeat the point of a “learning” system. Batch retraining with a simple threshold (if count % 10 == 0) felt like the right compromise.

The score_text(text, job_field) function is the other half of the ML story. It looks up the model bundle for the given field (if it exists), transforms the given text with the saved vectorizer, and returns the probability of the “Mash” class from predict_proba. The Flask app then converts that probability into a percentage and buckets it into a few qualitative feedback messages. I consciously kept the feedback heuristic simple: the “AI” is just logistic regression plus some hand-written thresholds (>= 80, >= 50, else).

BULK IMPORT TOOL:

I added a dedicated script (bulk_import_resumes.py) to handle the situation where I want to demo the app with many resumes but don’t want to upload them one by one through the UI.

The script takes a ZIP file (bulk_resumes.zip) that contains PDFs, unpacks it into a temporary folder, and then loops over each PDF. For each file, it:
	1.	Uses PyPDF2 to extract all text.
	2.	Extracts the metadata title if available.
	3.	Tries to guess the candidate’s first and last name. First, it looks at the first non-empty line of the resume text and splits it into tokens, using that as a name if it seems reasonable. If that fails, it falls back to a heuristic based on the filename.
	4.	Calls a guess_job_field helper that looks for field-specific keywords in the combined string of resume text, filename, and PDF title. Depending on what it finds (for example, “investment banking,” “product manager”), it assigns the resume to one of the job fields (software, data, finance, consulting, marketing, product, or general).
	5.	Creates a new users row for each resume with role set to "candidate", a placeholder password, and an auto-generated username based on the filename.
	6.	Copies the PDF into the uploads/ directory.
	7.	Inserts a new row into resumes tying the text and inferred job_field to the new user.

TRADEOFFS AND LIMITATIONS:

I think overall I reached my goals, but admittedly my AI model is pretty simple. It is only a logistic regression model that looks for similarities between resumes. As we know, a resume isn’t “good” just because it has the same keywords as other resumes that recruiters like. That’s part of it, but the world is more complicated than that. I think the model would be better if I could incorporate something like ChatGPT into it to provide more qualitative, content-aware feedback on the resume in addition to the current similarity-based scoring.
