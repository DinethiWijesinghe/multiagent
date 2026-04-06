"""
UniAssist API Server v7.1 — Windows-Fixed Edition
===================================================
FIXES vs v7.0:
  ✓ FIXED   Tesseract "not found" on Windows
            → Auto-detects common Windows install paths
            → Falls back to EasyOCR if Tesseract unavailable
  ✓ ADDED   _configure_tesseract() called at startup
  ✓ ADDED   OCR_ENGINE auto-selection logic:
              1. Try Tesseract (fast, low RAM)
              2. If not found → try EasyOCR (CPU mode)
              3. If neither → clear error message with install guide
  ✓ KEPT    All ML classifier, field extractors, FastAPI endpoints

WINDOWS INSTALL (choose one):
  Option A — Tesseract (recommended, fast):
    Download: https://github.com/UB-Mannheim/tesseract/wiki
    Install to: C:\\Program Files\\Tesseract-OCR\\tesseract.exe
    Then restart this server.

  Option B — EasyOCR (no extra install):
    Set USE_EASYOCR = True below.
    First run downloads ~100 MB model weights.

Run:
  uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

import os
import re
import sys
import json
import pickle
import random
import secrets
import time
from datetime import datetime, timezone
from hashlib import pbkdf2_hmac
from pathlib import Path
from uuid import uuid4

import numpy as np
import cv2
from PIL import Image

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

USE_EASYOCR          = False   # Set True to force EasyOCR (overrides Tesseract)
MODEL_PATH           = "uniassist_classifier.pkl"
CONFIDENCE_THRESHOLD = 0.40
MAX_IMAGE_DIM        = 1000    # px cap — prevents RAM freeze

# ─────────────────────────────────────────────────────────────────────────────
# TESSERACT WINDOWS PATH AUTO-DETECTION  ← NEW FIX
# ─────────────────────────────────────────────────────────────────────────────

# Common Windows install locations for Tesseract
_WINDOWS_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Tesseract-OCR\tesseract.exe",
    r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(
        os.environ.get("USERNAME", "")
    ),
]

_OCR_ENGINE = None  # will be set to "tesseract", "easyocr", or None after startup


def _configure_tesseract() -> bool:
    """
    Try to locate and configure Tesseract on Windows (or any OS).
    Returns True if Tesseract is available and configured.
    """
    try:
        import pytesseract

        # 1. Try current PATH first (works on Linux/macOS and correctly set Windows)
        try:
            ver = pytesseract.get_tesseract_version()
            print(f"[Tesseract] Found in PATH — version {ver}")
            return True
        except Exception:
            pass

        # 2. Try known Windows install paths
        if sys.platform == "win32":
            for path in _WINDOWS_TESSERACT_PATHS:
                if os.path.isfile(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    try:
                        ver = pytesseract.get_tesseract_version()
                        print(f"[Tesseract] Found at {path} — version {ver}")
                        return True
                    except Exception:
                        continue

        # 3. Try environment variable TESSERACT_CMD
        env_path = os.environ.get("TESSERACT_CMD", "")
        if env_path and os.path.isfile(env_path):
            pytesseract.pytesseract.tesseract_cmd = env_path
            try:
                ver = pytesseract.get_tesseract_version()
                print(f"[Tesseract] Found via env TESSERACT_CMD — version {ver}")
                return True
            except Exception:
                pass

        print("[Tesseract] NOT FOUND on this system.")
        _print_tesseract_install_guide()
        return False

    except ImportError:
        print("[Tesseract] pytesseract package not installed.")
        print("  Run: pip install pytesseract")
        return False


def _print_tesseract_install_guide():
    print()
    print("=" * 60)
    print("  TESSERACT NOT FOUND — Install Guide")
    print("=" * 60)
    print("  Windows:")
    print("    1. Download from:")
    print("       https://github.com/UB-Mannheim/tesseract/wiki")
    print("    2. Run installer → install to default location")
    print("       (C:\\Program Files\\Tesseract-OCR\\)")
    print("    3. Restart this server")
    print()
    print("  Alternative: Set environment variable before running:")
    print("    set TESSERACT_CMD=C:\\path\\to\\tesseract.exe")
    print()
    print("  OR: Set USE_EASYOCR = True in api_server.py")
    print("      to use EasyOCR instead (no extra install)")
    print("=" * 60)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING DATA  (225 raw → 1,125 augmented)
# ─────────────────────────────────────────────────────────────────────────────

TRAINING_DATA = {
    "alevel": [
        "sri lanka advanced level examination results 2023",
        "department of examinations general certificate of education advanced level",
        "subject combined mathematics physics chemistry biology",
        "grade A B C S F result slip candidate",
        "z score district rank island rank university admission",
        "gce al examination colombo district 2022",
        "candidate index number al results stream science",
        "stream science mathematics technology arts al result",
        "al result mathematics A physics B chemistry C grade",
        "advanced level examination august 2024 results certified",
        "certificate of qualification general certificate education",
        "combined maths physics biology grade pass al commissioner",
        "gce advanced level result sheet candidate number index",
        "examination index number subject grade obtained al",
        "aggregate score stream district island rank al university",
        "commissioner general of examinations sri lanka certificate",
        "z score 1.8542 district rank island rank al result",
        "physics chemistry combined mathematics grade pass al",
        "date of issue certificate of qualification al department",
        "arts stream sinhala political science general english al",
        "common general test arts stream al result grade",
        "logic scientific method sinhala political science arts al",
        "subject stream arts science mathematics technology al gce",
        "island rank district rank z score gce al result",
        "general certificate education advanced level result sri lanka",
    ],
    "bachelor": [
        "bachelor of science degree university of colombo awarded",
        "bachelor of engineering faculty of engineering university",
        "undergraduate degree certificate university of moratuwa awarded",
        "bachelor of arts honours degree university of peradeniya",
        "degree awarded bsc engineering information technology university",
        "university of kelaniya bachelor degree 2020 convocation",
        "faculty of science bachelor degree convocation ceremony",
        "awarded degree first class second class bachelor honours",
        "bachelor of business administration bba university awarded",
        "undergraduate programme four year bachelor degree faculty",
        "university of sri jayewardenepura bsc degree awarded",
        "bachelor degree gpa cumulative grade point average",
        "degree awarded first class honours upper second lower second pass",
        "university convocation ceremony degree certificate bachelor",
        "bachelor of medicine mbbs medical degree university faculty",
        "bachelor of laws llb university degree certificate faculty",
        "faculty engineering architecture bachelor degree award",
        "information technology computing bachelor degree university awarded",
        "south eastern university bachelor degree programme faculty",
        "wayamba university bachelor degree agriculture science awarded",
        "uva wellassa university bachelor degree award ceremony convocation",
        "rajarata university bachelor of science award faculty",
        "open university of sri lanka bachelor degree programme",
        "sabaragamuwa university bachelor degree certificate awarded",
        "eastern university vavuniya campus bachelor awarded ceremony",
    ],
    "master": [
        "master of science msc university of colombo postgraduate",
        "master of business administration mba postgraduate degree",
        "master of engineering postgraduate degree university faculty",
        "msc information technology university of moratuwa postgraduate",
        "postgraduate degree master programme faculty university",
        "master of arts ma postgraduate university of kelaniya awarded",
        "mba degree awarded university of sri jayewardenepura postgraduate",
        "master of philosophy mphil research degree university",
        "master degree gpa grade point average postgraduate distinction",
        "msc computer science postgraduate programme university awarded",
        "master degree awarded distinction merit pass university",
        "master of education med postgraduate degree university",
        "msc data science analytics postgraduate university awarded",
        "master degree thesis research dissertation university postgraduate",
        "postgraduate master degree certificate awarded ceremony",
        "mba executive programme part time master degree university",
        "master of nursing science postgraduate degree university awarded",
        "msc electrical engineering postgraduate degree university faculty",
        "master of public administration mpa postgraduate degree",
        "master of finance mfin postgraduate degree awarded university",
        "postgraduate studies master programme two year full time",
        "master of social sciences mssci postgraduate awarded",
        "master degree programme colombo moratuwa peradeniya university",
        "postgraduate master awarded distinction merit pass university",
        "postgraduate diploma master degree university peradeniya",
    ],
    "diploma": [
        "diploma in information technology DPIT certificate awarded",
        "national diploma engineering NDT technical college",
        "diploma programme institute of technology certificate",
        "diploma certificate awarded successfully completed programme",
        "NIBM national institute business management diploma certificate",
        "diploma in accountancy CIMA AAT foundation certificate",
        "higher national diploma HND computing business awarded",
        "diploma in teacher education college of education certificate",
        "professional diploma marketing management certificate awarded",
        "technical college vocational training diploma award certificate",
        "diploma nursing paramedical institute healthcare awarded",
        "national vocational qualification NVQ diploma certificate",
        "diploma programme one year two year part time certificate",
        "institute chartered accountants foundation diploma certificate",
        "diploma journalism mass communication media certificate",
        "ESOFT SLIIT NSBM diploma information technology awarded",
        "diploma hotel management tourism hospitality certificate",
        "VTA vocational training authority diploma certificate awarded",
        "NAITA national apprentice industrial training diploma",
        "BIT external degree diploma information technology certificate",
        "diploma english language proficiency certificate awarded",
        "diploma graphic design multimedia arts certificate awarded",
        "diploma electrical electronics engineering technical college",
        "diploma banking finance institute diploma award certificate",
        "professional certificate diploma awarded completed programme",
    ],
    "ielts": [
        "international english language testing system IELTS",
        "IELTS test report form overall band score 7.0",
        "listening reading writing speaking band score IELTS",
        "british council IDP IELTS certificate test taker result",
        "IELTS academic general training band 6.5 7.5 score",
        "test date IELTS report form candidate number result",
        "overall band 7 listening 8 reading 7 writing 6 speaking 7",
        "IELTS score valid two years british council report",
        "IDP education IELTS result test report form score",
        "candidate IELTS band score academic module result",
        "IELTS overall 6.0 6.5 7.0 7.5 8.0 band score",
        "test report form TRF number IELTS valid two years",
        "listening band reading band writing band speaking band IELTS",
        "IELTS academic test result score certificate 2023",
        "centre number IELTS examination colombo sri lanka result",
        "british council IELTS test report overall band score",
        "IELTS general training result form band score report",
        "nine band scale IELTS overall score report form",
        "IELTS certificate date of birth nationality result score",
        "idp ielts score report candidate registration number",
        "ielts band 5.5 6.0 6.5 7.0 7.5 8.0 result score",
        "test taker IELTS academic overall band 7.0 score result",
        "IELTS score listening 7.5 reading 8.0 writing 6.5 speaking",
        "IELTS report form valid two years from test date result",
        "british council IELTS test centre colombo kandy result score",
    ],
    "toefl": [
        "test of english as a foreign language TOEFL iBT score",
        "TOEFL score report total score reading listening writing",
        "TOEFL iBT score writing speaking reading listening total",
        "educational testing service ETS TOEFL score report",
        "TOEFL total score 100 110 90 80 score report ETS",
        "TOEFL iBT home edition test score report ETS result",
        "reading score listening score writing score speaking score TOEFL",
        "TOEFL score valid two years ETS report result",
        "TOEFL registration number test date score report ETS",
        "ETS TOEFL internet based test result total 105 score",
        "TOEFL score report reading 28 listening 27 speaking 24 writing 26",
        "test of english foreign language score certificate ETS result",
        "TOEFL iBT score report institutional code registration number",
        "ETS educational testing service score report TOEFL result",
        "TOEFL total 90 95 100 105 110 reading listening speaking writing",
        "TOEFL paper based test PBT score certificate result",
        "ETS TOEFL iBT test score date registration number result",
        "speaking writing reading listening section score TOEFL result",
        "TOEFL score report candidate name date birth nationality",
        "internet based TOEFL iBT official score report 2023 ETS",
        "ETS toefl total score 100 110 institution report result",
        "toefl reading 24 listening 26 speaking 22 writing 25 total 97",
        "TOEFL score valid two years test date ETS report result",
        "test english foreign language toefl ibt score report ets result",
        "TOEFL score send institution university score report ETS",
    ],
    "pte": [
        "pearson test of english PTE academic score report",
        "PTE academic overall score communicative skills enabling",
        "PTE score report listening reading writing speaking pearson",
        "pearson PTE academic score 65 70 75 80 result report",
        "PTE score enabling skills grammar oral fluency pronunciation",
        "pearson vue PTE academic score report valid two years",
        "PTE academic test result score report 2023 pearson",
        "PTE score reading 72 listening 68 speaking 70 writing 65",
        "pearson test english academic PTE overall 65 score",
        "PTE academic score report communicative skills enabling skills",
        "PTE score valid two years pearson academic result report",
        "pearson PTE academic score listening reading speaking writing",
        "PTE score report registration number test date result pearson",
        "PTE overall score 50 58 65 79 academic result pearson",
        "pearson academic PTE test score report enabling skills",
        "PTE score report spelling punctuation grammar vocabulary pearson",
        "pearson test english PTE score overall communicative result",
        "PTE academic score report pearson vue 2022 2023 result",
        "PTE test result reading writing listening speaking score report",
        "pearson PTE academic result score report valid two years",
        "PTE score report fluency pronunciation grammar spelling pearson",
        "PTE academic test overall 65 score pearson report result",
        "PTE score 58 65 72 79 overall academic result pearson",
        "pearson test of english academic score certificate report",
        "PTE academic pearson score report listening 68 reading 74",
    ],
    "passport": [
        "republic of sri lanka passport nationality travel document",
        "passport number date of issue expiry date holder",
        "surname given names date of birth nationality passport",
        "machine readable passport MRP travel document holder",
        "department of immigration emigration sri lanka passport",
        "passport photo page personal details nationality SRI LANKAN",
        "type P country code LKA passport number holder",
        "passport valid from date to expiry date travel",
        "place of birth nationality Sri Lankan passport holder",
        "biometric passport e-passport chip photograph signature",
        "MRZ machine readable zone passport holder nationality",
        "passport personal information page surname given name",
        "P LKA passport number nationality SRI LANKAN holder",
        "immigration emigration department passport certificate travel",
        "travel document passport republic sri lanka valid expiry",
        "passport bio data page photograph signature holder nationality",
        "P<<LKA surname given names nationality travel document",
        "national identity card number passport holder signature",
        "date issue colombo date expiry passport valid holder",
        "passport type regular diplomatic official emergency travel",
        "sri lankan passport photo page details biometric chip",
        "immigration department controller general passport issued",
        "passport holder signature photo date birth nationality LKA",
        "ordinary passport LKA nationality SRI LANKAN holder travel",
        "travel document passport number valid expiry date issued LKA",
    ],
    "financial": [
        "bank statement account number balance transaction history",
        "statement of account savings current account holder",
        "bank of ceylon commercial bank peoples bank statement",
        "account balance closing balance available balance LKR",
        "transaction history debit credit bank statement certified",
        "financial statement monthly bank account holder branch",
        "account summary opening balance closing balance LKR",
        "bank statement certified true copy bank stamp official",
        "fixed deposit certificate amount interest rate bank",
        "bank statement account number LKR USD balance available",
        "people bank hatton national bank NDB account statement",
        "statement period account holder name branch bank",
        "transaction date description debit credit balance bank",
        "bank certified statement financial proof funds LKR",
        "sampath bank nations trust bank account statement",
        "bank account statement branch code SWIFT IBAN certified",
        "financial capability letter bank manager certified stamp",
        "statement balance LKR 500000 1000000 2000000 account",
        "bank statement sponsor financial support letter certified",
        "account type savings account balance certificate bank",
        "bank statement authorized signatory official stamp certified",
        "financial statement proof of funds bank certified manager",
        "dfcc bank seylan bank account statement balance LKR",
        "bank statement three months six months transaction history",
        "account holder name address bank statement period certified",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def _augment(text):
    words = text.split()
    v = [text.lower()]
    v.append(" ".join(w.upper() if random.random() > 0.55 else w for w in words))
    if len(words) > 3:
        i = random.randint(0, len(words) - 2)
        sw = words[:]
        sw[i], sw[i + 1] = sw[i + 1], sw[i]
        v.append(" ".join(sw))
    else:
        v.append(text)
    if len(words) > 4:
        drop = set(random.sample(range(len(words)), min(2, len(words) // 5)))
        v.append(" ".join(w for i, w in enumerate(words) if i not in drop))
    else:
        v.append(text.lower())
    return v


def _build_corpus():
    texts, labels = [], []
    for label, samples in TRAINING_DATA.items():
        for s in samples:
            texts.append(s)
            labels.append(label)
            for a in _augment(s):
                texts.append(a)
                labels.append(label)
    return texts, labels


# ─────────────────────────────────────────────────────────────────────────────
# ML MODEL — TF-IDF + Naive Bayes
# ─────────────────────────────────────────────────────────────────────────────

def _train(verbose=True):
    texts, labels = _build_corpus()
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), max_features=8000,
            sublinear_tf=True, min_df=1,
            token_pattern=r"[a-zA-Z0-9]{2,}",
        )),
        ("nb", MultinomialNB(alpha=0.3)),
    ])
    if verbose:
        cv = cross_val_score(pipe, texts, labels, cv=5, scoring="accuracy")
        print(f"[ML] CV accuracy: {cv.mean():.1%} ± {cv.std():.1%}")
        print(f"[ML] Training {len(texts)} samples | {len(TRAINING_DATA)} classes ...")
    pipe.fit(texts, labels)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)
    if verbose:
        print(f"[ML] Saved → {MODEL_PATH}")
    return pipe


def _load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return _train()


# ─────────────────────────────────────────────────────────────────────────────
# OCR ERROR CORRECTOR
# ─────────────────────────────────────────────────────────────────────────────

_FIXES = {
    "examlnation": "examination", "certlficate": "certificate",
    "unlversity": "university", "passporl": "passport",
    "natlonality": "nationality", "llstening": "listening",
    "wrlting": "writing", "speaklng": "speaking", "readlng": "reading",
    "candldate": "candidate", "dlstrict": "district", "agregate": "aggregate",
    "zscore": "z score", "bandsoore": "band score", "balence": "balance",
    "statment": "statement", "certifled": "certified",
    "bachelar": "bachelor", "mastar": "master", "diplama": "diploma",
}


def _correct(text):
    return " ".join(_FIXES.get(w.lower(), w) for w in text.split())


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess(image_path):
    """Load, resize and threshold image. Returns grayscale numpy array."""
    img = cv2.imread(image_path)
    if img is None:
        pil = Image.open(image_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max_dim
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    elif max_dim < 600:
        scale = 800 / max_dim
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_LANCZOS4)

    h, w = img.shape[:2]
    if w > h:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray


# ─────────────────────────────────────────────────────────────────────────────
# OCR ENGINES
# ─────────────────────────────────────────────────────────────────────────────

_easyocr_reader = None


def _run_tesseract(image_path):
    """Run Tesseract OCR. Raises if unavailable."""
    import pytesseract
    gray = _preprocess(image_path)
    text = pytesseract.image_to_string(gray, config="--oem 1 --psm 3")
    data = pytesseract.image_to_data(
        gray, config="--oem 1 --psm 3",
        output_type=pytesseract.Output.DICT,
    )
    confs = [int(c) for c in data["conf"] if int(c) > 0]
    conf = (sum(confs) / len(confs) / 100) if confs else 0.0
    text = re.sub(r'[\u0D80-\u0DFF\u0B80-\u0BFF]+', '', text)
    return _correct(text.strip()), conf


def _run_easyocr_engine(image_path):
    """Run EasyOCR in CPU mode."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        print("[EasyOCR] Loading model (first time ~5 sec) ...")
        _easyocr_reader = easyocr.Reader(
            ["en"], gpu=False,
            model_storage_directory=os.path.join(
                os.path.expanduser("~"), ".EasyOCR", "model"),
            download_enabled=True,
            verbose=False,
        )
        print("[EasyOCR] Ready.")
    img = _preprocess(image_path)
    results = _easyocr_reader.readtext(
        cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
        detail=1, batch_size=1, workers=0, beamWidth=3,
    )
    words = [t for (_, t, c) in results if t.strip() and c > 0.25]
    confs = [c for (_, t, c) in results if t.strip() and c > 0.25]
    text = " ".join(words)
    conf = float(np.mean(confs)) if confs else 0.0
    text = re.sub(r'[\u0D80-\u0DFF\u0B80-\u0BFF]+', '', text)
    return _correct(text.strip()), conf


def _run_ocr(image_path):
    """
    Smart OCR dispatcher:
      - USE_EASYOCR=True  → EasyOCR always
      - _OCR_ENGINE set to "tesseract" → use Tesseract
      - _OCR_ENGINE set to "easyocr"   → use EasyOCR
      - Neither available → raise with clear instructions
    """
    global _OCR_ENGINE

    if USE_EASYOCR or _OCR_ENGINE == "easyocr":
        return _run_easyocr_engine(image_path)

    if _OCR_ENGINE == "tesseract":
        return _run_tesseract(image_path)

    # Engine not yet determined — try both
    try:
        result = _run_tesseract(image_path)
        _OCR_ENGINE = "tesseract"
        return result
    except Exception as te:
        print(f"[Tesseract] Failed: {te}")
        print("[OCR] Falling back to EasyOCR ...")
        try:
            result = _run_easyocr_engine(image_path)
            _OCR_ENGINE = "easyocr"
            return result
        except ImportError:
            raise RuntimeError(
                "No OCR engine available!\n\n"
                "To fix this, choose ONE of:\n"
                "  Option A (Tesseract — recommended):\n"
                "    1. Download: https://github.com/UB-Mannheim/tesseract/wiki\n"
                "    2. Install to C:\\Program Files\\Tesseract-OCR\\\n"
                "    3. Restart server\n\n"
                "  Option B (EasyOCR — no extra install):\n"
                "    pip install easyocr\n"
                "    Set USE_EASYOCR = True in api_server.py\n"
            )


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE SCORER
# ─────────────────────────────────────────────────────────────────────────────

_FW = {
    "alevel":    {"index_number": 0.25, "year": 0.1, "subjects": 0.25,
                  "grades": 0.2, "z_score": 0.2},
    "bachelor":  {"name": 0.2, "degree": 0.3, "university": 0.3, "year": 0.2},
    "master":    {"name": 0.2, "degree": 0.3, "university": 0.3, "year": 0.2},
    "diploma":   {"degree": 0.4, "year": 0.3, "university": 0.3},
    "ielts":     {"overall": 0.4, "listening": 0.15, "reading": 0.15,
                  "writing": 0.15, "speaking": 0.15},
    "toefl":     {"total": 0.4, "reading": 0.15, "listening": 0.15,
                  "speaking": 0.15, "writing": 0.15},
    "pte":       {"overall": 0.4, "listening": 0.15, "reading": 0.15,
                  "writing": 0.15, "speaking": 0.15},
    "passport":  {"passport_no": 0.3, "surname": 0.2, "given_names": 0.2, "expiry": 0.3},
    "financial": {"bank_name": 0.2, "account_no": 0.3, "closing_bal": 0.5},
}


def _score(doc_type, fields, ocr_conf, ml_conf):
    fw = _FW.get(doc_type, {})
    fs = sum(
        w for f, w in fw.items()
        if fields.get(f) and (fields[f] if isinstance(fields[f], str) else len(fields[f]) > 0)
    )
    return round(min(fs * 0.50 + ocr_conf * 0.25 + ml_conf * 0.25, 1.0), 3)


# ─────────────────────────────────────────────────────────────────────────────
# FIELD EXTRACTORS
# ─────────────────────────────────────────────────────────────────────────────

def _f(p, text):
    m = re.search(p, text, re.I)
    return m.group(1).strip() if m else None


def _fa(p, text):
    return re.findall(p, text, re.I)


def _extract(doc_type, text):
    t = text.lower()
    if doc_type == "alevel":
        return {
            "name":          _f(r"certify that\s+([A-Z][A-Z.\s]+)\n", text),
            "index_number":  _f(r"(?:Index\s*Number[:\s]+|Index\s*\n\s*Number[:\s]+)(\d{7})", text),
            "year":          _f(r"(20\d{2})", text),
            "subjects":      _fa(
                r"(Physics|Chemistry|Combined Mathematic?s?|Biology|Sinhala|"
                r"Political Science|General English|Logic.*?Method|"
                r"Common General Test|Economics|Geography|Accounting|ICT)", text
            ),
            "grades":        _fa(r"\|\s*([ABCSF])\s*\||([ABCSF])\s+Pass", text),
            "z_score":       _f(r"Z[\-\s]?Score[:\s]+([0-9.]+)", text),
            "district_rank": _f(r"District\s*Rank[:\s]+(\d+)", text),
            "island_rank":   _f(r"Island\s*Rank[:\s]+(\d+)", text),
            "stream":        _f(r"(?:Subject\s*)?Stream[:\s]+([A-Z]+)", text),
            "date_of_issue": _f(r"(?:Date of Issue|Date)[:\s]+(.+?\d{4})", text),
        }
    elif doc_type in ("bachelor", "master", "diploma"):
        return {
            "name":       _f(r"(?:certify that|awarded to|conferred upon)\s+([A-Z\s]+)", text),
            "degree":     _f(r"(bachelor|master|diploma)[^.\n]{0,60}", t),
            "university": _f(r"university of [\w\s]+|[\w\s]+ university", t),
            "year":       _f(r"(20\d{2})", text),
            "class":      _f(r"(first class|second class|upper second|lower second|"
                             r"distinction|merit|pass)", t),
            "gpa":        _f(r"gpa[\s:]+([0-9.]+)", t),
        }
    elif doc_type == "ielts":
        return {
            "overall":   _f(r"overall[\s\w]{0,20}?([0-9]\.[05])", t),
            "listening": _f(r"listening[\s:]+([0-9]\.[05])", t),
            "reading":   _f(r"reading[\s:]+([0-9]\.[05])", t),
            "writing":   _f(r"writing[\s:]+([0-9]\.[05])", t),
            "speaking":  _f(r"speaking[\s:]+([0-9]\.[05])", t),
            "test_date": _f(r"(\d{1,2}[\s/\-]\w+[\s/\-]20\d{2})", text),
            "trf":       _f(r"(?:trf|reference)[\s:]+([A-Z0-9\-]+)", t),
        }
    elif doc_type == "toefl":
        return {
            "total":     _f(r"total[\s:]+(\d{2,3})", t),
            "reading":   _f(r"reading[\s:]+(\d{1,2})", t),
            "listening": _f(r"listening[\s:]+(\d{1,2})", t),
            "speaking":  _f(r"speaking[\s:]+(\d{1,2})", t),
            "writing":   _f(r"writing[\s:]+(\d{1,2})", t),
            "test_date": _f(r"(\d{1,2}[\s/\-]\w+[\s/\-]20\d{2})", text),
        }
    elif doc_type == "pte":
        return {
            "overall":   _f(r"overall[\s:]+(\d{2,3})", t),
            "listening": _f(r"listening[\s:]+(\d{2,3})", t),
            "reading":   _f(r"reading[\s:]+(\d{2,3})", t),
            "writing":   _f(r"writing[\s:]+(\d{2,3})", t),
            "speaking":  _f(r"speaking[\s:]+(\d{2,3})", t),
            "test_date": _f(r"(\d{1,2}[\s/\-]\w+[\s/\-]20\d{2})", text),
        }
    elif doc_type == "passport":
        return {
            "surname":     _f(r"surname[\s:]+([A-Z]+)", text),
            "given_names": _f(r"given\s*names?[\s:]+([A-Z\s]+)", text),
            "passport_no": _f(r"\b([A-Z]\d{7})\b", text),
            "nationality": _f(r"nationality[\s:]+([A-Z\s]+)", text),
            "dob":         _f(r"(?:date of birth|dob)[\s:]+([0-9/\-\s\w]+)", t),
            "expiry":      _f(r"(?:date of expiry|expiry)[\s:]+([0-9/\-\s\w]+)", t),
            "mrz":         _f(r"(P<<LKA[A-Z<]+)", text),
        }
    elif doc_type == "financial":
        return {
            "bank_name":   _f(
                r"(bank of ceylon|peoples bank|commercial bank|hatton national|"
                r"sampath|nations trust|dfcc|seylan|ndb)", t
            ),
            "account_no":  _f(r"account\s*(?:number|no)[\s:]+([0-9\-]+)", t),
            "closing_bal": _f(
                r"(?:closing|available|current)\s*balance[\s:]*"
                r"(?:lkr|rs\.?)?[\s]*([0-9,]+(?:\.\d{2})?)", t
            ),
            "currency":    _f(r"\b(LKR|USD|GBP|AUD|SGD)\b", text),
            "period":      _f(r"(?:statement period|from)[\s:]+([0-9/\-\s\w]+)", t),
        }
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import tempfile
from typing import Any
from sqlalchemy import Column, DateTime, String, Text, create_engine, func
from sqlalchemy.orm import declarative_base, sessionmaker

try:
    from .core.agents.chatbot_agent import ChatbotAgent
    from .core.agents.eligibility_verification_agent import EligibilityVerificationAgent
    from .core.agents.financial_feasibility_agent import FinancialFeasibilityAgent
    from .core.agents.recommendation_agent import RecommendationAgent
    from .core.rag_system import RAGSystem
except Exception:
    try:
        from core.agents.chatbot_agent import ChatbotAgent
        from core.agents.eligibility_verification_agent import EligibilityVerificationAgent
        from core.agents.financial_feasibility_agent import FinancialFeasibilityAgent
        from core.agents.recommendation_agent import RecommendationAgent
        from core.rag_system import RAGSystem
    except Exception:
        ChatbotAgent = None
        EligibilityVerificationAgent = None
        FinancialFeasibilityAgent = None
        RecommendationAgent = None
        RAGSystem = None

Base = declarative_base()


class DBUser(Base):
    __tablename__ = "users"

    email = Column(String(255), primary_key=True, index=True)
    data = Column(Text, nullable=False, default="{}")


class DBSession(Base):
    __tablename__ = "sessions"

    token = Column(String(255), primary_key=True, index=True)
    data = Column(Text, nullable=False, default="{}")


class DBChatHistory(Base):
    __tablename__ = "chat_history"

    user_id = Column(String(255), primary_key=True, index=True)
    data = Column(Text, nullable=False, default="{}")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class DBUserState(Base):
    __tablename__ = "user_state"

    user_id = Column(String(255), primary_key=True, index=True)
    data = Column(Text, nullable=False, default="{}")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class DBDocumentUpload(Base):
    __tablename__ = "document_uploads"

    document_id = Column(String(64), primary_key=True, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    data = Column(Text, nullable=False, default="{}")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


def _normalize_database_url(raw_url: str) -> str:
    url = (raw_url or "").strip()
    if url.startswith("postgresql://"):
        converted = f"postgresql+psycopg://{url[len('postgresql://'):]}"
    elif url.startswith("postgres://"):
        converted = f"postgresql+psycopg://{url[len('postgres://'):]}"
    else:
        converted = url

    if converted.startswith("postgresql+psycopg://") and "sslmode=" not in converted:
        glue = "&" if "?" in converted else "?"
        converted = f"{converted}{glue}sslmode=require"
    return converted


def _env_true(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


DATABASE_URL = _normalize_database_url(
    os.environ.get("DATABASE_URL") or os.environ.get("NEON_DATABASE_URL") or ""
)
DB_STRICT_MODE = _env_true("DB_STRICT_MODE", False)

_db_engine = None
_SessionLocal = None
_db_backend = "json"
_chatbot_agent = None
_rag_provider = "none"

CHAT_HISTORY_DIR = Path(__file__).resolve().parent / "data" / "chat_history"
USERS_DIR = Path(__file__).resolve().parent / "data" / "users"
USER_STATE_DIR = Path(__file__).resolve().parent / "data" / "user_state"
DOCUMENTS_DIR = Path(__file__).resolve().parent / "data" / "documents"
UNIVERSITIES_DB_PATH = Path(__file__).resolve().parent / "data" / "databases" / "universities_database.json"
CHAT_HISTORY_LIMIT = 500
PASSWORD_HASH_ROUNDS = 200_000


class ChatMessage(BaseModel):
    id: str | None = None
    role: str
    text: str
    time: str
    metadata: dict[str, Any] | None = None


class ChatHistoryPayload(BaseModel):
    user_id: str
    messages: list[ChatMessage] = []
    agent_data: dict[str, Any] | None = None


class RegisterPayload(BaseModel):
    name: str
    email: str
    password: str


class LoginPayload(BaseModel):
    email: str
    password: str


class UserStatePayload(BaseModel):
    user_id: str
    state: dict[str, Any]


class ChatRespondPayload(BaseModel):
    user_message: str
    context: dict[str, Any] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_email(value: str) -> str:
    return (value or "").strip().lower()


def _safe_user_key(value: str) -> str:
    safe_value = re.sub(r"[^a-zA-Z0-9_.@-]", "_", (value or "").strip().lower())
    if not safe_value:
        raise HTTPException(status_code=400, detail="user_id is required")
    return safe_value


def _read_json(path: Path, default: Any):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _normalize_country_name(country: str | None) -> str | None:
    if not country:
        return None
    normalized = country.strip().lower()
    aliases = {
        "uk": "UK",
        "u.k.": "UK",
        "united kingdom": "UK",
        "gb": "UK",
        "great britain": "UK",
        "sg": "Singapore",
        "au": "Australia",
    }
    if normalized in aliases:
        return aliases[normalized]
    return country.strip()


def _load_universities_database() -> dict[str, Any]:
    data = _read_json(UNIVERSITIES_DB_PATH, {})
    if isinstance(data, dict):
        return data
    return {}


def _users_path() -> Path:
    return USERS_DIR / "users.json"


def _sessions_path() -> Path:
    return USERS_DIR / "sessions.json"


def _user_state_path(user_id: str) -> Path:
    return USER_STATE_DIR / f"{_safe_user_key(user_id)}.json"


def _user_documents_path(user_id: str) -> Path:
    return DOCUMENTS_DIR / _safe_user_key(user_id)


def _user_documents_index_path(user_id: str) -> Path:
    return _user_documents_path(user_id) / "index.json"


def _load_users() -> dict[str, dict[str, Any]]:
    if _SessionLocal:
        with _SessionLocal() as session:
            rows = session.query(DBUser).all()
            users: dict[str, dict[str, Any]] = {}
            for row in rows:
                payload = _read_json_text(row.data, {})
                users[row.email] = payload if isinstance(payload, dict) else {}
            return users
    data = _read_json(_users_path(), {})
    return data if isinstance(data, dict) else {}


def _save_users(users: dict[str, dict[str, Any]]) -> None:
    if _SessionLocal:
        with _SessionLocal() as session:
            for email, payload in users.items():
                row = session.get(DBUser, email)
                data = json.dumps(payload if isinstance(payload, dict) else {}, ensure_ascii=False)
                if row is None:
                    session.add(DBUser(email=email, data=data))
                else:
                    row.data = data
            session.commit()
        return
    _write_json(_users_path(), users)


def _load_sessions() -> dict[str, dict[str, Any]]:
    if _SessionLocal:
        with _SessionLocal() as session:
            rows = session.query(DBSession).all()
            sessions: dict[str, dict[str, Any]] = {}
            for row in rows:
                payload = _read_json_text(row.data, {})
                sessions[row.token] = payload if isinstance(payload, dict) else {}
            return sessions
    data = _read_json(_sessions_path(), {})
    return data if isinstance(data, dict) else {}


def _save_sessions(sessions: dict[str, dict[str, Any]]) -> None:
    if _SessionLocal:
        with _SessionLocal() as session:
            for token, payload in sessions.items():
                row = session.get(DBSession, token)
                data = json.dumps(payload if isinstance(payload, dict) else {}, ensure_ascii=False)
                if row is None:
                    session.add(DBSession(token=token, data=data))
                else:
                    row.data = data
            session.commit()
        return
    _write_json(_sessions_path(), sessions)


def _hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), PASSWORD_HASH_ROUNDS)
    return f"{salt}${digest.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    if "$" not in stored_hash:
        return False
    salt, expected_hash = stored_hash.split("$", 1)
    candidate_hash = _hash_password(password, salt).split("$", 1)[1]
    return secrets.compare_digest(candidate_hash, expected_hash)


def _issue_session_token(email: str) -> str:
    token = secrets.token_urlsafe(32)
    sessions = _load_sessions()
    sessions[token] = {
        "email": email,
        "created_at": _utc_now(),
    }
    _save_sessions(sessions)
    return token


def _authenticate_token(authorization: str | None, *, required: bool = True) -> str | None:
    if not authorization:
        if required:
            raise HTTPException(status_code=401, detail="Authentication required")
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    session = _load_sessions().get(token.strip())
    if not session or not session.get("email"):
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return _normalize_email(session["email"])


def _require_current_user(authorization: str | None = Header(default=None)) -> str:
    return _authenticate_token(authorization, required=True) or ""


def _ensure_user_access(user_id: str, current_user_email: str) -> str:
    normalized_user_id = _normalize_email(user_id)
    if not normalized_user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    if normalized_user_id != current_user_email:
        raise HTTPException(status_code=403, detail="You can only access your own data")
    return normalized_user_id


def _load_user_state_record(user_id: str) -> dict[str, Any]:
    if _SessionLocal:
        with _SessionLocal() as session:
            row = session.get(DBUserState, _safe_user_key(user_id))
            if row is None:
                return {}
            payload = _read_json_text(row.data, {})
            return payload if isinstance(payload, dict) else {}
    data = _read_json(_user_state_path(user_id), {})
    return data if isinstance(data, dict) else {}


def _save_user_state_record(user_id: str, state: dict[str, Any]) -> None:
    if _SessionLocal:
        safe_id = _safe_user_key(user_id)
        with _SessionLocal() as session:
            row = session.get(DBUserState, safe_id)
            data = json.dumps(state if isinstance(state, dict) else {}, ensure_ascii=False)
            if row is None:
                session.add(DBUserState(user_id=safe_id, data=data))
            else:
                row.data = data
            session.commit()
        return
    _write_json(_user_state_path(user_id), state)


def _load_document_records(user_id: str) -> list[dict[str, Any]]:
    if _SessionLocal:
        safe_id = _safe_user_key(user_id)
        with _SessionLocal() as session:
            rows = session.query(DBDocumentUpload).filter(DBDocumentUpload.user_id == safe_id).all()
            records: list[dict[str, Any]] = []
            for row in rows:
                payload = _read_json_text(row.data, {})
                if isinstance(payload, dict):
                    records.append(payload)
            return sorted(records, key=lambda item: item.get("stored_at", ""), reverse=True)
    data = _read_json(_user_documents_index_path(user_id), [])
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _save_document_records(user_id: str, records: list[dict[str, Any]]) -> None:
    if _SessionLocal:
        safe_id = _safe_user_key(user_id)
        clean_records = [item for item in records if isinstance(item, dict)]
        with _SessionLocal() as session:
            existing_rows = session.query(DBDocumentUpload).filter(DBDocumentUpload.user_id == safe_id).all()
            existing_ids = {row.document_id for row in existing_rows}
            incoming_ids = set()

            for record in clean_records:
                doc_id = record.get("document_id")
                if not doc_id:
                    continue
                incoming_ids.add(doc_id)
                data = json.dumps(record, ensure_ascii=False)
                row = session.get(DBDocumentUpload, doc_id)
                if row is None:
                    session.add(DBDocumentUpload(document_id=doc_id, user_id=safe_id, data=data))
                else:
                    row.user_id = safe_id
                    row.data = data

            for doc_id in existing_ids - incoming_ids:
                stale = session.get(DBDocumentUpload, doc_id)
                if stale is not None:
                    session.delete(stale)

            session.commit()
        return
    _write_json(_user_documents_index_path(user_id), records)


def _read_json_text(raw_text: str, default: Any):
    if not raw_text:
        return default
    try:
        return json.loads(raw_text)
    except Exception:
        return default


def _sanitize_filename(filename: str | None) -> str:
    candidate = Path(filename or "document").name.strip()
    candidate = re.sub(r"[^a-zA-Z0-9._ -]", "_", candidate)
    return candidate or "document"


def _store_user_document(
    user_id: str,
    *,
    filename: str | None,
    content_type: str | None,
    file_bytes: bytes,
    extracted_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    user_dir = _user_documents_path(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    document_id = uuid4().hex
    safe_name = _sanitize_filename(filename)
    stored_name = f"{document_id}_{safe_name}"
    file_path = user_dir / stored_name
    with open(file_path, "wb") as handle:
        handle.write(file_bytes)

    record = {
        "document_id": document_id,
        "filename": safe_name,
        "content_type": content_type or "application/octet-stream",
        "file_size": len(file_bytes),
        "stored_at": _utc_now(),
        "stored_name": stored_name,
    }
    if extracted_data:
        record["document_type"] = extracted_data.get("document_type")

    records = _load_document_records(user_id)
    records.insert(0, record)
    _save_document_records(user_id, records)
    return record


def _find_document_record(user_id: str, document_id: str) -> dict[str, Any]:
    for record in _load_document_records(user_id):
        if record.get("document_id") == document_id:
            return record
    raise HTTPException(status_code=404, detail="Document not found")


def _chat_history_path(user_id: str) -> Path:
    return CHAT_HISTORY_DIR / f"{_safe_user_key(user_id)}.json"


def _load_chat_history(user_id: str) -> list[dict]:
    record = _load_chat_record(user_id)
    return record.get("messages", [])


def _save_chat_history(user_id: str, messages: list[dict]) -> None:
    record = _load_chat_record(user_id)
    record["messages"] = messages[-CHAT_HISTORY_LIMIT:]
    _save_chat_record(user_id, record)


def _load_chat_record(user_id: str) -> dict[str, Any]:
    if _SessionLocal:
        safe_id = _safe_user_key(user_id)
        with _SessionLocal() as session:
            row = session.get(DBChatHistory, safe_id)
            if row is None:
                return {"messages": [], "agent_data": {}}
            payload = _read_json_text(row.data, {"messages": [], "agent_data": {}})
            if isinstance(payload, list):
                messages = [item for item in payload if isinstance(item, dict)]
                return {"messages": messages[-CHAT_HISTORY_LIMIT:], "agent_data": {}}
            if not isinstance(payload, dict):
                return {"messages": [], "agent_data": {}}
            messages = payload.get("messages", [])
            agent_data = payload.get("agent_data", {})
            if not isinstance(messages, list):
                messages = []
            if not isinstance(agent_data, dict):
                agent_data = {}
            return {
                "messages": [item for item in messages if isinstance(item, dict)][-CHAT_HISTORY_LIMIT:],
                "agent_data": agent_data,
            }

    path = _chat_history_path(user_id)
    if not path.exists():
        return {"messages": [], "agent_data": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"messages": [], "agent_data": {}}

    # Backward compatibility: older format stored a raw list of messages.
    if isinstance(data, list):
        messages = [item for item in data if isinstance(item, dict)]
        return {"messages": messages[-CHAT_HISTORY_LIMIT:], "agent_data": {}}

    if not isinstance(data, dict):
        return {"messages": [], "agent_data": {}}

    messages = data.get("messages", [])
    if not isinstance(messages, list):
        messages = []
    messages = [item for item in messages if isinstance(item, dict)][-CHAT_HISTORY_LIMIT:]

    agent_data = data.get("agent_data", {})
    if not isinstance(agent_data, dict):
        agent_data = {}

    return {
        "messages": messages,
        "agent_data": agent_data,
    }


def _save_chat_record(user_id: str, record: dict[str, Any]) -> None:
    if _SessionLocal:
        safe_id = _safe_user_key(user_id)
        payload = {
            "messages": record.get("messages", [])[-CHAT_HISTORY_LIMIT:],
            "agent_data": record.get("agent_data", {}),
        }
        with _SessionLocal() as session:
            row = session.get(DBChatHistory, safe_id)
            data = json.dumps(payload, ensure_ascii=False)
            if row is None:
                session.add(DBChatHistory(user_id=safe_id, data=data))
            else:
                row.data = data
            session.commit()
        return

    path = _chat_history_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "messages": record.get("messages", [])[-CHAT_HISTORY_LIMIT:],
        "agent_data": record.get("agent_data", {}),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _merge_chat_messages(existing: list[dict], incoming: list[dict]) -> list[dict]:
    merged = []
    seen = set()
    for message in [*existing, *incoming]:
        key = message.get("id") or f"{message.get('role','')}|{message.get('text','')}|{message.get('time','')}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(message)
    return merged[-CHAT_HISTORY_LIMIT:]


def _merge_agent_data(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(existing, dict):
        existing = {}
    if not isinstance(incoming, dict):
        return existing

    merged = dict(existing)
    for key, value in incoming.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_agent_data(merged[key], value)
            continue
        if key in merged and isinstance(merged[key], list) and isinstance(value, list):
            merged[key] = [*merged[key], *value][-CHAT_HISTORY_LIMIT:]
            continue
        merged[key] = value
    return merged

app = FastAPI(
    title="UniAssist OCR API",
    version="7.1",
    description=(
        "Windows-fixed edition. "
        "Auto-detects Tesseract; falls back to EasyOCR if not found. "
        "TF-IDF + NaiveBayes ML classifier. 9 Sri Lankan document types."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_ml: Pipeline = None


@app.on_event("startup")
def startup():
    global _ml, _OCR_ENGINE, _db_engine, _SessionLocal, _db_backend, _chatbot_agent, _rag_provider

    if DB_STRICT_MODE and not DATABASE_URL:
        raise RuntimeError("DB_STRICT_MODE=true but DATABASE_URL/NEON_DATABASE_URL is not set")

    if DATABASE_URL:
        try:
            _db_engine = create_engine(DATABASE_URL, pool_pre_ping=True)
            Base.metadata.create_all(bind=_db_engine)
            _SessionLocal = sessionmaker(bind=_db_engine, autoflush=False, autocommit=False)
            _db_backend = "postgresql"
            print("[DB] Connected to PostgreSQL/Neon.")
        except Exception as exc:
            _db_engine = None
            _SessionLocal = None
            _db_backend = "json"
            print(f"[DB] PostgreSQL init failed: {exc}")
            if DB_STRICT_MODE:
                raise RuntimeError("DB_STRICT_MODE=true and PostgreSQL connection failed") from exc
    else:
        _db_backend = "json"
        if DB_STRICT_MODE:
            raise RuntimeError("DB_STRICT_MODE=true requires DATABASE_URL or NEON_DATABASE_URL")

    if ChatbotAgent:
        eligibility_agent = EligibilityVerificationAgent() if EligibilityVerificationAgent else None
        financial_agent = FinancialFeasibilityAgent() if FinancialFeasibilityAgent else None
        recommendation_agent = RecommendationAgent() if RecommendationAgent else None

        rag_system = None
        if RAGSystem:
            try:
                rag_system = RAGSystem()
                _rag_provider = getattr(rag_system, "llm_provider", "none") or "none"
            except Exception as exc:
                rag_system = None
                _rag_provider = "none"
                print(f"[Chat] RAG init failed: {exc}")

        _chatbot_agent = ChatbotAgent(
            eligibility_agent=eligibility_agent,
            financial_agent=financial_agent,
            recommendation_agent=recommendation_agent,
            rag_system=rag_system,
        )
    else:
        _chatbot_agent = None

    # 1. Train / load ML model
    _ml = _load_model()

    # 2. Configure Tesseract (Windows path detection)
    if not USE_EASYOCR:
        tess_ok = _configure_tesseract()
        if tess_ok:
            _OCR_ENGINE = "tesseract"
        else:
            # Try EasyOCR as automatic fallback
            try:
                import easyocr  # noqa: F401
                _OCR_ENGINE = "easyocr"
                print("[OCR] Tesseract unavailable — using EasyOCR as fallback.")
            except ImportError:
                _OCR_ENGINE = None
                print(
                    "[OCR] WARNING: Neither Tesseract nor EasyOCR available!\n"
                    "  Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki\n"
                    "  OR: pip install easyocr  and set USE_EASYOCR=True"
                )
    else:
        _OCR_ENGINE = "easyocr"

    print(
        f"[UniAssist v7.1] Ready | "
        f"OCR: {_OCR_ENGINE or 'NONE — see warnings above'} | "
        f"ML: TF-IDF + NaiveBayes"
    )


@app.get("/health")
def health():
    return {
        "status": "ok" if _OCR_ENGINE else "degraded",
        "version": "7.1",
        "db": _db_backend,
        "db_url_set": bool(DATABASE_URL),
        "db_strict_mode": DB_STRICT_MODE,
        "ocr_engine": _OCR_ENGINE or "none",
        "use_easyocr_flag": USE_EASYOCR,
        "rag_provider": _rag_provider,
        "ml_model": "TF-IDF bigrams + MultinomialNB",
        "doc_types": list(TRAINING_DATA.keys()),
        "tesseract_paths_checked": _WINDOWS_TESSERACT_PATHS if sys.platform == "win32" else [],
    }


@app.get("/universities")
def get_universities(country: str | None = None):
    if not UNIVERSITIES_DB_PATH.exists():
        raise HTTPException(status_code=500, detail="Universities database is missing")

    database = _load_universities_database()
    available_countries = sorted(
        key for key, value in database.items()
        if key != "metadata" and isinstance(value, list)
    )

    normalized_country = _normalize_country_name(country)
    if normalized_country:
        universities = database.get(normalized_country)
        if not isinstance(universities, list):
            return {
                "universities": [],
                "count": 0,
                "country": normalized_country,
                "available_countries": available_countries,
            }
        return {
            "universities": universities,
            "count": len(universities),
            "country": normalized_country,
            "available_countries": available_countries,
        }

    universities = []
    for key in available_countries:
        universities.extend(database.get(key, []))

    return {
        "universities": universities,
        "count": len(universities),
        "available_countries": available_countries,
    }


@app.post("/auth/register")
def register(payload: RegisterPayload):
    name = (payload.name or "").strip()
    email = _normalize_email(payload.email)
    password = payload.password or ""
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    users = _load_users()
    if email in users:
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    users[email] = {
        "name": name,
        "email": email,
        "password_hash": _hash_password(password),
        "created_at": _utc_now(),
    }
    _save_users(users)
    return {
        "success": True,
        "user": {
            "name": name,
            "email": email,
        },
    }


@app.post("/auth/login")
def login(payload: LoginPayload):
    email = _normalize_email(payload.email)
    password = payload.password or ""
    users = _load_users()
    user = users.get(email)
    if not user or not _verify_password(password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = _issue_session_token(email)
    return {
        "token": token,
        "user": {
            "name": user.get("name") or email.split("@")[0],
            "email": email,
        },
    }


@app.get("/user/state")
def get_user_state(user_id: str, current_user_email: str = Depends(_require_current_user)):
    normalized_user_id = _ensure_user_access(user_id, current_user_email)
    return {
        "user_id": normalized_user_id,
        "state": _load_user_state_record(normalized_user_id),
    }


@app.post("/user/state")
def save_user_state(payload: UserStatePayload, current_user_email: str = Depends(_require_current_user)):
    normalized_user_id = _ensure_user_access(payload.user_id, current_user_email)
    _save_user_state_record(normalized_user_id, payload.state or {})
    return {
        "success": True,
        "user_id": normalized_user_id,
        "state": payload.state or {},
    }


@app.get("/documents")
def list_documents(current_user_email: str = Depends(_require_current_user)):
    return {
        "documents": _load_document_records(current_user_email),
    }


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user_email: str = Depends(_require_current_user),
):
    file_bytes = await file.read()
    record = _store_user_document(
        current_user_email,
        filename=file.filename,
        content_type=file.content_type,
        file_bytes=file_bytes,
    )
    return {
        "success": True,
        "document": record,
    }


@app.get("/documents/{document_id}/content")
def get_document_content(document_id: str, current_user_email: str = Depends(_require_current_user)):
    record = _find_document_record(current_user_email, document_id)
    file_path = _user_documents_path(current_user_email) / record.get("stored_name", "")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Document file is missing")
    return FileResponse(
        path=file_path,
        media_type=record.get("content_type") or "application/octet-stream",
        filename=record.get("filename") or file_path.name,
    )


@app.delete("/documents/{document_id}")
def delete_document(document_id: str, current_user_email: str = Depends(_require_current_user)):
    records = _load_document_records(current_user_email)
    remaining_records = []
    deleted_record = None
    for record in records:
        if record.get("document_id") == document_id and deleted_record is None:
            deleted_record = record
            continue
        remaining_records.append(record)

    if deleted_record is None:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = _user_documents_path(current_user_email) / deleted_record.get("stored_name", "")
    if file_path.exists():
        file_path.unlink()
    _save_document_records(current_user_email, remaining_records)
    return {
        "success": True,
        "document_id": document_id,
    }


@app.get("/chat/history")
def get_chat_history(user_id: str, current_user_email: str = Depends(_require_current_user)):
    normalized_user_id = _ensure_user_access(user_id, current_user_email)
    record = _load_chat_record(normalized_user_id)
    return {
        "user_id": normalized_user_id,
        "messages": record.get("messages", []),
        "agent_data": record.get("agent_data", {}),
    }


@app.post("/chat/history")
def append_chat_history(payload: ChatHistoryPayload, current_user_email: str = Depends(_require_current_user)):
    normalized_user_id = _ensure_user_access(payload.user_id, current_user_email)
    record = _load_chat_record(normalized_user_id)
    existing = record.get("messages", [])
    incoming = [message.model_dump() for message in payload.messages]
    merged = _merge_chat_messages(existing, incoming)
    merged_agent_data = _merge_agent_data(record.get("agent_data", {}), payload.agent_data or {})
    _save_chat_record(normalized_user_id, {
        "messages": merged,
        "agent_data": merged_agent_data,
    })
    return {
        "success": True,
        "user_id": normalized_user_id,
        "messages": merged,
        "agent_data": merged_agent_data,
        "count": len(merged),
    }


@app.delete("/chat/history")
def clear_chat_history(user_id: str, current_user_email: str = Depends(_require_current_user)):
    normalized_user_id = _ensure_user_access(user_id, current_user_email)
    if _SessionLocal:
        with _SessionLocal() as session:
            row = session.get(DBChatHistory, _safe_user_key(normalized_user_id))
            if row is not None:
                session.delete(row)
                session.commit()
    else:
        path = _chat_history_path(normalized_user_id)
        if path.exists():
            path.unlink()
    return {
        "success": True,
        "user_id": normalized_user_id,
        "messages": [],
        "agent_data": {},
    }


@app.post("/chat/respond")
def chat_respond(payload: ChatRespondPayload, authorization: str | None = Header(default=None)):
    message = (payload.user_message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="user_message is required")

    current_user_email = _authenticate_token(authorization, required=False)
    context = payload.context if isinstance(payload.context, dict) else {}

    if not isinstance(context.get("universities"), list):
        database = _load_universities_database()
        all_universities: list[dict[str, Any]] = []
        for country_name, country_universities in database.items():
            if country_name == "metadata" or not isinstance(country_universities, list):
                continue
            all_universities.extend([item for item in country_universities if isinstance(item, dict)])
        context = {
            **context,
            "universities": all_universities,
        }

    if current_user_email:
        if not isinstance(context.get("conversation_history"), list):
            stored_record = _load_chat_record(current_user_email)
            stored_messages = stored_record.get("messages", [])
            history = [
                {"role": m.get("role"), "text": m.get("text", "")}
                for m in stored_messages[-20:]
                if isinstance(m, dict) and m.get("role") in ("user", "bot", "assistant")
            ]
        else:
            history = context["conversation_history"]
        context = {
            **context,
            "profile_data": context.get("profile_data") or _load_user_state_record(current_user_email),
            "document_data": context.get("document_data") or {
                "documents": _load_document_records(current_user_email),
            },
            "conversation_history": history,
        }

    if _chatbot_agent:
        result = _chatbot_agent.process_message(message, context)
        response_text = result.get("response") or "I'm here to help with your application journey."
        return {
            "response": response_text,
            "intent": result.get("intent"),
            "actions": result.get("actions", []),
            "agent_data": result.get("agent_data", {}),
            "source": "backend_agent",
        }

    raise HTTPException(status_code=503, detail="Chat agent is unavailable. Check server startup logs.")


@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    doc_type: str = Form(default="auto"),
    authorization: str | None = Header(default=None),
):
    if _OCR_ENGINE is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No OCR engine available. "
                "Install Tesseract (https://github.com/UB-Mannheim/tesseract/wiki) "
                "OR install easyocr (pip install easyocr) and set USE_EASYOCR=True."
            ),
        )

    file_bytes = await file.read()
    suffix = os.path.splitext(file.filename or "doc.jpg")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        t0 = time.time()
        text, ocr_conf = _run_ocr(tmp_path)
        ocr_time = round(time.time() - t0, 2)

        if not text.strip():
            return {
                "success": False,
                "errors": ["No text extracted — check image quality / lighting."],
                "doc_type": "unknown",
            }

        # ML classification — always use the trained model
        proba    = _ml.predict_proba([text])[0]
        best_idx = int(np.argmax(proba))
        ml_label = _ml.classes_[best_idx]
        ml_conf  = float(proba[best_idx])

        # When top-1 confidence is low, check top-2 and pick the one
        # with higher probability (still fully ML — no rule keywords).
        if ml_conf < CONFIDENCE_THRESHOLD:
            sorted_idx = np.argsort(proba)[::-1]
            second_idx = int(sorted_idx[1])
            second_conf = float(proba[second_idx])
            if second_conf > ml_conf * 0.9:
                detected, final_conf, method = _ml.classes_[second_idx], second_conf, "ml_low_conf_top2"
            else:
                detected, final_conf, method = ml_label, ml_conf, "ml_low_conf"
        else:
            detected, final_conf, method = ml_label, ml_conf, "ml"

        # Override if doc_type explicitly passed
        if doc_type != "auto" and doc_type in TRAINING_DATA:
            detected = doc_type

        fields = _extract(detected, text)
        conf   = _score(detected, fields, ocr_conf, final_conf)

        # Map internal keys to frontend-expected keys
        _TYPE_MAP = {
            "alevel":    "A-Level Results",
            "bachelor":  "Bachelor's Degree",
            "master":    "Master's Degree",
            "diploma":   "Diploma",
            "ielts":     "IELTS Certificate",
            "toefl":     "TOEFL Certificate",
            "pte":       "PTE Certificate",
            "passport":  "Passport",
            "financial": "Financial Statement",
        }

        response_payload = {
            "success":                True,
            "data":                   {**fields, "document_type": _TYPE_MAP.get(detected, detected)},
            "confidence":             conf,
            "ml_confidence":          round(final_conf, 3),
            "classification_method":  method,
            "ocr_confidence":         round(ocr_conf, 3),
            "ocr_engine":             _OCR_ENGINE,
            "ocr_time_sec":           ocr_time,
            "raw_text_preview":       text[:800],
            "message":                (
                f"{_OCR_ENGINE.capitalize()} OCR — "
                f"{round(conf * 100)}% confidence — {_TYPE_MAP.get(detected, detected)}"
            ),
            "warnings":               (
                ["Low confidence extraction — please verify fields manually."]
                if conf < 0.5 else []
            ),
        }
        current_user_email = _authenticate_token(authorization, required=False)
        if current_user_email:
            response_payload["document"] = _store_user_document(
                current_user_email,
                filename=file.filename,
                content_type=file.content_type,
                file_bytes=file_bytes,
                extracted_data=response_payload["data"],
            )
        return response_payload

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing error: {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    _ml = _load_model()
    _configure_tesseract()

    if len(sys.argv) > 1:
        image_file = sys.argv[1]
        if not os.path.isfile(image_file):
            print(f"File not found: {image_file}")
            sys.exit(1)
        text, ocr_conf = _run_ocr(image_file)
        proba = _ml.predict_proba([text])[0]
        best  = _ml.classes_[proba.argmax()]
        conf  = proba.max()
        # ML-only: use top-2 when low confidence
        if conf < CONFIDENCE_THRESHOLD:
            sorted_idx = proba.argsort()[::-1]
            second_conf = proba[sorted_idx[1]]
            detected = _ml.classes_[sorted_idx[1]] if second_conf > conf * 0.9 else best
        else:
            detected = best
        fields = _extract(detected, text)
        print(json.dumps({
            "doc_type":           detected,
            "ml_confidence":      round(conf, 3),
            "ocr_confidence":     round(ocr_conf, 3),
            "fields":             fields,
            "ocr_text_preview":   text[:400],
        }, indent=2))
    else:
        import uvicorn
        uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)