"""
UniAssist API Server v7.3 — Dual OCR Edition
============================================
Highlights:
    ✓ Tesseract OCR with Windows auto-detection
    ✓ EasyOCR fallback (CPU mode) when Tesseract is unavailable
    ✓ TF-IDF + Naive Bayes document classifier
    ✓ FastAPI endpoints for OCR and multi-agent flows

Windows install:
    Download: https://github.com/UB-Mannheim/tesseract/wiki
    Install to: C:\\Program Files\\Tesseract-OCR\\tesseract.exe

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
import urllib.request
from datetime import datetime, timezone, timedelta
from hashlib import pbkdf2_hmac
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import cv2
from PIL import Image
from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

_MODULE_DIR = Path(__file__).resolve().parent
for _env_path in (_MODULE_DIR / ".env", _MODULE_DIR.parent / ".env"):
    if _env_path.exists():
        load_dotenv(_env_path, override=False)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH           = "uniassist_classifier.pkl"
CONFIDENCE_THRESHOLD = 0.40
MAX_IMAGE_DIM        = 1000    # px cap — prevents RAM freeze
# Default to degraded mode unless explicitly forced strict; this keeps boot probes alive
# in lightweight runtimes (e.g., Colab) where OCR dependencies may be absent.
OCR_STRICT_MODE = os.environ.get("OCR_STRICT_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}

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

USE_EASYOCR = False  # legacy switch; prefer OCR_ENGINE env var
OCR_ENGINE_MODE = os.environ.get("OCR_ENGINE", "auto").strip().lower()
if OCR_ENGINE_MODE not in {"auto", "tesseract", "easyocr"}:
    OCR_ENGINE_MODE = "auto"
_OCR_ENGINE = None  # will be set to "tesseract", "easyocr", or None after startup
_OCR_READINESS = {
    "ready": False,
    "mode": OCR_ENGINE_MODE,
    "strict_mode": OCR_STRICT_MODE,
    "checks": {},
    "messages": [],
}


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
    print("  Fallback: install EasyOCR (pip install easyocr)")
    print()
    print("=" * 60)
    print()


def _collect_ocr_readiness() -> dict[str, Any]:
    checks: dict[str, Any] = {
        "pytesseract_package": False,
        "tesseract_binary": False,
        "easyocr_package": False,
    }
    messages: list[str] = []

    try:
        import pytesseract  # noqa: F401
        checks["pytesseract_package"] = True
        checks["tesseract_binary"] = _configure_tesseract()
    except Exception:
        messages.append("Install pytesseract package: pip install pytesseract")

    try:
        import easyocr  # noqa: F401
        checks["easyocr_package"] = True
    except Exception:
        messages.append("Install EasyOCR package: pip install easyocr")

    ready = bool(checks["tesseract_binary"] or checks["easyocr_package"])
    if not checks["tesseract_binary"]:
        messages.append("Install Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki")

    return {
        "ready": ready,
        "mode": OCR_ENGINE_MODE,
        "strict_mode": OCR_STRICT_MODE,
        "checks": checks,
        "messages": sorted(set(messages)),
    }


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


def _tesseract_config_for_doc(doc_type_hint):
    """Return OCR preset key and tesseract config based on doc type hint."""
    hint = (doc_type_hint or "auto").strip().lower()

    # Table-heavy docs benefit from sparse text layout mode.
    if hint == "financial":
        return "statement", "--oem 1 --psm 4"

    # MRZ + compact personal fields are typically line/block text.
    if hint == "passport":
        return "passport", "--oem 1 --psm 6"

    # Score reports and certificates are generally dense block text.
    if hint in {"ielts", "toefl", "pte", "alevel"}:
        return "score_report", "--oem 1 --psm 6"

    if hint in {"bachelor", "master", "diploma"}:
        return "certificate", "--oem 1 --psm 3"

    return "default", "--oem 1 --psm 3"


# ─────────────────────────────────────────────────────────────────────────────
# OCR ENGINE
# ─────────────────────────────────────────────────────────────────────────────

_easyocr_reader = None


def _run_easyocr_engine(image_path):
    """Run EasyOCR in CPU mode."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        print("[EasyOCR] Loading model (first run may take a few seconds) ...")
        _easyocr_reader = easyocr.Reader(
            ["en"],
            gpu=False,
            model_storage_directory=os.path.join(
                os.path.expanduser("~"), ".EasyOCR", "model"
            ),
            download_enabled=True,
            verbose=False,
        )
        print("[EasyOCR] Ready.")

    img = _preprocess(image_path)
    bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    results = _easyocr_reader.readtext(
        bgr_img,
        detail=1,
        batch_size=1,
        workers=0,
        beamWidth=3,
    )

    words = [t for (_, t, c) in results if t.strip() and c > 0.25]
    confs = [c for (_, t, c) in results if t.strip() and c > 0.25]
    text = " ".join(words)
    conf = float(np.mean(confs)) if confs else 0.0
    text = re.sub(r'[\u0D80-\u0DFF\u0B80-\u0BFF]+', '', text)
    return _correct(text.strip()), conf, "easyocr_default"


def _run_tesseract(image_path, doc_type_hint="auto"):
    """Run Tesseract OCR. Raises if unavailable."""
    import pytesseract
    gray = _preprocess(image_path)
    preset_key, config = _tesseract_config_for_doc(doc_type_hint)
    text = pytesseract.image_to_string(gray, config=config)
    data = pytesseract.image_to_data(
        gray, config=config,
        output_type=pytesseract.Output.DICT,
    )
    confs = [int(c) for c in data["conf"] if int(c) > 0]
    conf = (sum(confs) / len(confs) / 100) if confs else 0.0
    text = re.sub(r'[\u0D80-\u0DFF\u0B80-\u0BFF]+', '', text)
    return _correct(text.strip()), conf, preset_key


def _run_ocr(image_path, doc_type_hint="auto"):
    """Run OCR via selected engine with Tesseract-first fallback logic."""
    global _OCR_ENGINE

    force_easyocr = OCR_ENGINE_MODE == "easyocr" or USE_EASYOCR
    force_tesseract = OCR_ENGINE_MODE == "tesseract"

    if force_easyocr:
        try:
            result = _run_easyocr_engine(image_path)
            _OCR_ENGINE = "easyocr"
            return result
        except Exception as ee:
            raise RuntimeError(f"EasyOCR failed while OCR mode is forced to easyocr: {ee}")

    if _OCR_ENGINE == "easyocr" and not force_tesseract:
        return _run_easyocr_engine(image_path)

    if _OCR_ENGINE == "tesseract":
        try:
            return _run_tesseract(image_path, doc_type_hint=doc_type_hint)
        except Exception:
            if force_tesseract:
                raise RuntimeError("Tesseract failed while OCR mode is forced to tesseract.")
            # Keep service resilient: when Tesseract fails on a specific file,
            # attempt EasyOCR as a fallback before returning an error.
            try:
                result = _run_easyocr_engine(image_path)
                _OCR_ENGINE = "easyocr"
                return result
            except Exception:
                pass

    try:
        result = _run_tesseract(image_path, doc_type_hint=doc_type_hint)
        _OCR_ENGINE = "tesseract"
        return result
    except Exception as te:
        if force_tesseract:
            _OCR_ENGINE = None
            raise RuntimeError(
                "Tesseract OCR unavailable while OCR_ENGINE=tesseract.\n\n"
                "To fix this:\n"
                "  1. Download: https://github.com/UB-Mannheim/tesseract/wiki\n"
                "  2. Install to C:\\Program Files\\Tesseract-OCR\\\n"
                "  3. Restart server\n"
                f"\nDetails: {te}"
            )
        try:
            result = _run_easyocr_engine(image_path)
            _OCR_ENGINE = "easyocr"
            return result
        except Exception as ee:
            _OCR_ENGINE = None
            raise RuntimeError(
                "No OCR engine available.\n\n"
                "To fix this:\n"
                "  Option A (Tesseract):\n"
                "    1. Download: https://github.com/UB-Mannheim/tesseract/wiki\n"
                "    2. Install to C:\\Program Files\\Tesseract-OCR\\\n"
                "  Option B (EasyOCR):\n"
                "    pip install easyocr\n"
                f"\nDetails: tesseract_error={te} | easyocr_error={ee}"
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


_REQUIRED_FIELDS = {
    "alevel": ["subjects", "grades"],
    "bachelor": ["degree", "university"],
    "master": ["degree", "university"],
    "diploma": ["degree"],
    "ielts": ["overall"],
    "toefl": ["total"],
    "pte": ["overall"],
    "passport": ["passport_no", "surname", "given_names"],
    "financial": ["bank_name", "account_no", "closing_bal"],
}


def _has_value(v):
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, (list, tuple, set, dict)):
        return len(v) > 0
    return True


def _field_diagnostics(doc_type, fields, ocr_conf, ml_conf):
    fw = _FW.get(doc_type, {})
    required = _REQUIRED_FIELDS.get(doc_type, [])

    field_confidence = {}
    missing_reasons = {}

    for key, value in fields.items():
        present = _has_value(value)
        base_w = fw.get(key, 0.15)
        if present:
            field_confidence[key] = round(min(1.0, 0.35 + base_w * 0.35 + ocr_conf * 0.15 + ml_conf * 0.15), 3)
        else:
            field_confidence[key] = 0.0

    for key in required:
        if not _has_value(fields.get(key)):
            missing_reasons[key] = (
                "required_field_not_detected_from_ocr_text"
                if ocr_conf >= 0.4 else
                "low_ocr_confidence_image_quality_or_layout"
            )

    return field_confidence, missing_reasons


def _normalize_fields(doc_type, fields):
    """Add normalized schema keys while keeping existing keys for compatibility."""
    normalized = dict(fields)
    normalized["schema_version"] = "2.0"
    normalized["doc_type_key"] = doc_type

    if doc_type == "alevel":
        subjects = normalized.get("subjects") or []
        grades = normalized.get("grades") or []

        if isinstance(subjects, dict):
            subject_grade_map = {str(k): str(v).upper() for k, v in subjects.items()}
        else:
            subject_grade_map = {}
            if isinstance(subjects, list):
                if isinstance(grades, list):
                    for idx, subject in enumerate(subjects):
                        if not subject:
                            continue
                        grade = grades[idx] if idx < len(grades) else None
                        subject_grade_map[str(subject)] = str(grade).upper() if grade else ""

        normalized["subject_grade_map"] = subject_grade_map

    return normalized


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except Exception:
            return None
    return None


def _validate_parsed_fields(doc_type: str, fields: dict[str, Any]) -> tuple[dict[str, Any], list[str], float]:
    validated = dict(fields or {})
    issues: list[str] = []

    year_val = validated.get("year")
    if year_val is not None:
        y = _to_float(year_val)
        current_year = datetime.now(timezone.utc).year
        if y is None or y < 1990 or y > current_year + 1:
            issues.append("invalid_year")

    if doc_type == "alevel":
        grades = validated.get("grades") or []
        allowed = {"A", "B", "C", "S", "F"}
        if isinstance(grades, list):
            bad = [g for g in grades if str(g).upper() not in allowed]
            if bad:
                issues.append("invalid_alevel_grades")

    if doc_type == "ielts":
        for key in ["overall", "listening", "reading", "writing", "speaking"]:
            v = validated.get(key)
            if v is None:
                continue
            f = _to_float(v)
            if f is None or f < 0 or f > 9 or (f * 2) % 1 != 0:
                issues.append(f"invalid_{key}_band")

    if doc_type == "toefl":
        for key in ["total", "reading", "listening", "speaking", "writing"]:
            v = validated.get(key)
            if v is None:
                continue
            f = _to_float(v)
            if f is None or f < 0 or f > 120:
                issues.append(f"invalid_{key}_score")

    if doc_type == "pte":
        for key in ["overall", "listening", "reading", "writing", "speaking"]:
            v = validated.get(key)
            if v is None:
                continue
            f = _to_float(v)
            if f is None or f < 10 or f > 90:
                issues.append(f"invalid_{key}_score")

    if doc_type == "passport":
        passport_no = str(validated.get("passport_no") or "").strip().upper()
        if passport_no and not re.match(r"^[A-Z][0-9]{7}$", passport_no):
            issues.append("invalid_passport_number")

    if doc_type == "financial":
        closing_bal = _to_float(validated.get("closing_bal"))
        if closing_bal is None:
            issues.append("invalid_closing_balance")
        elif closing_bal <= 0:
            issues.append("non_positive_closing_balance")
        else:
            validated["closing_bal"] = round(closing_bal, 2)

    unique_issues = sorted(set(issues))
    parse_confidence = round(max(0.2, 1.0 - (0.12 * len(unique_issues))), 3)
    return validated, unique_issues, parse_confidence


def _score_with_validation(doc_type, fields, ocr_conf, ml_conf, parse_conf):
    base = _score(doc_type, fields, ocr_conf, ml_conf)
    return round(max(0.0, min(1.0, base * 0.85 + parse_conf * 0.15)), 3)


_TYPE_MAP = {
    "alevel": "A-Level Results",
    "bachelor": "Bachelor's Degree",
    "master": "Master's Degree",
    "diploma": "Diploma",
    "ielts": "IELTS Certificate",
    "toefl": "TOEFL Certificate",
    "pte": "PTE Certificate",
    "passport": "Passport",
    "financial": "Financial Statement",
}

_TYPE_LABEL_TO_KEY = {label: key for key, label in _TYPE_MAP.items()}


def _build_manual_review_block(
    confidence: float,
    missing_field_reasons: dict[str, str],
    validation_issues: list[str],
) -> dict[str, Any]:
    reasons: list[str] = []
    if confidence < 0.65:
        reasons.append("low_overall_confidence")
    if missing_field_reasons:
        reasons.append("missing_required_fields")
    if validation_issues:
        reasons.append("field_validation_issues")

    required = bool(reasons)
    return {
        "required": required,
        "reason_codes": reasons,
        "recommended_action": (
            "Review and correct extracted fields before eligibility calculations."
            if required else
            "No manual correction required."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FIELD EXTRACTORS
# ─────────────────────────────────────────────────────────────────────────────

def _f(p, text):
    m = re.search(p, text, re.I)
    return m.group(1).strip() if m else None


def _fa(p, text):
    return re.findall(p, text, re.I)


_ALEVEL_SUBJECT_MAP = [
    ("Combined Mathematics", r"combined\s+mathematic(?:s|z)|combined\s+maths?"),
    ("Physics", r"physics"),
    ("Chemistry", r"chemistr(?:y|v)"),
    ("Biology", r"biolog(?:y|v)"),
    ("General English", r"general\s+english"),
    ("Political Science", r"political\s+science"),
    ("Logic & Scientific Method", r"logic.*scientific\s+method"),
    ("Common General Test", r"common\s+general\s+test"),
    ("Economics", r"economics"),
    ("Geography", r"geograph(?:y|v)"),
    ("Accounting", r"accounting"),
    ("ICT", r"\b(?:ict|information\s+and\s+communication\s+technology)\b"),
    ("Sinhala", r"\bsinhala\b"),
]


def _extract_alevel_subjects_grades(text):
    subjects = []
    grades = []

    # Line-wise parsing helps when OCR outputs table rows as plain lines.
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue

        matched_subject = None
        for subject_name, subject_pattern in _ALEVEL_SUBJECT_MAP:
            if re.search(subject_pattern, line, re.I):
                matched_subject = subject_name
                if subject_name not in subjects:
                    subjects.append(subject_name)
                break

        if not matched_subject:
            continue

        grade_match = re.search(r"(?:grade\s*[:\-]?\s*)?\b([ABCSF])\b", line, re.I)
        if grade_match:
            grades.append(grade_match.group(1).upper())

    # Fallback scan for compact OCR text where lines are merged.
    if not subjects:
        for subject_name, subject_pattern in _ALEVEL_SUBJECT_MAP:
            if re.search(subject_pattern, text, re.I):
                subjects.append(subject_name)

    if not grades:
        grades = [g.upper() for g in re.findall(r"\b(?:grade\s*[:\-]?\s*)?([ABCSF])\b", text, re.I)]

    return subjects, grades


def _extract(doc_type, text):
    t = text.lower()
    if doc_type == "alevel":
        subjects, grades = _extract_alevel_subjects_grades(text)

        stream = _f(r"(?:subject\s*)?stream[:\s]+([A-Z]+)", text)
        if not stream:
            if any(s in subjects for s in ["Physics", "Chemistry", "Biology"]):
                stream = "BIO SCIENCE"
            elif "Combined Mathematics" in subjects:
                stream = "PHYSICAL SCIENCE"
            elif any(s in subjects for s in ["Political Science", "Sinhala", "Economics"]):
                stream = "ARTS"

        return {
            "name":          _f(r"(?:certify\s+that|candidate\s*name)[:\s]+([A-Z][A-Z.\s]{3,})", text),
            "index_number":  _f(r"(?:index\s*(?:number|no\.?))[:\s]*([0-9]{6,10})", text),
            "year":          _f(r"(20\d{2})", text),
            "subjects":      subjects,
            "grades":        grades,
            "z_score":       _f(r"z\s*[-.]?\s*score[:\s]*([0-9.]+)", text),
            "district_rank": _f(r"District\s*Rank[:\s]+(\d+)", text),
            "island_rank":   _f(r"Island\s*Rank[:\s]+(\d+)", text),
            "stream":        stream,
            "date_of_issue": _f(r"(?:date\s*of\s*issue|issued\s*date|date)[:\s]+(.+?\d{4})", text),
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

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import tempfile
from typing import Literal
from sqlalchemy import Column, DateTime, LargeBinary, String, Text, create_engine, func, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

try:
    from .core.agents.chatbot_agent import ChatbotAgent
    from .core.agents.eligibility_verification_agent import EligibilityVerificationAgent
    from .core.agents.financial_feasibility_agent import FinancialFeasibilityAgent
    from .core.agents.recommendation_agent import RecommendationAgent
    from .core.rag_system import RAGSystem
    from .core.monitoring import get_metrics_collector
except Exception:
    try:
        from core.agents.chatbot_agent import ChatbotAgent
        from core.agents.eligibility_verification_agent import EligibilityVerificationAgent
        from core.agents.financial_feasibility_agent import FinancialFeasibilityAgent
        from core.agents.recommendation_agent import RecommendationAgent
        from core.rag_system import RAGSystem
        from core.monitoring import get_metrics_collector
    except Exception:
        ChatbotAgent = None
        EligibilityVerificationAgent = None
        FinancialFeasibilityAgent = None
        RecommendationAgent = None
        RAGSystem = None
        get_metrics_collector = None

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
    binary_data = Column(LargeBinary, nullable=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class DBApplication(Base):
    __tablename__ = "applications"

    application_id = Column(String(64), primary_key=True, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    status = Column(String(32), nullable=False, default="submitted")
    data = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class DBPolicySnapshot(Base):
    __tablename__ = "policy_snapshots"

    policy_key = Column(String(128), primary_key=True, index=True)
    source = Column(String(1024), nullable=False, default="unknown")
    confidence = Column(String(32), nullable=False, default="0.0")
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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except Exception:
        return default


def _env_csv(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or default


DATABASE_URL = _normalize_database_url(
    os.environ.get("DATABASE_URL") or os.environ.get("NEON_DATABASE_URL") or ""
)
DB_STRICT_MODE = _env_true("DB_STRICT_MODE", True)
RAG_ENABLED = _env_true("RAG_ENABLED", True)
POLICY_INGEST_ON_STARTUP = _env_true("POLICY_INGEST_ON_STARTUP", True)
POLICY_VISA_RISK_URL = (os.environ.get("POLICY_VISA_RISK_URL") or os.environ.get("VISA_RISK_DATA_URL") or "").strip()
POLICY_VISA_RISK_CONFIDENCE = _env_float("POLICY_VISA_RISK_CONFIDENCE", 0.9)
POLICY_LIVING_COSTS_URL = (os.environ.get("POLICY_LIVING_COSTS_URL") or "").strip()
POLICY_LIVING_COSTS_CONFIDENCE = _env_float("POLICY_LIVING_COSTS_CONFIDENCE", 0.85)
POLICY_SCHOLARSHIPS_URL = (os.environ.get("POLICY_SCHOLARSHIPS_URL") or "").strip()
POLICY_SCHOLARSHIPS_CONFIDENCE = _env_float("POLICY_SCHOLARSHIPS_CONFIDENCE", 0.85)
POLICY_ELIGIBILITY_THRESHOLDS_URL = (os.environ.get("POLICY_ELIGIBILITY_THRESHOLDS_URL") or "").strip()
POLICY_ELIGIBILITY_THRESHOLDS_CONFIDENCE = _env_float("POLICY_ELIGIBILITY_THRESHOLDS_CONFIDENCE", 0.9)
SESSION_TTL_HOURS = _env_int("SESSION_TTL_HOURS", 24 * 30)
ALLOW_PRIVILEGED_SELF_REGISTRATION = _env_true("ALLOW_PRIVILEGED_SELF_REGISTRATION", False)
PASSWORD_MIN_LENGTH = _env_int("PASSWORD_MIN_LENGTH", 6)
PASSWORD_REQUIRE_COMPLEXITY = _env_true("PASSWORD_REQUIRE_COMPLEXITY", False)
BOOTSTRAP_ADMIN_EMAIL = (os.environ.get("BOOTSTRAP_ADMIN_EMAIL") or "").strip().lower()
BOOTSTRAP_ADMIN_PASSWORD = os.environ.get("BOOTSTRAP_ADMIN_PASSWORD") or ""
BOOTSTRAP_ADMIN_NAME = (os.environ.get("BOOTSTRAP_ADMIN_NAME") or "System Admin").strip() or "System Admin"
BOOTSTRAP_ADVISOR_EMAIL = (os.environ.get("BOOTSTRAP_ADVISOR_EMAIL") or "").strip().lower()
BOOTSTRAP_ADVISOR_PASSWORD = os.environ.get("BOOTSTRAP_ADVISOR_PASSWORD") or ""
BOOTSTRAP_ADVISOR_NAME = (os.environ.get("BOOTSTRAP_ADVISOR_NAME") or "System Advisor").strip() or "System Advisor"
AUTH_WINDOW_SECONDS = _env_int("AUTH_WINDOW_SECONDS", 300)
AUTH_MAX_LOGIN_ATTEMPTS = _env_int("AUTH_MAX_LOGIN_ATTEMPTS", 10)
METRICS_PUBLIC = _env_true("METRICS_PUBLIC", True)
CORS_ALLOW_ORIGINS = _env_csv(
    "CORS_ALLOW_ORIGINS",
    [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
)
CORS_ALLOW_METHODS = _env_csv(
    "CORS_ALLOW_METHODS",
    ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
CORS_ALLOW_HEADERS = _env_csv(
    "CORS_ALLOW_HEADERS",
    ["Authorization", "Content-Type", "Accept", "Origin"],
)
CORS_ALLOW_CREDENTIALS = _env_true("CORS_ALLOW_CREDENTIALS", True)
_default_cors_origin_regex = r"^https://[-a-zA-Z0-9]+\.trycloudflare\.com$" if os.environ.get("COLAB_RELEASE_TAG") else ""
CORS_ALLOW_ORIGIN_REGEX = (os.environ.get("CORS_ALLOW_ORIGIN_REGEX") or _default_cors_origin_regex).strip() or None

_db_engine = None
_SessionLocal = None
_db_backend = "uninitialized"
_chatbot_agent = None
_rag_provider = "none"
_metrics_collector = get_metrics_collector() if get_metrics_collector else None
_LOGIN_ATTEMPTS: dict[str, list[float]] = {}

CHAT_HISTORY_DIR = Path(__file__).resolve().parent / "data" / "chat_history"
USERS_DIR = Path(__file__).resolve().parent / "data" / "users"
USER_STATE_DIR = Path(__file__).resolve().parent / "data" / "user_state"
DOCUMENTS_DIR = Path(__file__).resolve().parent / "data" / "documents"
APPLICATIONS_DIR = Path(__file__).resolve().parent / "data" / "applications"
HISTORICAL_OUTCOMES_PATH = Path(__file__).resolve().parent / "data" / "training" / "historical_admissions_outcomes.jsonl"
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
    role: Literal["student", "advisor", "admin"] = "student"


class LoginPayload(BaseModel):
    email: str
    password: str


class UserStatePayload(BaseModel):
    user_id: str
    state: dict[str, Any]


class UpdateRolePayload(BaseModel):
    role: Literal["student", "advisor", "admin"]


class SubmitApplicationPayload(BaseModel):
    university_name: str
    university_id: str | None = None
    program: str
    country: str
    eligibility_tier: str | None = None
    grade_point: float | None = None
    notes: str | None = None
    university_data: dict[str, Any] | None = None


class UpdateApplicationStatusPayload(BaseModel):
    status: Literal["submitted", "under_review", "accepted", "rejected", "withdrawn"]
    advisor_notes: str | None = None


class HistoricalOutcomePayload(BaseModel):
    application_id: str | None = None
    user_id: str | None = None
    country: str | None = None
    university_id: str | None = None
    university_name: str | None = None
    program: str | None = None
    stream: str | None = None
    gpa: float | None = None
    university_min_gpa: float | None = None
    gpa_diff: float | None = None
    tier_label: Literal["top", "good", "average", "foundation"] | None = None
    match_label: Literal["strong_match", "meets_minimum", "borderline", "below_minimum"] | None = None
    alignment_label: Literal[0, 1] | None = None
    admission_outcome: Literal["accepted", "rejected"]


class ImportHistoricalOutcomesPayload(BaseModel):
    records: list[HistoricalOutcomePayload]


class ChatRespondPayload(BaseModel):
    user_message: str
    context: dict[str, Any] = {}


class OCRManualCorrectionPayload(BaseModel):
    corrected_fields: dict[str, Any]
    reviewer_note: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_email(value: str) -> str:
    return (value or "").strip().lower()


def _normalize_user_role(value: str | None, *, strict: bool = False) -> str:
    """Normalize role labels from payloads or persisted records.

    strict=True should be used for external input validation.
    strict=False is tolerant for legacy stored values to prevent auth lockouts.
    """
    raw = (value or "student").strip().lower()
    alias_map = {
        "adviser": "advisor",
        "advisr": "advisor",
        "counselor": "advisor",
        "counsellor": "advisor",
        "administrator": "admin",
        "superadmin": "admin",
    }
    role = alias_map.get(raw, raw)
    if role in {"student", "advisor", "admin"}:
        return role
    if strict:
        raise HTTPException(status_code=400, detail="Role must be one of: student, advisor, admin")
    return "student"


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


def _require_session_local():
    if _SessionLocal is None:
        raise RuntimeError(
            "Database session is unavailable. Configure DATABASE_URL for PostgreSQL/Neon and restart the server."
        )
    return _SessionLocal


def _initialize_database() -> None:
    global _db_engine, _SessionLocal, _db_backend

    if not DB_STRICT_MODE:
        raise RuntimeError("DB_STRICT_MODE=false is no longer supported. Database-only persistence is required.")
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL or NEON_DATABASE_URL is required. Local SQLite and JSON fallback have been removed."
        )
    if DATABASE_URL.startswith("sqlite://"):
        raise RuntimeError("SQLite is not supported for the API server. Use PostgreSQL/Neon for persistence.")

    if _SessionLocal is not None and _db_engine is not None:
        return

    db_url = DATABASE_URL
    try:
        _db_engine = create_engine(db_url, pool_pre_ping=True)
        _db_backend = "postgresql"
        print("[DB] Connected to PostgreSQL/Neon.")

        Base.metadata.create_all(bind=_db_engine)
        _ensure_document_upload_schema()
        _SessionLocal = sessionmaker(bind=_db_engine, autoflush=False, autocommit=False)
    except Exception as exc:
        _db_engine = None
        _SessionLocal = None
        _db_backend = "unavailable"
        print(f"[DB] Database init failed: {exc}")
        raise RuntimeError("Database initialization failed. Set DATABASE_URL to a reachable PostgreSQL/Neon instance.") from exc


def _validate_visa_risk_snapshot(payload: Any) -> dict[str, Any]:
    """Validate and normalize visa-risk matrix before persisting a snapshot."""
    if not isinstance(payload, dict):
        raise RuntimeError("Visa-risk snapshot must be a JSON object.")

    clean: dict[str, dict[str, str]] = {}
    for destination, table in payload.items():
        destination_name = _normalize_country_name(str(destination)) or str(destination)
        if not isinstance(table, dict):
            continue
        normalized_table: dict[str, str] = {}
        for nationality, risk in table.items():
            normalized_nat = "_default" if str(nationality) == "_default" else (_normalize_country_name(str(nationality)) or str(nationality))
            risk_level = str(risk).strip().lower()
            if risk_level in {"low", "medium", "high"}:
                normalized_table[normalized_nat] = risk_level
        if normalized_table:
            clean[destination_name] = normalized_table

    if not clean:
        raise RuntimeError("Visa-risk snapshot validation failed: no valid destination records.")

    return {
        "destinations": clean,
        "record_count": sum(len(v) for v in clean.values()),
    }


def _validate_living_costs_snapshot(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("Living-costs snapshot must be a JSON object.")

    clean: dict[str, dict[str, Any]] = {}
    for country, value in payload.items():
        if not isinstance(value, dict):
            continue
        amount = value.get("amount")
        currency = str(value.get("currency") or "").strip().upper()
        try:
            amount_val = float(amount)
        except Exception:
            continue
        if amount_val <= 0 or not currency:
            continue
        clean[str(country).strip()] = {"amount": amount_val, "currency": currency}

    if not clean:
        raise RuntimeError("Living-costs snapshot validation failed: no valid records.")

    return {"costs": clean, "record_count": len(clean)}


def _validate_scholarships_snapshot(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("Scholarships snapshot must be a JSON object.")

    clean: dict[str, list[dict[str, str]]] = {}
    total = 0
    for country, rows in payload.items():
        if not isinstance(rows, list):
            continue
        cleaned_rows: list[dict[str, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            entry = {
                "name": str(row.get("name") or "").strip(),
                "type": str(row.get("type") or "").strip(),
                "coverage": str(row.get("coverage") or "").strip(),
                "eligibility": str(row.get("eligibility") or "").strip(),
                "deadline": str(row.get("deadline") or "").strip(),
                "website": str(row.get("website") or "").strip(),
            }
            if not entry["name"]:
                continue
            cleaned_rows.append(entry)
        if cleaned_rows:
            clean[str(country).strip()] = cleaned_rows
            total += len(cleaned_rows)

    if not clean:
        raise RuntimeError("Scholarships snapshot validation failed: no valid records.")

    return {"catalog": clean, "record_count": total}


def _validate_eligibility_thresholds_snapshot(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("Eligibility-threshold snapshot must be a JSON object.")

    program_min_gpa_raw = payload.get("program_min_gpa")
    english_requirements_raw = payload.get("english_requirements")
    default_english_raw = payload.get("default_english_requirement")

    clean_program_min_gpa: dict[str, float] = {}
    if isinstance(program_min_gpa_raw, dict):
        for program, val in program_min_gpa_raw.items():
            try:
                clean_program_min_gpa[str(program).strip()] = float(val)
            except Exception:
                continue

    clean_english_requirements: dict[str, dict[str, float]] = {}
    if isinstance(english_requirements_raw, dict):
        for country, thresholds in english_requirements_raw.items():
            if not isinstance(thresholds, dict):
                continue
            row: dict[str, float] = {}
            for key in ("ielts", "toefl", "pte"):
                try:
                    row[key] = float(thresholds.get(key))
                except Exception:
                    continue
            if row:
                clean_english_requirements[str(country).strip()] = row

    clean_default_english: dict[str, float] = {}
    if isinstance(default_english_raw, dict):
        for key in ("ielts", "toefl", "pte"):
            try:
                clean_default_english[key] = float(default_english_raw.get(key))
            except Exception:
                continue

    if not (clean_program_min_gpa or clean_english_requirements or clean_default_english):
        raise RuntimeError("Eligibility-threshold snapshot validation failed: no valid records.")

    return {
        "program_min_gpa": clean_program_min_gpa,
        "english_requirements": clean_english_requirements,
        "default_english_requirement": clean_default_english,
        "record_count": len(clean_program_min_gpa) + len(clean_english_requirements),
    }


def _fetch_policy_payload_from_url(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=8) as response:
        return json.loads(response.read().decode("utf-8"))


def _upsert_policy_snapshot(policy_key: str, data: dict[str, Any], source: str, confidence: float) -> None:
    SessionLocal = _require_session_local()
    with SessionLocal() as session:
        row = session.get(DBPolicySnapshot, policy_key)
        payload_text = json.dumps(data, ensure_ascii=False)
        if row is None:
            row = DBPolicySnapshot(
                policy_key=policy_key,
                data=payload_text,
                source=source,
                confidence=f"{confidence:.2f}",
            )
            session.add(row)
        else:
            row.data = payload_text
            row.source = source
            row.confidence = f"{confidence:.2f}"
        session.commit()


def _load_policy_snapshot(policy_key: str) -> dict[str, Any]:
    SessionLocal = _require_session_local()
    with SessionLocal() as session:
        row = session.get(DBPolicySnapshot, policy_key)
    if row is None:
        return {}

    try:
        payload = json.loads(row.data or "{}")
    except Exception:
        payload = {}

    updated_at = None
    if getattr(row, "updated_at", None) is not None:
        try:
            updated_at = row.updated_at.isoformat()
        except Exception:
            updated_at = str(row.updated_at)

    return {
        "policy_key": policy_key,
        "source": row.source,
        "confidence": row.confidence,
        "updated_at": updated_at,
        "data": payload,
    }


def _policy_snapshot_status(policy_key: str) -> dict[str, Any]:
    snap = _load_policy_snapshot(policy_key)
    if not snap:
        return {
            "policy_key": policy_key,
            "available": False,
            "source": None,
            "updated_at": None,
            "confidence": None,
        }
    return {
        "policy_key": policy_key,
        "available": True,
        "source": snap.get("source"),
        "updated_at": snap.get("updated_at"),
        "confidence": snap.get("confidence"),
    }


def _ingest_visa_risk_snapshot() -> None:
    if not POLICY_VISA_RISK_URL:
        print("[Policy] VISA risk ingest skipped: POLICY_VISA_RISK_URL is empty.")
        return

    try:
        raw_payload = _fetch_policy_payload_from_url(POLICY_VISA_RISK_URL)
        validated = _validate_visa_risk_snapshot(raw_payload)
        _upsert_policy_snapshot(
            policy_key="visa_risk_matrix",
            data=validated,
            source=POLICY_VISA_RISK_URL,
            confidence=POLICY_VISA_RISK_CONFIDENCE,
        )
        print(
            f"[Policy] Ingested visa_risk_matrix snapshot: {validated.get('record_count', 0)} entries from {POLICY_VISA_RISK_URL}"
        )
    except Exception as exc:
        print(f"[Policy] VISA risk ingest failed: {exc}")


def _ingest_living_costs_snapshot() -> None:
    if not POLICY_LIVING_COSTS_URL:
        print("[Policy] Living-cost ingest skipped: POLICY_LIVING_COSTS_URL is empty.")
        return
    try:
        raw_payload = _fetch_policy_payload_from_url(POLICY_LIVING_COSTS_URL)
        validated = _validate_living_costs_snapshot(raw_payload)
        _upsert_policy_snapshot(
            policy_key="living_costs",
            data=validated,
            source=POLICY_LIVING_COSTS_URL,
            confidence=POLICY_LIVING_COSTS_CONFIDENCE,
        )
        print(f"[Policy] Ingested living_costs snapshot: {validated.get('record_count', 0)} entries")
    except Exception as exc:
        print(f"[Policy] Living-cost ingest failed: {exc}")


def _ingest_scholarships_snapshot() -> None:
    if not POLICY_SCHOLARSHIPS_URL:
        print("[Policy] Scholarships ingest skipped: POLICY_SCHOLARSHIPS_URL is empty.")
        return
    try:
        raw_payload = _fetch_policy_payload_from_url(POLICY_SCHOLARSHIPS_URL)
        validated = _validate_scholarships_snapshot(raw_payload)
        _upsert_policy_snapshot(
            policy_key="scholarships",
            data=validated,
            source=POLICY_SCHOLARSHIPS_URL,
            confidence=POLICY_SCHOLARSHIPS_CONFIDENCE,
        )
        print(f"[Policy] Ingested scholarships snapshot: {validated.get('record_count', 0)} entries")
    except Exception as exc:
        print(f"[Policy] Scholarships ingest failed: {exc}")


def _ingest_eligibility_thresholds_snapshot() -> None:
    if not POLICY_ELIGIBILITY_THRESHOLDS_URL:
        print("[Policy] Eligibility-threshold ingest skipped: POLICY_ELIGIBILITY_THRESHOLDS_URL is empty.")
        return
    try:
        raw_payload = _fetch_policy_payload_from_url(POLICY_ELIGIBILITY_THRESHOLDS_URL)
        validated = _validate_eligibility_thresholds_snapshot(raw_payload)
        _upsert_policy_snapshot(
            policy_key="eligibility_thresholds",
            data=validated,
            source=POLICY_ELIGIBILITY_THRESHOLDS_URL,
            confidence=POLICY_ELIGIBILITY_THRESHOLDS_CONFIDENCE,
        )
        print(f"[Policy] Ingested eligibility_thresholds snapshot: {validated.get('record_count', 0)} entries")
    except Exception as exc:
        print(f"[Policy] Eligibility-threshold ingest failed: {exc}")


def _run_policy_ingestion_jobs() -> None:
    if not POLICY_INGEST_ON_STARTUP:
        print("[Policy] Startup ingestion disabled.")
        return
    _ingest_visa_risk_snapshot()
    _ingest_living_costs_snapshot()
    _ingest_scholarships_snapshot()
    _ingest_eligibility_thresholds_snapshot()


def _ensure_document_upload_schema() -> None:
    if _db_engine is None:
        return

    try:
        columns = {column["name"] for column in inspect(_db_engine).get_columns("document_uploads")}
    except Exception as exc:
        print(f"[DB] Could not inspect document_uploads schema: {exc}")
        return

    if "binary_data" in columns:
        return

    with _db_engine.begin() as connection:
        connection.execute(text("ALTER TABLE document_uploads ADD COLUMN binary_data BYTEA"))
    print("[DB] Added document_uploads.binary_data column.")


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


def _applications_index_path() -> Path:
    return APPLICATIONS_DIR / "applications.json"


def _load_applications() -> list[dict[str, Any]]:
    """Load all applications from the configured database."""
    SessionLocal = _require_session_local()
    with SessionLocal() as session:
        rows = session.query(DBApplication).all()
        result = []
        for row in rows:
            payload = _read_json_text(row.data, {})
            if isinstance(payload, dict):
                payload["application_id"] = row.application_id
                payload["user_id"] = row.user_id
                payload["status"] = row.status
                result.append(payload)
        return result


def _save_application(app_record: dict[str, Any]) -> None:
    """Upsert a single application record."""
    app_id = app_record.get("application_id", "")
    SessionLocal = _require_session_local()
    with SessionLocal() as session:
        row = session.get(DBApplication, app_id)
        data_str = json.dumps(app_record, ensure_ascii=False)
        if row is None:
            session.add(DBApplication(
                application_id=app_id,
                user_id=app_record.get("user_id", ""),
                status=app_record.get("status", "submitted"),
                data=data_str,
            ))
        else:
            row.status = app_record.get("status", row.status)
            row.data = data_str
        session.commit()


def _append_historical_outcome(app_record: dict[str, Any], status: str) -> None:
    """Append a real admissions outcome record for ML training (JSONL)."""
    if status not in {"accepted", "rejected"}:
        return

    user_id = app_record.get("user_id", "")
    state = _load_user_state_record(user_id) if user_id else {}
    profile = state.get("profile") if isinstance(state, dict) else {}
    if not isinstance(profile, dict):
        profile = {}
    elig = state.get("elig") if isinstance(state, dict) else {}
    if not isinstance(elig, dict):
        elig = {}

    gpa = elig.get("grade_point")
    try:
        gpa = float(gpa) if gpa is not None else None
    except Exception:
        gpa = None

    uni_data = app_record.get("university_data") if isinstance(app_record.get("university_data"), dict) else {}
    uni_min = uni_data.get("minGpa")
    try:
        uni_min = float(uni_min) if uni_min is not None else None
    except Exception:
        uni_min = None

    gpa_diff = (gpa - uni_min) if (gpa is not None and uni_min is not None) else None

    app_id = str(app_record.get("application_id", "")).strip()
    # Deduplicate by (application_id, admission_outcome)
    existing_keys: set[str] = set()
    if HISTORICAL_OUTCOMES_PATH.exists():
        try:
            with open(HISTORICAL_OUTCOMES_PATH, "r", encoding="utf-8") as handle:
                for line in handle:
                    raw = (line or "").strip()
                    if not raw:
                        continue
                    try:
                        row = json.loads(raw)
                    except Exception:
                        continue
                    if isinstance(row, dict):
                        key = f"{row.get('application_id','')}::{row.get('admission_outcome','')}"
                        existing_keys.add(key)
        except Exception:
            pass

    dedupe_key = f"{app_id}::{status}"
    if app_id and dedupe_key in existing_keys:
        return

    record = {
        "timestamp": _utc_now(),
        "application_id": app_id,
        "user_id": user_id,
        "country": app_record.get("country", ""),
        "university_id": app_record.get("university_id", ""),
        "university_name": app_record.get("university_name", ""),
        "program": app_record.get("program", "") or profile.get("program_interest", ""),
        "stream": profile.get("stream", ""),
        "gpa": gpa,
        "university_min_gpa": uni_min,
        "gpa_diff": gpa_diff,
        "qs_world": uni_data.get("qs"),
        "the_world": uni_data.get("the"),
        "tier_label": app_record.get("eligibility_tier") or elig.get("eligibility_tier", ""),
        "match_label": elig.get("tier_match", ""),
        "alignment_label": 1 if (elig.get("program_alignment", "").startswith("✅")) else None,
        "admission_outcome": status,
    }

    HISTORICAL_OUTCOMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORICAL_OUTCOMES_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_imported_historical_outcome(record: dict[str, Any]) -> bool:
    """Append imported historical outcome row with basic dedupe check."""
    status = str(record.get("admission_outcome", "")).strip().lower()
    if status not in {"accepted", "rejected"}:
        return False

    app_id = str(record.get("application_id", "")).strip()
    university = str(record.get("university_name", "")).strip().lower()
    program = str(record.get("program", "")).strip().lower()
    dedupe_key = f"{app_id}::{status}::{university}::{program}"

    existing_keys: set[str] = set()
    if HISTORICAL_OUTCOMES_PATH.exists():
        try:
            with open(HISTORICAL_OUTCOMES_PATH, "r", encoding="utf-8") as handle:
                for line in handle:
                    raw = (line or "").strip()
                    if not raw:
                        continue
                    try:
                        row = json.loads(raw)
                    except Exception:
                        continue
                    if isinstance(row, dict):
                        k = (
                            f"{str(row.get('application_id','')).strip()}::"
                            f"{str(row.get('admission_outcome','')).strip().lower()}::"
                            f"{str(row.get('university_name','')).strip().lower()}::"
                            f"{str(row.get('program','')).strip().lower()}"
                        )
                        existing_keys.add(k)
        except Exception:
            pass

    if dedupe_key in existing_keys:
        return False

    clean = dict(record)
    clean["timestamp"] = clean.get("timestamp") or _utc_now()
    HISTORICAL_OUTCOMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORICAL_OUTCOMES_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(clean, ensure_ascii=False) + "\n")
    return True


def _backfill_historical_outcomes_from_applications() -> int:
    """Backfill outcomes file from already-decided applications."""
    all_apps = _load_applications()
    count = 0
    for app in all_apps:
        status = str(app.get("status", "")).strip().lower()
        if status in {"accepted", "rejected"}:
            before_size = HISTORICAL_OUTCOMES_PATH.stat().st_size if HISTORICAL_OUTCOMES_PATH.exists() else 0
            _append_historical_outcome(app, status)
            after_size = HISTORICAL_OUTCOMES_PATH.stat().st_size if HISTORICAL_OUTCOMES_PATH.exists() else 0
            if after_size > before_size:
                count += 1
    return count


def _load_users() -> dict[str, dict[str, Any]]:
    SessionLocal = _require_session_local()
    with SessionLocal() as session:
        rows = session.query(DBUser).all()
        users: dict[str, dict[str, Any]] = {}
        for row in rows:
            payload = _read_json_text(row.data, {})
            users[row.email] = payload if isinstance(payload, dict) else {}
        return users


def _save_users(users: dict[str, dict[str, Any]]) -> None:
    SessionLocal = _require_session_local()
    with SessionLocal() as session:
        for email, payload in users.items():
            row = session.get(DBUser, email)
            data = json.dumps(payload if isinstance(payload, dict) else {}, ensure_ascii=False)
            if row is None:
                session.add(DBUser(email=email, data=data))
            else:
                row.data = data
        session.commit()


def _upsert_user_account(email_raw: str, password: str, name_raw: str, role: str) -> dict[str, Any] | None:
    email = _normalize_email(email_raw)
    name = (name_raw or role.title()).strip() or role.title()
    normalized_role = _normalize_user_role(role, strict=True)

    if not email or not password:
        return None

    _validate_password_policy(password)

    users = _load_users()
    existing = users.get(email, {}) if isinstance(users.get(email), dict) else {}
    users[email] = {
        **existing,
        "name": name,
        "email": email,
        "password_hash": _hash_password(password),
        "role": normalized_role,
        "updated_at": _utc_now(),
    }
    if "created_at" not in users[email]:
        users[email]["created_at"] = _utc_now()
    _save_users(users)
    return users[email]


def _bootstrap_user_from_env(email_raw: str, password: str, name_raw: str, role: str) -> bool:
    """Create or update one bootstrap user from environment variables."""
    return _upsert_user_account(email_raw, password, name_raw, role) is not None


_DEMO_SEED_ACCOUNTS = [
    ("admin@example.com",   "Admin@123",   "System Admin",   "admin"),
    ("advisor@example.com", "Advisor@123", "System Advisor", "advisor"),
    ("student@example.com", "Student@123", "Demo Student",   "student"),
]


def _auto_seed_demo_users_if_empty() -> None:
    """Seed demo accounts (admin/advisor/student) on first run when the DB has no users at all."""
    try:
        users = _load_users()
        if users:
            return  # DB already has users — leave it alone
        print("[Auth] No users found — seeding default demo accounts...")
        for email, password, name, role in _DEMO_SEED_ACCOUNTS:
            try:
                _upsert_user_account(email, password, name, role)
                print(f"[Auth]   ✓ {role}: {email}  password: {password}")
            except Exception as exc:
                print(f"[Auth]   ✗ Failed to seed {email}: {exc}")
        print("[Auth] Demo seed complete. Log in with admin@example.com / Admin@123")
    except Exception as exc:
        print(f"[Auth] Auto-seed skipped: {exc}")


def _bootstrap_admin_user_from_env() -> None:
    """Create or update one bootstrap admin user from environment variables."""
    created = _bootstrap_user_from_env(
        BOOTSTRAP_ADMIN_EMAIL,
        BOOTSTRAP_ADMIN_PASSWORD or "",
        BOOTSTRAP_ADMIN_NAME,
        "admin",
    )
    if created:
        print(f"[Auth] Bootstrap admin ensured: {_normalize_email(BOOTSTRAP_ADMIN_EMAIL)}")


def _bootstrap_advisor_user_from_env() -> None:
    """Create or update one bootstrap advisor user from environment variables."""
    created = _bootstrap_user_from_env(
        BOOTSTRAP_ADVISOR_EMAIL,
        BOOTSTRAP_ADVISOR_PASSWORD or "",
        BOOTSTRAP_ADVISOR_NAME,
        "advisor",
    )
    if created:
        print(f"[Auth] Bootstrap advisor ensured: {_normalize_email(BOOTSTRAP_ADVISOR_EMAIL)}")


def _load_sessions() -> dict[str, dict[str, Any]]:
    SessionLocal = _require_session_local()
    with SessionLocal() as session:
        rows = session.query(DBSession).all()
        sessions: dict[str, dict[str, Any]] = {}
        for row in rows:
            payload = _read_json_text(row.data, {})
            sessions[row.token] = payload if isinstance(payload, dict) else {}
        return sessions


def _save_sessions(sessions: dict[str, dict[str, Any]]) -> None:
    SessionLocal = _require_session_local()
    with SessionLocal() as session:
        for token, payload in sessions.items():
            row = session.get(DBSession, token)
            data = json.dumps(payload if isinstance(payload, dict) else {}, ensure_ascii=False)
            if row is None:
                session.add(DBSession(token=token, data=data))
            else:
                row.data = data
        session.commit()


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


def _parse_utc_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = str(value).strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(normalized)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _is_session_expired(session_payload: dict[str, Any]) -> bool:
    created_raw = session_payload.get("created_at")
    created_at = _parse_utc_timestamp(created_raw)
    if created_at is None:
        return True
    expires_at = created_at + timedelta(hours=max(1, SESSION_TTL_HOURS))
    return datetime.now(timezone.utc) >= expires_at


def _validate_password_policy(password: str) -> None:
    if len(password) < PASSWORD_MIN_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Password must be at least {PASSWORD_MIN_LENGTH} characters",
        )

    if not PASSWORD_REQUIRE_COMPLEXITY:
        return

    checks = {
        "lowercase": any(ch.islower() for ch in password),
        "uppercase": any(ch.isupper() for ch in password),
        "digit": any(ch.isdigit() for ch in password),
        "special": any(not ch.isalnum() for ch in password),
    }
    if not all(checks.values()):
        raise HTTPException(
            status_code=400,
            detail=(
                "Password must include uppercase, lowercase, number, and special character"
            ),
        )


def _login_rate_limit_key(email: str, client_host: str | None) -> str:
    return f"{email}|{client_host or 'unknown'}"


def _check_login_rate_limit(key: str) -> None:
    now = time.time()
    window = max(60, AUTH_WINDOW_SECONDS)
    max_attempts = max(3, AUTH_MAX_LOGIN_ATTEMPTS)
    attempts = [ts for ts in _LOGIN_ATTEMPTS.get(key, []) if now - ts <= window]
    _LOGIN_ATTEMPTS[key] = attempts
    if len(attempts) >= max_attempts:
        raise HTTPException(
            status_code=429,
            detail="Too many login attempts. Please try again later.",
        )


def _record_login_failure(key: str) -> None:
    now = time.time()
    window = max(60, AUTH_WINDOW_SECONDS)
    attempts = [ts for ts in _LOGIN_ATTEMPTS.get(key, []) if now - ts <= window]
    attempts.append(now)
    _LOGIN_ATTEMPTS[key] = attempts


def _clear_login_attempts(key: str) -> None:
    if key in _LOGIN_ATTEMPTS:
        del _LOGIN_ATTEMPTS[key]


def _issue_session_token(email: str) -> str:
    token = secrets.token_urlsafe(32)
    sessions = _load_sessions()
    created_at = _utc_now()
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=max(1, SESSION_TTL_HOURS))).isoformat()
    sessions[token] = {
        "email": email,
        "created_at": created_at,
        "expires_at": expires_at,
    }
    _save_sessions(sessions)
    return token


def _extract_bearer_token(authorization: str | None) -> str:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    return token.strip()


def _revoke_session_token(token: str) -> bool:
    sessions = _load_sessions()
    removed = sessions.pop(token, None) is not None
    if removed:
        _save_sessions(sessions)
    return removed


def _authenticate_token(authorization: str | None, *, required: bool = True) -> str | None:
    if not authorization:
        if required:
            raise HTTPException(status_code=401, detail="Authentication required")
        return None
    token_value = _extract_bearer_token(authorization)
    sessions = _load_sessions()
    session = sessions.get(token_value)
    if not session or not session.get("email"):
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if _is_session_expired(session):
        sessions.pop(token_value, None)
        _save_sessions(sessions)
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return _normalize_email(session["email"])


def _require_current_user(authorization: str | None = Header(default=None)) -> str:
    return _authenticate_token(authorization, required=True) or ""


def _require_role(authorization: str | None, allowed: set[str]) -> str:
    email = _authenticate_token(authorization, required=True) or ""
    users = _load_users()
    role = _normalize_user_role(users.get(email, {}).get("role"), strict=False)
    if role not in allowed:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    return email


def _require_admin(authorization: str | None = Header(default=None)) -> str:
    return _require_role(authorization, {"admin"})


def _require_advisor_or_admin(authorization: str | None = Header(default=None)) -> str:
    return _require_role(authorization, {"advisor", "admin"})


def _require_auth(authorization: str | None = Header(default=None)) -> str:
    """Any authenticated user (student, advisor, or admin)."""
    return _require_role(authorization, {"student", "advisor", "admin"})


def _ensure_user_access(user_id: str, current_user_email: str) -> str:
    normalized_user_id = _normalize_email(user_id)
    if not normalized_user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    if normalized_user_id != current_user_email:
        raise HTTPException(status_code=403, detail="You can only access your own data")
    return normalized_user_id


def _load_user_state_record(user_id: str) -> dict[str, Any]:
    SessionLocal = _require_session_local()
    with SessionLocal() as session:
        row = session.get(DBUserState, _safe_user_key(user_id))
        if row is None:
            return {}
        payload = _read_json_text(row.data, {})
        return payload if isinstance(payload, dict) else {}


def _save_user_state_record(user_id: str, state: dict[str, Any]) -> None:
    safe_id = _safe_user_key(user_id)
    SessionLocal = _require_session_local()
    with SessionLocal() as session:
        row = session.get(DBUserState, safe_id)
        data = json.dumps(state if isinstance(state, dict) else {}, ensure_ascii=False)
        if row is None:
            session.add(DBUserState(user_id=safe_id, data=data))
        else:
            row.data = data
        session.commit()


def _load_document_records(user_id: str) -> list[dict[str, Any]]:
    safe_id = _safe_user_key(user_id)
    SessionLocal = _require_session_local()
    with SessionLocal() as session:
        rows = session.query(DBDocumentUpload).filter(DBDocumentUpload.user_id == safe_id).all()
        records: list[dict[str, Any]] = []
        for row in rows:
            payload = _read_json_text(row.data, {})
            if isinstance(payload, dict):
                records.append(payload)
        return sorted(records, key=lambda item: item.get("stored_at", ""), reverse=True)


def _save_document_records(user_id: str, records: list[dict[str, Any]]) -> None:
    safe_id = _safe_user_key(user_id)
    clean_records = [item for item in records if isinstance(item, dict)]
    SessionLocal = _require_session_local()
    with SessionLocal() as session:
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


def _get_document_row(user_id: str, document_id: str) -> DBDocumentUpload:
    safe_id = _safe_user_key(user_id)
    SessionLocal = _require_session_local()
    with SessionLocal() as session:
        row = session.get(DBDocumentUpload, document_id)
        if row is None or row.user_id != safe_id:
            raise HTTPException(status_code=404, detail="Document not found")
        session.expunge(row)
        return row


def _upsert_document_record(user_id: str, record: dict[str, Any], file_bytes: bytes | None = None) -> None:
    safe_id = _safe_user_key(user_id)
    SessionLocal = _require_session_local()
    with SessionLocal() as session:
        row = session.get(DBDocumentUpload, record["document_id"])
        payload = json.dumps(record, ensure_ascii=False)
        if row is None:
            row = DBDocumentUpload(
                document_id=record["document_id"],
                user_id=safe_id,
                data=payload,
                binary_data=file_bytes,
            )
            session.add(row)
        else:
            row.user_id = safe_id
            row.data = payload
            if file_bytes is not None:
                row.binary_data = file_bytes
        session.commit()


def _delete_document_record(user_id: str, document_id: str) -> None:
    safe_id = _safe_user_key(user_id)
    SessionLocal = _require_session_local()
    with SessionLocal() as session:
        row = session.get(DBDocumentUpload, document_id)
        if row is None or row.user_id != safe_id:
            raise HTTPException(status_code=404, detail="Document not found")
        session.delete(row)
        session.commit()


def _backfill_document_blobs_from_disk() -> int:
    SessionLocal = _require_session_local()
    migrated = 0
    with SessionLocal() as session:
        rows = session.query(DBDocumentUpload).all()
        for row in rows:
            if row.binary_data:
                continue
            payload = _read_json_text(row.data, {})
            if not isinstance(payload, dict):
                continue
            legacy_name = payload.get("stored_name")
            if not legacy_name:
                continue
            legacy_path = _user_documents_path(row.user_id) / legacy_name
            if not legacy_path.exists():
                continue
            try:
                row.binary_data = legacy_path.read_bytes()
            except Exception as exc:
                print(f"[Documents] Failed to import legacy file {legacy_path}: {exc}")
                continue
            payload["storage_backend"] = "database"
            row.data = json.dumps(payload, ensure_ascii=False)
            migrated += 1
            try:
                legacy_path.unlink()
            except Exception:
                pass

        if migrated:
            session.commit()
    return migrated


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
    extraction_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    document_id = uuid4().hex
    safe_name = _sanitize_filename(filename)

    record = {
        "document_id": document_id,
        "filename": safe_name,
        "content_type": content_type or "application/octet-stream",
        "file_size": len(file_bytes),
        "stored_at": _utc_now(),
        "storage_backend": "database",
    }
    if extracted_data:
        record["document_type"] = extracted_data.get("document_type")
        record["extracted_data"] = extracted_data
    if extraction_meta:
        record["ocr_extraction"] = extraction_meta
        record["manual_review"] = extraction_meta.get("manual_review")

    _upsert_document_record(user_id, record, file_bytes)
    return record


def _find_document_record(user_id: str, document_id: str) -> dict[str, Any]:
    row = _get_document_row(user_id, document_id)
    payload = _read_json_text(row.data, {})
    if isinstance(payload, dict):
        return payload
    raise HTTPException(status_code=404, detail="Document not found")


def _get_document_content_bytes(user_id: str, document_id: str) -> tuple[dict[str, Any], bytes]:
    row = _get_document_row(user_id, document_id)
    payload = _read_json_text(row.data, {})
    if not isinstance(payload, dict):
        raise HTTPException(status_code=404, detail="Document not found")

    if row.binary_data:
        return payload, bytes(row.binary_data)

    legacy_name = payload.get("stored_name")
    if legacy_name:
        legacy_path = _user_documents_path(user_id) / legacy_name
        if legacy_path.exists():
            file_bytes = legacy_path.read_bytes()
            payload["storage_backend"] = "database"
            payload.pop("stored_name", None)
            _upsert_document_record(user_id, payload, file_bytes)
            try:
                legacy_path.unlink()
            except Exception:
                pass
            return payload, file_bytes

    raise HTTPException(status_code=404, detail="Document file is missing")


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
    safe_id = _safe_user_key(user_id)
    SessionLocal = _require_session_local()
    with SessionLocal() as session:
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


def _save_chat_record(user_id: str, record: dict[str, Any]) -> None:
    safe_id = _safe_user_key(user_id)
    payload = {
        "messages": record.get("messages", [])[-CHAT_HISTORY_LIMIT:],
        "agent_data": record.get("agent_data", {}),
    }
    SessionLocal = _require_session_local()
    with SessionLocal() as session:
        row = session.get(DBChatHistory, safe_id)
        data = json.dumps(payload, ensure_ascii=False)
        if row is None:
            session.add(DBChatHistory(user_id=safe_id, data=data))
        else:
            row.data = data
        session.commit()


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
    version="7.3",
    description=(
        "Dual OCR edition with Tesseract auto-detection and EasyOCR fallback. "
        "TF-IDF + NaiveBayes ML classifier. 9 Sri Lankan document types."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_origin_regex=CORS_ALLOW_ORIGIN_REGEX,
)

_ml: Pipeline = None


@app.on_event("startup")
def startup():
    global _ml, _OCR_ENGINE, _chatbot_agent, _rag_provider, _OCR_READINESS

    _initialize_database()

    _bootstrap_admin_user_from_env()
    _bootstrap_advisor_user_from_env()

    # Auto-seed demo accounts when the database is empty (first run or fresh DB)
    _auto_seed_demo_users_if_empty()

    try:
        migrated_docs = _backfill_document_blobs_from_disk()
        if migrated_docs:
            print(f"[Documents] Imported {migrated_docs} legacy on-disk document(s) into PostgreSQL.")
    except Exception as exc:
        print(f"[Documents] Legacy document import skipped: {exc}")

    try:
        backfilled = _backfill_historical_outcomes_from_applications()
        if backfilled:
            print(f"[Applications] Backfilled {backfilled} historical admission outcome(s).")
    except Exception as exc:
        print(f"[Applications] Historical outcomes backfill skipped: {exc}")

    _run_policy_ingestion_jobs()

    if ChatbotAgent:
        eligibility_agent = None
        financial_agent = None
        recommendation_agent = None

        eligibility_policy_snapshot = _load_policy_snapshot("eligibility_thresholds")
        financial_living_snapshot = _load_policy_snapshot("living_costs")
        financial_scholarship_snapshot = _load_policy_snapshot("scholarships")
        visa_policy_snapshot = _load_policy_snapshot("visa_risk_matrix")

        if EligibilityVerificationAgent:
            try:
                eligibility_data = eligibility_policy_snapshot.get("data") or {}
                eligibility_agent = EligibilityVerificationAgent(
                    program_min_gpa_snapshot=eligibility_data.get("program_min_gpa") or {},
                    english_requirements_snapshot=eligibility_data.get("english_requirements") or {},
                    default_english_requirement_snapshot=eligibility_data.get("default_english_requirement") or {},
                    policy_metadata={
                        "eligibility_thresholds": {
                            "source": eligibility_policy_snapshot.get("source", "unavailable"),
                            "updated_at": eligibility_policy_snapshot.get("updated_at", "unknown"),
                            "confidence": str(eligibility_policy_snapshot.get("confidence", "0.0")),
                        }
                    },
                )
            except Exception as exc:
                print(f"[Chat] Eligibility agent disabled: {exc}")

        if FinancialFeasibilityAgent:
            try:
                living_data = (financial_living_snapshot.get("data") or {}).get("costs") or {}
                scholarship_data = (financial_scholarship_snapshot.get("data") or {}).get("catalog") or {}
                financial_agent = FinancialFeasibilityAgent(
                    living_costs_snapshot=living_data,
                    scholarships_snapshot=scholarship_data,
                    policy_metadata={
                        "living_costs": {
                            "source": financial_living_snapshot.get("source", "unavailable"),
                            "updated_at": financial_living_snapshot.get("updated_at", "unknown"),
                            "confidence": str(financial_living_snapshot.get("confidence", "0.0")),
                        },
                        "scholarships": {
                            "source": financial_scholarship_snapshot.get("source", "unavailable"),
                            "updated_at": financial_scholarship_snapshot.get("updated_at", "unknown"),
                            "confidence": str(financial_scholarship_snapshot.get("confidence", "0.0")),
                        },
                    },
                )
            except Exception as exc:
                print(f"[Chat] Financial agent disabled: {exc}")

        if RecommendationAgent:
            try:
                recommendation_agent = RecommendationAgent(
                    visa_risk_snapshot=(visa_policy_snapshot.get("data") or {}).get("destinations", {}),
                    policy_metadata={
                        "visa_risk": {
                            "source": visa_policy_snapshot.get("source", "unavailable"),
                            "updated_at": visa_policy_snapshot.get("updated_at", "unknown"),
                            "confidence": str(visa_policy_snapshot.get("confidence", "0.0")),
                        }
                    },
                    disable_direct_visa_sources=True,
                )
            except Exception as exc:
                print(f"[Chat] Recommendation agent disabled: {exc}")

        rag_system = None
        if RAG_ENABLED and RAGSystem:
            try:
                rag_system = RAGSystem()
                _rag_provider = getattr(rag_system, "llm_provider", "none") or "none"
            except Exception as exc:
                rag_system = None
                _rag_provider = "none"
                print(f"[Chat] RAG init failed: {exc}")
        elif not RAG_ENABLED:
            _rag_provider = "disabled"
            print("[Chat] RAG initialization skipped (RAG_ENABLED=false).")

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

    # 2. Configure OCR engine based on OCR_ENGINE mode.
    if OCR_ENGINE_MODE == "easyocr" or USE_EASYOCR:
        try:
            import easyocr  # noqa: F401
            _OCR_ENGINE = "easyocr"
            print("[OCR] OCR_ENGINE=easyocr — using EasyOCR.")
        except ImportError:
            _OCR_ENGINE = None
            print("[OCR] EasyOCR not installed. Run: pip install easyocr")
    elif OCR_ENGINE_MODE == "tesseract":
        tess_ok = _configure_tesseract()
        if tess_ok:
            _OCR_ENGINE = "tesseract"
        else:
            _OCR_ENGINE = None
            print(
                "[OCR] WARNING: OCR_ENGINE=tesseract but Tesseract is unavailable!\n"
                "  Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki"
            )
    else:
        tess_ok = _configure_tesseract()
        if tess_ok:
            _OCR_ENGINE = "tesseract"
        else:
            try:
                import easyocr  # noqa: F401
                _OCR_ENGINE = "easyocr"
                print("[OCR] Tesseract unavailable — using EasyOCR fallback.")
            except ImportError:
                _OCR_ENGINE = None
                print(
                    "[OCR] WARNING: Neither Tesseract nor EasyOCR available!\n"
                    "  Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki\n"
                    "  OR run: pip install easyocr"
                )

    _OCR_READINESS = _collect_ocr_readiness()
    if OCR_STRICT_MODE and _OCR_ENGINE is None:
        hints = " | ".join(_OCR_READINESS.get("messages", []))
        raise RuntimeError(
            "OCR_STRICT_MODE=true and no OCR engine is available. "
            "Document ingestion cannot continue in degraded mode. "
            f"Fix installation and restart. Hints: {hints}"
        )

    print(
        f"[UniAssist v7.3] Ready | "
        f"Mode: {OCR_ENGINE_MODE} | "
        f"OCR: {_OCR_ENGINE or 'NONE — see warnings above'} | "
        f"ML: TF-IDF + NaiveBayes"
    )


@app.get("/health")
def health():
    policy_status = {
        "visa_risk_matrix": _policy_snapshot_status("visa_risk_matrix"),
        "living_costs": _policy_snapshot_status("living_costs"),
        "scholarships": _policy_snapshot_status("scholarships"),
        "eligibility_thresholds": _policy_snapshot_status("eligibility_thresholds"),
    }
    return {
        "status": "ok" if _OCR_ENGINE else "degraded",
        "version": "7.3",
        "db": _db_backend,
        "db_url_set": bool(DATABASE_URL),
        "db_strict_mode": DB_STRICT_MODE,
        "rag_enabled": RAG_ENABLED,
        "ocr_mode": OCR_ENGINE_MODE,
        "ocr_engine": _OCR_ENGINE or "none",
        "ocr_ready": bool(_OCR_ENGINE),
        "ocr_strict_mode": OCR_STRICT_MODE,
        "ocr_installation": _OCR_READINESS,
        "use_easyocr_flag": USE_EASYOCR,
        "rag_provider": _rag_provider,
        "ml_model": "TF-IDF bigrams + MultinomialNB",
        "doc_types": list(TRAINING_DATA.keys()),
        "tesseract_paths_checked": _WINDOWS_TESSERACT_PATHS if sys.platform == "win32" else [],
        "policy_snapshots": policy_status,
    }


@app.get("/ocr/readiness")
def ocr_readiness():
    return {
        "ocr_engine": _OCR_ENGINE or "none",
        "ocr_mode": OCR_ENGINE_MODE,
        "ocr_strict_mode": OCR_STRICT_MODE,
        "installation": _OCR_READINESS,
    }


@app.get("/metrics")
def get_metrics(authorization: str | None = Header(default=None)):
    if not METRICS_PUBLIC:
        _require_admin(authorization)
    if _metrics_collector is None:
        raise HTTPException(status_code=503, detail="Metrics collector is unavailable")
    return _metrics_collector.get_summary()


@app.get("/metrics/flows")
def get_metrics_flows(limit: int = 20, authorization: str | None = Header(default=None)):
    if not METRICS_PUBLIC:
        _require_admin(authorization)
    if _metrics_collector is None:
        raise HTTPException(status_code=503, detail="Metrics collector is unavailable")
    safe_limit = max(1, min(int(limit), 200))
    flows = _metrics_collector.get_recent_flows(limit=safe_limit)
    return {
        "flows": flows,
        "count": len(flows),
    }


@app.get("/universities")
def get_universities(country: str | None = None, current_user_email: str = Depends(_require_auth)):
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
    requested_role = _normalize_user_role(payload.role, strict=True)
    role = "student"
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")
    _validate_password_policy(password)

    users = _load_users()
    if email in users:
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    if requested_role != "student":
        if ALLOW_PRIVILEGED_SELF_REGISTRATION:
            role = requested_role
        else:
            role = "student"

    users[email] = {
        "name": name,
        "email": email,
        "password_hash": _hash_password(password),
        "role": role,
        "created_at": _utc_now(),
    }
    _save_users(users)
    return {
        "success": True,
        "user": {
            "name": name,
            "email": email,
            "role": role,
        },
    }


@app.get("/auth/config")
def auth_config():
    """Return safe auth-related feature flags for frontend UX."""
    return {
        "allow_privileged_self_registration": ALLOW_PRIVILEGED_SELF_REGISTRATION,
        "password_min_length": PASSWORD_MIN_LENGTH,
        "password_require_complexity": PASSWORD_REQUIRE_COMPLEXITY,
    }


@app.get("/auth/me")
def auth_me(current_user_email: str = Depends(_require_current_user)):
    users = _load_users()
    user = users.get(current_user_email, {})
    return {
        "user": {
            "name": user.get("name") or current_user_email.split("@")[0],
            "email": current_user_email,
            "role": _normalize_user_role(user.get("role")),
        }
    }


@app.post("/auth/logout")
def logout(authorization: str | None = Header(default=None)):
    _authenticate_token(authorization, required=True)
    token_value = _extract_bearer_token(authorization)
    _revoke_session_token(token_value)
    return {"success": True}


@app.get("/auth/check-email/{email:path}")
def check_email(email: str):
    """Return whether an email address has a registered account (no sensitive data exposed)."""
    normalized = _normalize_email(email)
    if not normalized:
        raise HTTPException(status_code=400, detail="Email is required")
    users = _load_users()
    registered = normalized in users
    has_hash = bool(users.get(normalized, {}).get("password_hash")) if registered else False
    return {"email": normalized, "registered": registered, "login_ready": has_hash}


@app.post("/auth/login")
def login(payload: LoginPayload, request: Request):
    email = _normalize_email(payload.email)
    password = payload.password or ""
    login_key = _login_rate_limit_key(email, getattr(request.client, "host", None))
    _check_login_rate_limit(login_key)

    users = _load_users()
    user = users.get(email)

    # Server-side diagnostic prints (visible in uvicorn console)
    if not user:
        print(f"[Auth] Login failed — email not found in DB: {email!r} | users_loaded={len(users)}")
        _record_login_failure(login_key)
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not user.get("password_hash"):
        print(f"[Auth] Login failed — password_hash missing for: {email!r}")
        _record_login_failure(login_key)
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not _verify_password(password, user["password_hash"]):
        print(f"[Auth] Login failed — wrong password for: {email!r}")
        _record_login_failure(login_key)
        raise HTTPException(status_code=401, detail="Invalid email or password")

    _clear_login_attempts(login_key)
    token = _issue_session_token(email)
    role = _normalize_user_role(user.get("role"))
    print(f"[Auth] Login success: {email!r} role={role!r}")
    return {
        "token": token,
        "user": {
            "name": user.get("name") or email.split("@")[0],
            "email": email,
            "role": role,
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
    record, file_bytes = _get_document_content_bytes(current_user_email, document_id)
    return Response(
        content=file_bytes,
        media_type=record.get("content_type") or "application/octet-stream",
        headers={
            "Content-Disposition": f'inline; filename="{record.get("filename") or document_id}"',
        },
    )


@app.delete("/documents/{document_id}")
def delete_document(document_id: str, current_user_email: str = Depends(_require_current_user)):
    deleted_record = _find_document_record(current_user_email, document_id)
    _delete_document_record(current_user_email, document_id)

    legacy_name = deleted_record.get("stored_name")
    if legacy_name:
        legacy_path = _user_documents_path(current_user_email) / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    return {
        "success": True,
        "document_id": document_id,
    }


@app.patch("/documents/{document_id}/corrections")
def apply_document_corrections(
    document_id: str,
    payload: OCRManualCorrectionPayload,
    current_user_email: str = Depends(_require_current_user),
):
    corrected_fields = payload.corrected_fields or {}
    if not isinstance(corrected_fields, dict) or not corrected_fields:
        raise HTTPException(status_code=400, detail="corrected_fields must be a non-empty object")

    existing_record, file_bytes = _get_document_content_bytes(current_user_email, document_id)
    existing_data = existing_record.get("extracted_data") or {}
    if not isinstance(existing_data, dict):
        existing_data = {}

    merged = {**existing_data, **corrected_fields}
    doc_type_label = merged.get("document_type") or existing_record.get("document_type")
    doc_type_key = merged.get("doc_type_key") or _TYPE_LABEL_TO_KEY.get(str(doc_type_label), "")
    if doc_type_key not in TRAINING_DATA:
        raise HTTPException(
            status_code=400,
            detail="Cannot apply corrections because document type is unknown. Re-run OCR first.",
        )

    normalized = _normalize_fields(doc_type_key, merged)
    normalized["document_type"] = _TYPE_MAP.get(doc_type_key, doc_type_label or doc_type_key)
    validated_fields, validation_issues, parse_confidence = _validate_parsed_fields(doc_type_key, normalized)

    ocr_meta = existing_record.get("ocr_extraction") or {}
    ocr_conf = float(ocr_meta.get("ocr_confidence") or 0.0)
    ml_conf = float(ocr_meta.get("ml_confidence") or 0.0)
    field_confidence, missing_field_reasons = _field_diagnostics(doc_type_key, validated_fields, ocr_conf, ml_conf)
    confidence = _score_with_validation(doc_type_key, validated_fields, ocr_conf, ml_conf, parse_confidence)
    manual_review = _build_manual_review_block(confidence, missing_field_reasons, validation_issues)

    correction_event = {
        "corrected_at": _utc_now(),
        "corrected_by": current_user_email,
        "corrected_keys": sorted(corrected_fields.keys()),
        "reviewer_note": payload.reviewer_note,
    }

    existing_record["document_type"] = validated_fields.get("document_type")
    existing_record["extracted_data"] = validated_fields
    existing_record["manual_review"] = manual_review
    existing_record["ocr_extraction"] = {
        **ocr_meta,
        "field_confidence": field_confidence,
        "missing_field_reasons": missing_field_reasons,
        "validation_issues": validation_issues,
        "parse_confidence": parse_confidence,
        "confidence": confidence,
        "manual_review": manual_review,
    }
    existing_record.setdefault("manual_corrections", [])
    if isinstance(existing_record["manual_corrections"], list):
        existing_record["manual_corrections"].append(correction_event)
    existing_record["updated_at"] = _utc_now()

    _upsert_document_record(current_user_email, existing_record, file_bytes)

    return {
        "success": True,
        "document_id": document_id,
        "data": validated_fields,
        "confidence": confidence,
        "parse_confidence": parse_confidence,
        "validation_issues": validation_issues,
        "missing_field_reasons": missing_field_reasons,
        "manual_review": manual_review,
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
    SessionLocal = _require_session_local()
    with SessionLocal() as session:
        row = session.get(DBChatHistory, _safe_user_key(normalized_user_id))
        if row is not None:
            session.delete(row)
            session.commit()
    return {
        "success": True,
        "user_id": normalized_user_id,
        "messages": [],
        "agent_data": {},
    }


@app.post("/chat/respond")
def chat_respond(payload: ChatRespondPayload, current_user_email: str = Depends(_require_auth)):
    t0 = time.time()
    message = (payload.user_message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="user_message is required")

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
        agent_data = result.get("agent_data") or {}
        recommendation_meta = ((agent_data.get("agent_results") or {}).get("recommendation") or {}).get("policy_metadata") or {}
        if recommendation_meta:
            agent_data["policy_sources"] = recommendation_meta
        duration_ms = (time.time() - t0) * 1000
        if _metrics_collector is not None:
            intent = str(result.get("intent") or "unknown")
            agent_calls = [a for a in (result.get("agent_calls") or []) if isinstance(a, str)]
            _metrics_collector.record_intent(
                detected_intent=intent,
                confidence=0.6,
                user_message=message,
                keywords=[],
            )
            per_agent_duration = duration_ms / max(1, len(agent_calls))
            for agent_name in agent_calls:
                _metrics_collector.record_agent(
                    agent_name=agent_name,
                    duration_ms=per_agent_duration,
                    success=True,
                )
            rag_used = bool(((result.get("agent_data") or {}).get("agent_results") or {}).get("rag"))
            if rag_used:
                _metrics_collector.record_rag(
                    query=message,
                    source_count=0,
                    relevance_level="unknown",
                    llm_provider=_rag_provider,
                    duration_ms=duration_ms,
                )
        return {
            "response": response_text,
            "intent": result.get("intent"),
            "actions": result.get("actions", []),
            "agent_data": agent_data,
            "source": "backend_agent",
        }

    raise HTTPException(status_code=503, detail="Chat agent is unavailable. Check server startup logs.")


@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    doc_type: str = Form(default="auto"),
    current_user_email: str = Depends(_require_auth),
):
    if _OCR_ENGINE is None:
        hints = " | ".join((_OCR_READINESS or {}).get("messages", []))
        raise HTTPException(
            status_code=503,
            detail=(
                "No OCR engine available. "
                "Install Tesseract (https://github.com/UB-Mannheim/tesseract/wiki) "
                "or install EasyOCR (pip install easyocr). "
                f"Readiness hints: {hints}"
            ),
        )

    file_bytes = await file.read()
    suffix = os.path.splitext(file.filename or "doc.jpg")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        t0 = time.time()
        text, ocr_conf, ocr_preset = _run_ocr(tmp_path, doc_type_hint=doc_type)
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

        fields_raw = _extract(detected, text)
        fields = _normalize_fields(detected, fields_raw)
        fields["document_type"] = _TYPE_MAP.get(detected, detected)
        fields, validation_issues, parse_confidence = _validate_parsed_fields(detected, fields)
        field_confidence, missing_field_reasons = _field_diagnostics(
            detected, fields, ocr_conf, final_conf
        )
        conf = _score_with_validation(detected, fields, ocr_conf, final_conf, parse_confidence)
        manual_review = _build_manual_review_block(conf, missing_field_reasons, validation_issues)

        response_payload = {
            "success":                True,
            "data":                   fields,
            "schema_version":         "2.0",
            "confidence":             conf,
            "ml_confidence":          round(final_conf, 3),
            "classification_method":  method,
            "ocr_confidence":         round(ocr_conf, 3),
            "ocr_engine":             _OCR_ENGINE,
            "ocr_preset":             ocr_preset,
            "ocr_time_sec":           ocr_time,
            "parse_confidence":       parse_confidence,
            "validation_issues":      validation_issues,
            "field_confidence":       field_confidence,
            "missing_field_reasons":  missing_field_reasons,
            "manual_review":          manual_review,
            "requires_manual_review": manual_review["required"],
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
        if missing_field_reasons:
            response_payload["warnings"] = [
                *response_payload["warnings"],
                f"Missing required fields: {', '.join(sorted(missing_field_reasons.keys()))}",
            ]
        if validation_issues:
            response_payload["warnings"] = [
                *response_payload["warnings"],
                f"Validation issues: {', '.join(validation_issues)}",
            ]
        response_payload["document"] = _store_user_document(
            current_user_email,
            filename=file.filename,
            content_type=file.content_type,
            file_bytes=file_bytes,
            extracted_data=response_payload["data"],
            extraction_meta={
                "confidence": response_payload["confidence"],
                "ml_confidence": response_payload["ml_confidence"],
                "ocr_confidence": response_payload["ocr_confidence"],
                "parse_confidence": response_payload["parse_confidence"],
                "classification_method": response_payload["classification_method"],
                "ocr_engine": response_payload["ocr_engine"],
                "ocr_preset": response_payload["ocr_preset"],
                "ocr_time_sec": response_payload["ocr_time_sec"],
                "field_confidence": response_payload["field_confidence"],
                "missing_field_reasons": response_payload["missing_field_reasons"],
                "validation_issues": response_payload["validation_issues"],
                "manual_review": response_payload["manual_review"],
            },
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
# ADVISOR / ADMIN ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/advisor/students")
def advisor_list_students(current_user_email: str = Depends(_require_advisor_or_admin)):
    """Return all student accounts with their live application state and document counts."""
    users = _load_users()
    result = []
    for email, u in users.items():
        role = _normalize_user_role(u.get("role"), strict=False)
        if role != "student":
            continue
        state = _load_user_state_record(email)
        docs = _load_document_records(email)
        if not isinstance(docs, list):
            docs = []

        raw_step = state.get("step", 1) if isinstance(state, dict) else 1
        try:
            step = int(raw_step)
        except Exception:
            step = 1
        step = max(1, min(step, 4))

        doc_count = len(docs)
        completion = min(int((step / 4) * 100), 100)
        if step >= 4:
            status = "completed"
        elif step >= 2:
            status = "in-progress"
        else:
            status = "started"
        quality = "excellent" if doc_count >= 3 else ("good" if doc_count >= 1 else "warning")
        elig = state.get("elig") if isinstance(state, dict) else None
        if elig is None:
            eligibility = "pending"
        elif isinstance(elig, dict) and bool(elig.get("eligible")):
            eligibility = "eligible"
        else:
            eligibility = "ineligible"
        result.append({
            "email": email,
            "name": u.get("name") or email,
            "status": status,
            "completion": completion,
            "lastUpdated": (u.get("created_at") or "")[:10],
            "dataQuality": quality,
            "documents": doc_count,
            "eligibility": eligibility,
            "step": step,
        })
    result.sort(key=lambda x: x["name"].lower())
    return {"students": result}


@app.get("/admin/stats")
def admin_stats(current_user_email: str = Depends(_require_admin)):
    """Return live system statistics derived from real user and application data."""
    users = _load_users()
    role_counts: dict[str, int] = {"student": 0, "advisor": 0, "admin": 0}
    completed = 0
    pending = 0
    for email, u in users.items():
        role = _normalize_user_role(u.get("role"), strict=False)
        role_counts[role] += 1
        if role == "student":
            state = _load_user_state_record(email)
            step = state.get("step", 1) if isinstance(state, dict) else 1
            if step >= 4:
                completed += 1
            elif step >= 2:
                pending += 1
    n_students = role_counts["student"]
    return {
        "totalUsers": len(users),
        "totalStudents": n_students,
        "totalAdvisors": role_counts["advisor"],
        "totalAdmins": role_counts["admin"],
        "completedApplications": completed,
        "pendingApplications": pending,
        "dataQualityScore": round((completed / n_students * 100) if n_students else 0.0, 1),
    }


@app.get("/admin/policies/{policy_key}")
def admin_get_policy_snapshot(policy_key: str, current_user_email: str = Depends(_require_admin)):
    snapshot = _load_policy_snapshot(policy_key)
    if not snapshot:
        raise HTTPException(status_code=404, detail=f"Policy snapshot not found: {policy_key}")
    return snapshot


@app.post("/admin/policies/ingest/visa-risk")
def admin_ingest_visa_risk_policy(current_user_email: str = Depends(_require_admin)):
    _ingest_visa_risk_snapshot()
    snapshot = _load_policy_snapshot("visa_risk_matrix")
    if not snapshot:
        raise HTTPException(status_code=500, detail="Visa risk snapshot ingestion did not produce a persisted record")
    return {
        "success": True,
        "policy_key": "visa_risk_matrix",
        "source": snapshot.get("source"),
        "updated_at": snapshot.get("updated_at"),
        "confidence": snapshot.get("confidence"),
    }


@app.post("/admin/policies/ingest/all")
def admin_ingest_all_policies(current_user_email: str = Depends(_require_admin)):
    _run_policy_ingestion_jobs()
    return {
        "success": True,
        "snapshots": {
            "visa_risk_matrix": _policy_snapshot_status("visa_risk_matrix"),
            "living_costs": _policy_snapshot_status("living_costs"),
            "scholarships": _policy_snapshot_status("scholarships"),
            "eligibility_thresholds": _policy_snapshot_status("eligibility_thresholds"),
        },
    }


@app.post("/admin/seed-defaults")
def admin_seed_defaults(current_user_email: str = Depends(_require_admin)):
    """(Re-)seed demo accounts. Existing accounts are updated; new ones created."""
    results = []
    for email, password, name, role in _DEMO_SEED_ACCOUNTS:
        try:
            record = _upsert_user_account(email, password, name, role)
            results.append({"email": email, "role": role, "status": "ok" if record else "skipped"})
        except Exception as exc:
            results.append({"email": email, "role": role, "status": f"error: {exc}"})
    return {"seeded": results}


@app.get("/admin/users")
def admin_list_users(current_user_email: str = Depends(_require_admin)):
    """Return the full user list for admin management."""
    users = _load_users()
    result = [
        {
            "email": email,
            "name": u.get("name") or email,
            "role": _normalize_user_role(u.get("role"), strict=False),
            "created": (u.get("created_at") or "")[:10],
            "login_ready": bool(u.get("password_hash")),
        }
        for email, u in users.items()
    ]
    result.sort(key=lambda x: x["created"], reverse=True)
    return {"users": result, "total": len(result)}


@app.get("/admin/audit")
def admin_audit_events(current_user_email: str = Depends(_require_admin)):
    """Return a best-effort audit stream derived from live persisted records."""
    events: list[dict[str, Any]] = []

    users = _load_users()
    for email, user in users.items():
        created_at = user.get("created_at")
        if created_at:
            events.append({
                "timestamp": created_at,
                "action": "USER_REGISTER",
                "actor": email,
                "target": "AUTH",
                "details": f"Registered as {_normalize_user_role(user.get('role'), strict=False)}",
            })

    sessions = _load_sessions()
    for _, session_data in sessions.items():
        email = _normalize_email(session_data.get("email", ""))
        created_at = session_data.get("created_at")
        if email and created_at:
            events.append({
                "timestamp": created_at,
                "action": "LOGIN",
                "actor": email,
                "target": "AUTH",
                "details": "Logged in successfully",
            })

    for email in users.keys():
        for doc in _load_document_records(email):
            timestamp = doc.get("stored_at")
            if not timestamp:
                continue
            doc_name = doc.get("filename") or "document"
            events.append({
                "timestamp": timestamp,
                "action": "DOCUMENT_UPLOAD",
                "actor": email,
                "target": email,
                "details": f"Uploaded {doc_name}",
            })

    events.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
    return {"events": events[:200]}


@app.patch("/admin/users/{user_email}/role")
def admin_update_user_role(
    user_email: str,
    payload: UpdateRolePayload,
    current_user_email: str = Depends(_require_admin),
):
    """Change a user's role. Admins cannot change their own role."""
    normalized = _normalize_email(user_email)
    if normalized == current_user_email:
        raise HTTPException(status_code=400, detail="You cannot change your own role")
    users = _load_users()
    if normalized not in users:
        raise HTTPException(status_code=404, detail="User not found")
    new_role = _normalize_user_role(payload.role, strict=True)
    users[normalized]["role"] = new_role
    users[normalized]["updated_at"] = _utc_now()
    _save_users(users)
    return {"success": True, "email": normalized, "role": new_role}


@app.post("/admin/historical-outcomes/import")
def admin_import_historical_outcomes(
    payload: ImportHistoricalOutcomesPayload,
    current_user_email: str = Depends(_require_admin),
):
    """Import real historical admissions outcomes for model training."""
    records = payload.records or []
    if not records:
        raise HTTPException(status_code=400, detail="No records provided")

    imported = 0
    skipped = 0
    for item in records:
        ok = _append_imported_historical_outcome(item.model_dump())
        if ok:
            imported += 1
        else:
            skipped += 1
    return {
        "success": True,
        "imported": imported,
        "skipped": skipped,
        "path": str(HISTORICAL_OUTCOMES_PATH),
    }


@app.get("/admin/historical-outcomes/stats")
def admin_historical_outcomes_stats(
    current_user_email: str = Depends(_require_admin),
):
    total = 0
    accepted = 0
    rejected = 0
    if HISTORICAL_OUTCOMES_PATH.exists():
        try:
            with open(HISTORICAL_OUTCOMES_PATH, "r", encoding="utf-8") as handle:
                for line in handle:
                    raw = (line or "").strip()
                    if not raw:
                        continue
                    try:
                        row = json.loads(raw)
                    except Exception:
                        continue
                    if not isinstance(row, dict):
                        continue
                    total += 1
                    outcome = str(row.get("admission_outcome", "")).strip().lower()
                    if outcome == "accepted":
                        accepted += 1
                    elif outcome == "rejected":
                        rejected += 1
        except Exception:
            pass
    return {
        "total": total,
        "accepted": accepted,
        "rejected": rejected,
        "path": str(HISTORICAL_OUTCOMES_PATH),
    }


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATIONS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/applications")
def submit_application(
    payload: SubmitApplicationPayload,
    current_user_email: str = Depends(_require_auth),
):
    """Student submits an application to a university."""
    import uuid
    # Prevent exact duplicate submissions for same user + university + program
    all_apps = _load_applications()
    existing = [
        a for a in all_apps
        if a.get("user_id") == current_user_email
        and a.get("university_name") == payload.university_name
        and a.get("program") == payload.program
        and a.get("status") not in ("withdrawn",)
    ]
    if existing:
        raise HTTPException(status_code=409, detail="Application already submitted for this university and program.")

    users = _load_users()
    user_info = users.get(current_user_email, {})

    app_record = {
        "application_id": str(uuid.uuid4()),
        "user_id": current_user_email,
        "user_name": user_info.get("name", ""),
        "university_name": payload.university_name,
        "university_id": payload.university_id or "",
        "program": payload.program,
        "country": payload.country,
        "eligibility_tier": payload.eligibility_tier or "",
        "grade_point": payload.grade_point or 0.0,
        "notes": payload.notes or "",
        "advisor_notes": "",
        "university_data": payload.university_data or {},
        "status": "submitted",
        "submitted_at": _utc_now(),
        "updated_at": _utc_now(),
    }
    _save_application(app_record)
    return {"success": True, "application": app_record}


@app.get("/applications")
def list_applications(
    current_user_email: str = Depends(_require_auth),
):
    """List applications. Students see their own; advisors/admins see all."""
    users = _load_users()
    role = (users.get(current_user_email, {}).get("role") or "student").lower()
    all_apps = _load_applications()
    if role in ("advisor", "admin"):
        return {"applications": sorted(all_apps, key=lambda a: a.get("submitted_at", ""), reverse=True)}
    own = [a for a in all_apps if a.get("user_id") == current_user_email]
    return {"applications": sorted(own, key=lambda a: a.get("submitted_at", ""), reverse=True)}


@app.get("/applications/{application_id}")
def get_application(
    application_id: str,
    current_user_email: str = Depends(_require_auth),
):
    all_apps = _load_applications()
    app_record = next((a for a in all_apps if a.get("application_id") == application_id), None)
    if not app_record:
        raise HTTPException(status_code=404, detail="Application not found")
    users = _load_users()
    role = (users.get(current_user_email, {}).get("role") or "student").lower()
    if role not in ("advisor", "admin") and app_record.get("user_id") != current_user_email:
        raise HTTPException(status_code=403, detail="Access denied")
    return app_record


@app.patch("/applications/{application_id}/status")
def update_application_status(
    application_id: str,
    payload: UpdateApplicationStatusPayload,
    current_user_email: str = Depends(_require_auth),
):
    """Advisor/admin updates status; student can only withdraw their own."""
    all_apps = _load_applications()
    app_record = next((a for a in all_apps if a.get("application_id") == application_id), None)
    if not app_record:
        raise HTTPException(status_code=404, detail="Application not found")
    previous_status = str(app_record.get("status", "")).strip().lower()
    users = _load_users()
    role = (users.get(current_user_email, {}).get("role") or "student").lower()
    if role not in ("advisor", "admin"):
        # Students may only withdraw their own
        if app_record.get("user_id") != current_user_email:
            raise HTTPException(status_code=403, detail="Access denied")
        if payload.status != "withdrawn":
            raise HTTPException(status_code=403, detail="Students can only withdraw applications")
    app_record["status"] = payload.status
    app_record["updated_at"] = _utc_now()
    if payload.advisor_notes is not None:
        app_record["advisor_notes"] = payload.advisor_notes
    _save_application(app_record)

    new_status = str(payload.status).strip().lower()
    if new_status in {"accepted", "rejected"} and new_status != previous_status:
        try:
            _append_historical_outcome(app_record, new_status)
        except Exception as exc:
            print(f"[Applications] Could not append historical outcome for {application_id}: {exc}")

    return {"success": True, "application": app_record}


@app.delete("/applications/{application_id}")
def delete_application(
    application_id: str,
    current_user_email: str = Depends(_require_auth),
):
    """Hard-delete (admin only) or withdraw (student/advisor)."""
    all_apps = _load_applications()
    app_record = next((a for a in all_apps if a.get("application_id") == application_id), None)
    if not app_record:
        raise HTTPException(status_code=404, detail="Application not found")
    users = _load_users()
    role = (users.get(current_user_email, {}).get("role") or "student").lower()
    if role == "admin":
        app_record["status"] = "withdrawn"
        app_record["updated_at"] = _utc_now()
        _save_application(app_record)
    elif app_record.get("user_id") == current_user_email:
        app_record["status"] = "withdrawn"
        app_record["updated_at"] = _utc_now()
        _save_application(app_record)
    else:
        raise HTTPException(status_code=403, detail="Access denied")
    return {"success": True}


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
        text, ocr_conf, _ocr_preset = _run_ocr(image_file)
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