"""
UniAssist - EasyOCR + ML Document Processor
Optimized for low-performance laptops (CPU mode, minimal RAM)

KEY OPTIMISATIONS vs your heavy api_server.py:
  ✓ EasyOCR in CPU-only mode (gpu=False)          → saves 1+ GB VRAM
  ✓ Single NB classifier instead of 3-model ensemble → 10× faster
  ✓ One-time model load, pickle persistence          → no retraining on restart
  ✓ Image downscale cap at 1200px                   → less RAM per image
  ✓ Single OCR pass (no inverted retry by default)  → 2× faster OCR
  ✓ Lazy EasyOCR init (loads only on first call)    → fast server startup

Install (one time):
  pip install easyocr scikit-learn opencv-python-headless numpy Pillow fastapi uvicorn python-multipart

Run API:
  uvicorn ml_document_processor_easyocr:app --host 0.0.0.0 --port 8000

Test single image:
  python ml_document_processor_easyocr.py path/to/doc.jpg
"""

import os
import re
import pickle
import random
import numpy as np
import cv2
from PIL import Image

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# ─────────────────────────────────────────────────────────────────────────────
# DEMO / SAMPLE TRAINING DATA
# 25 samples × 9 doc types = 225 raw sentences
# After augmentation → 1,125 samples  (5× free data)
# Vocabulary sourced from your existing codebase files
# ─────────────────────────────────────────────────────────────────────────────

TRAINING_DATA = {

    # ── A/L Results ──────────────────────────────────────────────────────────
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
        "advanced level examination august 2022 results certified",
        "ministry of education sri lanka examination certificate gce",
        "combined maths physics biology grade pass al",
        "gce advanced level 2023 result sheet candidate number",
        "examination index number subject grade obtained al",
        "aggregate score stream district island rank al university",
        "bio science stream al examination result gce",
        "mathematics stream al grade sheet 2022 result",
        "arts stream advanced level result certificate gce",
        "technology stream al examination result gce department",
        "examination results candidate number school code al",
        "department examinations al result colombo kandy galle 2023",
        "higher education university selection al z score cutoff",
        "subject wise marks grade advanced level results sheet",
        "al examination 2021 2022 2023 result slip certified copy",
        "general certificate education advanced level result sri lanka",
    ],

    # ── Bachelor's Degree ─────────────────────────────────────────────────────
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
        "eastern university vavuniya campus bachelor awarded ceremony",
        "sabaragamuwa university bachelor degree certificate awarded",
        "open university of sri lanka bachelor degree programme",
        "rajarata university bachelor of science award faculty",
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
    ],

    # ── Master's Degree ───────────────────────────────────────────────────────
    "master": [
        "master of science msc university of colombo postgraduate",
        "master of business administration mba postgraduate degree",
        "master of engineering postgraduate degree university faculty",
        "msc information technology university of moratuwa postgraduate",
        "postgraduate degree master programme faculty university",
        "master of arts ma postgraduate university of kelaniya awarded",
        "mba degree awarded university of sri jayewardenepura postgraduate",
        "master of philosophy mphil research degree university",
        "postgraduate diploma master degree university peradeniya",
        "master degree gpa grade point average postgraduate distinction",
        "master of public administration mpa postgraduate degree",
        "msc computer science postgraduate programme university awarded",
        "master degree awarded distinction merit pass university",
        "postgraduate studies master programme two year full time",
        "master of education med postgraduate degree university",
        "msc data science analytics postgraduate university awarded",
        "master of finance mfin postgraduate degree awarded university",
        "master degree thesis research dissertation university postgraduate",
        "postgraduate master degree certificate awarded ceremony",
        "mba executive programme part time master degree university",
        "master of nursing science postgraduate degree university awarded",
        "msc electrical engineering postgraduate degree university faculty",
        "master of social sciences mssci postgraduate awarded",
        "master degree programme colombo moratuwa peradeniya university",
        "postgraduate master awarded distinction merit pass university",
    ],

    # ── Diploma ───────────────────────────────────────────────────────────────
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

    # ── IELTS ─────────────────────────────────────────────────────────────────
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

    # ── TOEFL ─────────────────────────────────────────────────────────────────
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
        "TOEFL score send institution university score report ETS",
        "ETS TOEFL iBT test score date registration number result",
        "speaking writing reading listening section score TOEFL result",
        "TOEFL score report candidate name date birth nationality",
        "internet based TOEFL iBT official score report 2023 ETS",
        "ETS toefl total score 100 110 institution report result",
        "toefl reading 24 listening 26 speaking 22 writing 25 total 97",
        "TOEFL score valid two years test date ETS report result",
        "test english foreign language toefl ibt score report ets result",
    ],

    # ── PTE ───────────────────────────────────────────────────────────────────
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

    # ── Passport ──────────────────────────────────────────────────────────────
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

    # ── Financial Statement ───────────────────────────────────────────────────
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
# AUGMENTATION  — 5× data, zero extra libraries
# ─────────────────────────────────────────────────────────────────────────────

def _augment(text: str) -> list[str]:
    words = text.split()
    variants = []

    # 1. Lowercase (simulates OCR normalised output)
    variants.append(text.lower())

    # 2. Random partial uppercase (simulates scanned-doc caps)
    variants.append(" ".join(w.upper() if random.random() > 0.55 else w for w in words))

    # 3. Adjacent word swap (simulates OCR word-order noise)
    if len(words) > 3:
        i = random.randint(0, len(words) - 2)
        sw = words[:]
        sw[i], sw[i + 1] = sw[i + 1], sw[i]
        variants.append(" ".join(sw))
    else:
        variants.append(text)

    # 4. Drop 1-2 words (simulates OCR missing tokens)
    if len(words) > 4:
        drop = set(random.sample(range(len(words)), min(2, len(words) // 5)))
        variants.append(" ".join(w for i, w in enumerate(words) if i not in drop))
    else:
        variants.append(text.lower())

    return variants


def build_corpus():
    texts, labels = [], []
    for label, samples in TRAINING_DATA.items():
        for s in samples:
            texts.append(s)
            labels.append(label)
            for aug in _augment(s):
                texts.append(aug)
                labels.append(label)
    return texts, labels


# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED FALLBACK  (keyword scoring)
# Used when ML confidence < CONFIDENCE_THRESHOLD
# ─────────────────────────────────────────────────────────────────────────────

_KEYWORDS = {
    "alevel":    ["advanced level","al ","gce","z score","combined maths","stream",
                  "biology","physics","chemistry","department of examinations",
                  "candidate index","district rank","island rank","aggregate"],
    "bachelor":  ["bachelor","bsc","beng","bba","llb","mbbs","undergraduate",
                  "first class","second class","honours","convocation","faculty of"],
    "master":    ["master","msc","mba","mphil","postgraduate","post graduate",
                  "dissertation","thesis","distinction merit"],
    "diploma":   ["diploma","nibm","hnd","nvq","naita","vta",
                  "technical college","institute of technology","vocational"],
    "ielts":     ["ielts","band score","british council","idp","test report form",
                  "listening reading writing speaking","overall band"],
    "toefl":     ["toefl","ets","educational testing service","ibt",
                  "internet based","toefl score"],
    "pte":       ["pte","pearson","communicative skills","enabling skills",
                  "oral fluency","pearson test of english"],
    "passport":  ["passport","nationality","lka","date of expiry","date of issue",
                  "immigration","emigration","mrz","p<<lka","biometric","travel document"],
    "financial": ["bank statement","account number","closing balance","opening balance",
                  "transaction","debit","credit","bank of ceylon","peoples bank",
                  "commercial bank","hatton","sampath","fixed deposit","certified"],
}


def _rule_classify(text: str) -> tuple[str, float]:
    t = text.lower()
    scores = {lbl: sum(1 for kw in kws if kw in t) / len(kws)
              for lbl, kws in _KEYWORDS.items()}
    best = max(scores, key=scores.get)
    conf = min(scores[best] * 3, 1.0)
    return (best, conf) if conf > 0 else ("unknown", 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# ML MODEL — TF-IDF (bigrams) + Multinomial Naive Bayes
# Trains in < 1 second, uses ~5 MB RAM for the model
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH          = "sl_doc_classifier_easyocr.pkl"
CONFIDENCE_THRESHOLD = 0.40   # below this → rule-based fallback kicks in


def train_model(verbose: bool = True) -> Pipeline:
    texts, labels = build_corpus()

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),          # unigrams + bigrams
            max_features=8000,           # keeps model tiny
            sublinear_tf=True,           # log TF — better for short docs
            min_df=1,
            token_pattern=r"[a-zA-Z0-9]{2,}",
        )),
        ("nb", MultinomialNB(alpha=0.3)),  # sharper than default alpha=1
    ])

    if verbose:
        cv = cross_val_score(pipeline, texts, labels, cv=5, scoring="accuracy")
        print(f"[ML] 5-fold CV accuracy: {cv.mean():.1%} ± {cv.std():.1%}")
        print(f"[ML] Training on {len(texts)} samples | {len(TRAINING_DATA)} classes ...")

    pipeline.fit(texts, labels)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    if verbose:
        print(f"[ML] Model saved → {MODEL_PATH}")

    return pipeline


def load_or_train() -> Pipeline:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            mdl = pickle.load(f)
        print(f"[ML] Loaded model from {MODEL_PATH}")
        return mdl
    print("[ML] No saved model — training now ...")
    return train_model()


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PRE-PROCESSING  (lightweight, CPU-only)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(image_path: str) -> np.ndarray:
    """
    Prepare image for EasyOCR.
    Returns BGR numpy array sized ≤ 1200px wide.
    (EasyOCR accepts BGR arrays directly.)
    """
    img = cv2.imread(image_path)
    if img is None:
        pil = Image.open(image_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]

    # 1. Downscale to max 1200px wide — CRITICAL for low RAM
    #    (EasyOCR internally upscales, so we cap before handing over)
    max_w = 1200
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    elif w < 800:
        scale = 800 / w
        img = cv2.resize(img, (800, int(h * scale)), interpolation=cv2.INTER_LANCZOS4)

    h, w = img.shape[:2]

    # 2. Auto-rotate landscape → portrait
    if w > h:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # 3. Grayscale + CLAHE (improves faded / low-contrast docs)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 4. Gentle denoise (bilateral — preserves edges better than Gaussian)
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    # 5. Convert back to BGR (EasyOCR expects colour or grayscale-as-BGR)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# EASY OCR WRAPPER  — CPU mode, lazy init, English only
# ─────────────────────────────────────────────────────────────────────────────

_easyocr_reader = None   # singleton — loaded once, reused forever


def _get_reader():
    """
    Lazy-load EasyOCR reader in CPU mode.
    First call: ~3–5 sec to load model weights (~100 MB).
    Subsequent calls: instant (cached singleton).

    LOW-PERFORMANCE SETTINGS:
      gpu=False          → no CUDA needed, uses ~350 MB RAM
      model_storage_directory → cache weights locally
      download_enabled=True  → auto-downloads on first run
    """
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        print("[EasyOCR] Loading model (first time ~5 sec) ...")
        _easyocr_reader = easyocr.Reader(
            lang_list=["en"],          # English only → smallest model
            gpu=False,                 # CPU mode — KEY for low-spec laptops
            model_storage_directory=os.path.join(
                os.path.expanduser("~"), ".EasyOCR", "model"
            ),
            download_enabled=True,
            verbose=False,
        )
        print("[EasyOCR] Ready.")
    return _easyocr_reader


def run_easyocr(image_path: str) -> tuple[str, float]:
    """
    Run EasyOCR on an image.
    Returns (text, avg_confidence 0-1).

    PERFORMANCE NOTES for low-spec laptops:
      - A4 doc at 1200px width: ~4–8 sec on old dual-core CPU
      - Set detail=0 for speed (no bounding boxes, just text)
      - paragraph=True merges nearby text → cleaner output
    """
    img = preprocess(image_path)
    reader = _get_reader()

    results = reader.readtext(
        img,
        detail=1,           # need confidence scores
        paragraph=False,    # keep individual detections for confidence calc
        batch_size=1,       # low RAM: process one batch at a time
        workers=0,          # single-threaded — safer on low-spec
        beamWidth=3,        # reduced beam search → faster, small accuracy trade
    )

    # results = [(bbox, text, confidence), ...]
    words, confs = [], []
    for (_bbox, text, conf) in results:
        text = text.strip()
        if text and conf > 0.25:   # threshold: ignore very low-conf tokens
            words.append(text)
            confs.append(conf)

    full_text = " ".join(words)
    avg_conf  = float(np.mean(confs)) if confs else 0.0

    return full_text, avg_conf


# ─────────────────────────────────────────────────────────────────────────────
# OCR ERROR CORRECTOR  (Sri Lankan document common mis-reads)
# ─────────────────────────────────────────────────────────────────────────────

# Character confusion map  (OCR swaps these)
_CHAR_MAP = {
    "0": "O", "O": "0",    # zero ↔ letter O
    "1": "l", "l": "1",    # one  ↔ lowercase L
    "S": "5", "5": "S",    # S    ↔ 5
    "I": "1", "B": "8",    # I    ↔ 1, B ↔ 8
}

# Word-level fixes for common Sri Lankan document phrases
_WORD_FIXES = {
    "examlnation": "examination",
    "certlficate": "certificate",
    "unlversity":  "university",
    "passporl":    "passport",
    "natlonality": "nationality",
    "llstening":   "listening",
    "wrlting":     "writing",
    "speaklng":    "speaking",
    "readlng":     "reading",
    "candldate":   "candidate",
    "dlstrict":    "district",
    "agregate":    "aggregate",
    "zscore":      "z score",
    "bandsoore":   "band score",
    "balence":     "balance",
    "statment":    "statement",
    "certifled":   "certified",
    "postgratuate":"postgraduate",
    "bachelar":    "bachelor",
    "batchelor":   "bachelor",
    "mastar":      "master",
    "diplama":     "diploma",
    "ielts":       "IELTS",
    "toefl":       "TOEFL",
}


def correct_ocr(text: str) -> str:
    """Fix common EasyOCR errors in Sri Lankan documents."""
    words = text.split()
    fixed = []
    for w in words:
        wl = w.lower()
        if wl in _WORD_FIXES:
            fixed.append(_WORD_FIXES[wl])
        else:
            fixed.append(w)
    return " ".join(fixed)


# ─────────────────────────────────────────────────────────────────────────────
# FIELD EXTRACTORS  (regex per document type)
# ─────────────────────────────────────────────────────────────────────────────

def _find(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def _find_all(pattern: str, text: str) -> list[str]:
    return re.findall(pattern, text, re.IGNORECASE)


def extract_fields(doc_type: str, text: str) -> dict:
    t = text.lower()
    if doc_type == "alevel":
        return {
            "index_number": _find(r"\b(\d{7})\b", text),
            "year":         _find(r"(20\d{2})", text),
            "subjects":     _find_all(r"(combined maths?|physics|chemistry|biology|"
                                      r"economics|accounting|geography|ict)", t),
            "grades":       _find_all(r"\b([ABCSF])\b", text.upper()),
            "z_score":      _find(r"z[\s\-]?score[\s:]+([0-9.]+)", t),
        }
    elif doc_type in ("bachelor", "master", "diploma"):
        return {
            "name":       _find(r"(?:certify that|awarded to|conferred upon)\s+([A-Z\s]+)", text),
            "degree":     _find(r"(bachelor|master|diploma)[^.\n]{0,60}", t),
            "university": _find(r"university of [\w\s]+|[\w\s]+ university", t),
            "year":       _find(r"(20\d{2})", text),
            "class":      _find(r"(first class|second class|upper second|lower second|"
                                r"distinction|merit|pass)", t),
            "gpa":        _find(r"gpa[\s:]+([0-9.]+)", t),
        }
    elif doc_type == "ielts":
        return {
            "overall":    _find(r"overall[\s\w]{0,20}?([0-9]\.[05])", t),
            "listening":  _find(r"listening[\s:]+([0-9]\.[05])", t),
            "reading":    _find(r"reading[\s:]+([0-9]\.[05])", t),
            "writing":    _find(r"writing[\s:]+([0-9]\.[05])", t),
            "speaking":   _find(r"speaking[\s:]+([0-9]\.[05])", t),
            "test_date":  _find(r"(\d{1,2}[\s/\-]\w+[\s/\-]20\d{2})", text),
            "trf_number": _find(r"(?:trf|reference)[\s:]+([A-Z0-9\-]+)", t),
        }
    elif doc_type == "toefl":
        return {
            "total":      _find(r"total[\s:]+(\d{2,3})", t),
            "reading":    _find(r"reading[\s:]+(\d{1,2})", t),
            "listening":  _find(r"listening[\s:]+(\d{1,2})", t),
            "speaking":   _find(r"speaking[\s:]+(\d{1,2})", t),
            "writing":    _find(r"writing[\s:]+(\d{1,2})", t),
            "test_date":  _find(r"(\d{1,2}[\s/\-]\w+[\s/\-]20\d{2})", text),
        }
    elif doc_type == "pte":
        return {
            "overall":   _find(r"overall[\s:]+(\d{2,3})", t),
            "listening": _find(r"listening[\s:]+(\d{2,3})", t),
            "reading":   _find(r"reading[\s:]+(\d{2,3})", t),
            "writing":   _find(r"writing[\s:]+(\d{2,3})", t),
            "speaking":  _find(r"speaking[\s:]+(\d{2,3})", t),
            "test_date": _find(r"(\d{1,2}[\s/\-]\w+[\s/\-]20\d{2})", text),
        }
    elif doc_type == "passport":
        return {
            "surname":     _find(r"surname[\s:]+([A-Z]+)", text),
            "given_names": _find(r"given\s*names?[\s:]+([A-Z\s]+)", text),
            "passport_no": _find(r"\b([A-Z]\d{7})\b", text),
            "nationality": _find(r"nationality[\s:]+([A-Z\s]+)", text),
            "dob":         _find(r"(?:date of birth|dob)[\s:]+([0-9/\-\s\w]+)", t),
            "expiry":      _find(r"(?:date of expiry|expiry)[\s:]+([0-9/\-\s\w]+)", t),
            "mrz":         _find(r"(P<<LKA[A-Z<]+)", text),
        }
    elif doc_type == "financial":
        return {
            "bank_name":   _find(r"(bank of ceylon|peoples bank|commercial bank|"
                                 r"hatton national|sampath|nations trust|dfcc|seylan|ndb)", t),
            "account_no":  _find(r"account\s*(?:number|no)[\s:]+([0-9\-]+)", t),
            "closing_bal": _find(r"(?:closing|available|current)\s*balance[\s:]*"
                                 r"(?:lkr|rs\.?)?[\s]*([0-9,]+(?:\.\d{2})?)", t),
            "currency":    _find(r"\b(LKR|USD|GBP|AUD|SGD)\b", text),
            "period":      _find(r"(?:statement period|from)[\s:]+([0-9/\-\s\w]+)", t),
        }
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE SCORER
# ─────────────────────────────────────────────────────────────────────────────

_FIELD_WEIGHTS = {
    "alevel":    {"index_number":0.3,"year":0.2,"subjects":0.3,"grades":0.2},
    "bachelor":  {"name":0.2,"degree":0.3,"university":0.3,"year":0.2},
    "master":    {"name":0.2,"degree":0.3,"university":0.3,"year":0.2},
    "diploma":   {"degree":0.4,"year":0.3,"university":0.3},
    "ielts":     {"overall":0.4,"listening":0.15,"reading":0.15,"writing":0.15,"speaking":0.15},
    "toefl":     {"total":0.4,"reading":0.15,"listening":0.15,"speaking":0.15,"writing":0.15},
    "pte":       {"overall":0.4,"listening":0.15,"reading":0.15,"writing":0.15,"speaking":0.15},
    "passport":  {"passport_no":0.3,"surname":0.2,"given_names":0.2,"expiry":0.3},
    "financial": {"bank_name":0.2,"account_no":0.3,"closing_bal":0.5},
}


def overall_confidence(doc_type, fields, ocr_conf, ml_conf) -> float:
    weights = _FIELD_WEIGHTS.get(doc_type, {})
    field_score = sum(
        w for f, w in weights.items()
        if fields.get(f) and (fields[f] if isinstance(fields[f], str) else len(fields[f]) > 0)
    )
    return round(min(field_score * 0.50 + ocr_conf * 0.25 + ml_conf * 0.25, 1.0), 3)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

class UniAssistProcessor:
    """
    Full pipeline:  image → EasyOCR (CPU) → ML classify → fields → result
    Memory budget:  ~350 MB (EasyOCR model) + ~5 MB (ML model) + image buffers
    Minimum laptop: 2 GB RAM, any dual-core CPU (2012+), no GPU needed
    """

    def __init__(self):
        self.ml_model = load_or_train()
        # EasyOCR reader is lazy — loads on first image (saves startup RAM)
        print("[UniAssist] Processor ready  |  OCR: EasyOCR CPU  |  ML: TF-IDF + NaiveBayes")

    # ── Process a document image ──────────────────────────────────────────────
    def process_image(self, image_path: str) -> dict:
        """
        Main entry point.  Returns structured JSON-serialisable dict.
        """
        # 1. OCR
        raw_text, ocr_conf = run_easyocr(image_path)
        if not raw_text.strip():
            return {"error": "No text extracted — check image quality.", "doc_type": "unknown"}

        # 2. Correct common OCR errors
        text = correct_ocr(raw_text)

        # 3. ML classify
        ml_proba  = self.ml_model.predict_proba([text])[0]
        ml_best   = int(np.argmax(ml_proba))
        ml_label  = self.ml_model.classes_[ml_best]
        ml_conf   = float(ml_proba[ml_best])

        # 4. Fallback if ML is uncertain
        if ml_conf < CONFIDENCE_THRESHOLD:
            rb_label, rb_conf = _rule_classify(text)
            if rb_conf > ml_conf:
                doc_type, final_conf, method = rb_label, rb_conf, "rule_based"
            else:
                doc_type, final_conf, method = ml_label, ml_conf, "ml_low_conf"
        else:
            doc_type, final_conf, method = ml_label, ml_conf, "ml"

        # 5. Field extraction
        fields = extract_fields(doc_type, text)

        # 6. Holistic confidence
        oc = overall_confidence(doc_type, fields, ocr_conf, final_conf)

        return {
            "doc_type":           doc_type,
            "ml_confidence":      round(final_conf, 3),
            "overall_confidence": oc,
            "method":             method,
            "ocr_confidence":     round(ocr_conf, 3),
            "fields":             fields,
            "ocr_text_preview":   text[:500],
        }

    # ── Classify raw text (manual entry path) ────────────────────────────────
    def classify_text(self, text: str) -> dict:
        ml_proba = self.ml_model.predict_proba([text])[0]
        best     = int(np.argmax(ml_proba))
        label    = self.ml_model.classes_[best]
        conf     = float(ml_proba[best])
        method   = "ml" if conf >= CONFIDENCE_THRESHOLD else "rule_based"
        if method == "rule_based":
            label, conf = _rule_classify(text)
        return {
            "doc_type":   label,
            "confidence": round(conf, 3),
            "method":     method,
            "fields":     extract_fields(label, text),
        }

    # ── Retrain if you add more samples ──────────────────────────────────────
    def retrain(self):
        print("[ML] Retraining ...")
        self.ml_model = train_model(verbose=True)


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI  — same /health and /ocr endpoints as your api_server.py
# ─────────────────────────────────────────────────────────────────────────────

def _make_app():
    try:
        from fastapi import FastAPI, File, UploadFile, Form
        from fastapi.middleware.cors import CORSMiddleware
        import tempfile
    except ImportError:
        return None

    _app = FastAPI(
        title="UniAssist OCR API — Lightweight EasyOCR",
        description="CPU-only EasyOCR + TF-IDF/NaiveBayes. Low RAM. No GPU.",
        version="2.0",
    )
    _app.add_middleware(CORSMiddleware, allow_origins=["*"],
                        allow_methods=["*"], allow_headers=["*"])

    _proc = UniAssistProcessor()

    @_app.get("/health")
    def health():
        return {
            "status":  "ok",
            "ocr":     "EasyOCR CPU-only (gpu=False)",
            "ml":      "TF-IDF bigrams + MultinomialNB",
            "classes": list(TRAINING_DATA.keys()),
            "ram_est": "~400 MB",
        }

    @_app.post("/ocr")
    async def ocr_endpoint(
        file:     UploadFile = File(...),
        doc_type: str        = Form(default="auto"),
    ):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        try:
            return _proc.process_image(tmp_path)
        finally:
            os.unlink(tmp_path)

    @_app.post("/classify-text")
    async def classify_text_endpoint(text: str = Form(...)):
        return _proc.classify_text(text)

    return _app


app = _make_app()   # uvicorn picks this up automatically


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json

    print("=" * 55)
    print("  UniAssist — EasyOCR + ML Lightweight Processor")
    print("=" * 55)

    if len(sys.argv) > 1:
        proc = UniAssistProcessor()
        result = proc.process_image(sys.argv[1])
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # Show training stats + train model
        texts, labels = build_corpus()
        from collections import Counter
        counts = Counter(labels)
        print(f"\nTraining corpus:")
        for lbl in sorted(TRAINING_DATA):
            raw = len(TRAINING_DATA[lbl])
            aug = counts[lbl]
            print(f"  {lbl:12s}  {raw} raw → {aug} augmented")
        print(f"\nTotal: {len(texts)} samples | {len(TRAINING_DATA)} classes")
        print()
        train_model(verbose=True)
        print()
        print("Usage:")
        print("  python ml_document_processor_easyocr.py <image.jpg>")
        print("  uvicorn ml_document_processor_easyocr:app --port 8000")