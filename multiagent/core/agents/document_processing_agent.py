"""
UniAssist Document Processing Agent  —  Updated Edition
=========================================================
WHAT CHANGED FROM YOUR ORIGINAL document_processing_agent.py:
  ✗ REMOVED  Rule-based keyword-only document type detection
             → ✓ REPLACED with TF-IDF + NaiveBayes ML classifier
               (same model trained in api_server.py, loaded via pickle)
  ✗ REMOVED  EasyOCR with default gpu settings
             → ✓ EasyOCR with gpu=False (CPU mode, low RAM)
  ✗ REMOVED  Basic preprocessing (resize + CLAHE only)
             → ✓ Added bilateral denoise + HoughLines deskew
  ✓ KEPT     All dataclasses (IELTSScore, TOEFLScore, PTEScore, etc.)
  ✓ KEPT     ManualEntryProcessor (manual entry path unchanged)
  ✓ KEPT     All validation logic in dataclasses
  ✓ KEPT     DocumentProcessingAgent.manual_entry() method
  ✓ KEPT     Same output structure (OCR and manual entry produce identical dicts)
  ✓ ADDED    ML-only classification: top-2 comparison when confidence < 0.40
  ✓ ADDED    Pickle model load (shares model with api_server.py)

Install:
  pip install easyocr opencv-python-headless numpy scikit-learn Pillow
"""

from __future__ import annotations
import os
import re
import pickle
import random
import numpy as np
import cv2
from PIL import Image
from dataclasses import dataclass, field, asdict
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# SHARED ML MODEL
# Loads the pickle saved by api_server.py.
# If not found, trains a minimal model inline so this file works standalone.
# ─────────────────────────────────────────────────────────────────────────────

_SHARED_MODEL_PATH = "uniassist_classifier.pkl"   # shared with api_server.py
_FALLBACK_MODEL_PATH = "agent_classifier.pkl"      # standalone fallback

# Minimal training data for standalone use (mirrors api_server.py dataset)
_MINI_TRAIN = {
    "alevel":    ["sri lanka advanced level examination results gce al","z score district rank island rank al examination",
                  "combined mathematics physics chemistry biology grade","candidate index number al results department examinations",
                  "general certificate education advanced level result"],
    "bachelor":  ["bachelor of science degree university awarded","undergraduate degree bsc beng faculty engineering university",
                  "first class honours bachelor degree convocation","bachelor degree gpa cumulative grade point average",
                  "university colombo moratuwa bachelor awarded ceremony"],
    "master":    ["master of science msc postgraduate university","mba postgraduate degree awarded distinction merit",
                  "master degree thesis dissertation research university","postgraduate programme two year master degree awarded",
                  "msc information technology postgraduate university moratuwa"],
    "diploma":   ["diploma information technology NIBM certificate awarded","higher national diploma HND computing business",
                  "national vocational qualification NVQ diploma","technical college vocational training diploma award",
                  "diploma programme completed certificate awarded institute"],
    "ielts":     ["IELTS overall band score 7.0 listening reading writing speaking","british council IDP IELTS test report form",
                  "ielts academic band 6.5 7.5 test report form","IELTS score valid two years british council",
                  "overall band listening reading writing speaking ielts"],
    "toefl":     ["TOEFL iBT total score ETS educational testing service","toefl score report reading listening speaking writing",
                  "ETS TOEFL internet based test total 100","toefl ibt score report valid two years ets",
                  "reading listening speaking writing score TOEFL ETS"],
    "pte":       ["pearson PTE academic overall score communicative skills","PTE score report listening reading writing speaking",
                  "pearson test english PTE academic score 65","pte academic score valid two years pearson",
                  "enabling skills communicative skills PTE pearson score"],
    "passport":  ["republic sri lanka passport nationality travel document","passport number date expiry nationality LKA holder",
                  "immigration emigration department passport biometric","P<<LKA surname given names nationality mrz",
                  "passport photo page personal details nationality"],
    "financial": ["bank statement account number closing balance LKR","commercial bank peoples bank certified statement",
                  "transaction history debit credit balance bank statement","hatton national bank sampath account statement certified",
                  "financial proof funds bank certified manager stamp"],
}

def _mini_augment(text):
    words = text.split()
    return [text.lower(),
            " ".join(w.upper() if random.random()>0.6 else w for w in words)]

def _train_mini():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    texts, labels = [], []
    for lbl, samples in _MINI_TRAIN.items():
        for s in samples:
            texts.append(s); labels.append(lbl)
            for aug in _mini_augment(s):
                texts.append(aug); labels.append(lbl)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=5000,
                                   sublinear_tf=True, min_df=1,
                                   token_pattern=r"[a-zA-Z0-9]{2,}")),
        ("nb",    MultinomialNB(alpha=0.3)),
    ])
    pipe.fit(texts, labels)
    with open(_FALLBACK_MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)
    return pipe

def _load_ml_model():
    # Prefer the full model trained by api_server.py
    for path in (_SHARED_MODEL_PATH, _FALLBACK_MODEL_PATH):
        if os.path.exists(path):
            with open(path, "rb") as f:
                model = pickle.load(f)
            print(f"[Agent] ML model loaded from {path}")
            return model
    print("[Agent] No saved model found — training mini model ...")
    return _train_mini()

_ML_MODEL = None   # lazy load on first use

def _get_ml_model():
    global _ML_MODEL
    if _ML_MODEL is None:
        _ML_MODEL = _load_ml_model()
    return _ML_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# ML-ONLY CLASSIFY
# ─────────────────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.40

def classify_document(text: str) -> tuple[str, float, str]:
    """
    Returns (doc_type, confidence, method).
    Uses ML only: top-1 prediction, or top-2 when top-1 confidence is low.
    """
    model   = _get_ml_model()
    proba   = model.predict_proba([text])[0]
    best    = int(np.argmax(proba))
    ml_lbl  = model.classes_[best]
    ml_conf = float(proba[best])

    if ml_conf >= CONFIDENCE_THRESHOLD:
        return ml_lbl, ml_conf, "ml"

    # Low confidence: compare top-1 vs top-2 ML probabilities
    sorted_idx   = np.argsort(proba)[::-1]
    second_conf  = float(proba[sorted_idx[1]])
    if second_conf > ml_conf * 0.9:
        return model.classes_[sorted_idx[1]], second_conf, "ml_low_conf_top2"
    return ml_lbl, ml_conf, "ml_low_conf"


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASSES  (kept identical to your original — same structure)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ALevelResult:
    index_number: Optional[str] = None
    year:         Optional[str] = None
    subjects:     list[str]     = field(default_factory=list)
    grades:       list[str]     = field(default_factory=list)
    z_score:      Optional[str] = None
    stream:       Optional[str] = None

    def validate(self):
        valid_grades = set("ABCSF")
        self.grades  = [g for g in self.grades if g in valid_grades]
        return self

@dataclass
class DegreeResult:
    name:       Optional[str] = None
    degree:     Optional[str] = None
    university: Optional[str] = None
    year:       Optional[str] = None
    degree_class: Optional[str] = None
    gpa:        Optional[str] = None

    def validate(self):
        if self.gpa:
            try:
                gpa = float(self.gpa)
                if not (0.0 <= gpa <= 4.0):
                    self.gpa = None
            except ValueError:
                self.gpa = None
        return self

@dataclass
class IELTSScore:
    overall:    Optional[float] = None
    listening:  Optional[float] = None
    reading:    Optional[float] = None
    writing:    Optional[float] = None
    speaking:   Optional[float] = None
    test_date:  Optional[str]   = None
    trf_number: Optional[str]   = None

    def validate(self):
        for attr in ("overall","listening","reading","writing","speaking"):
            val = getattr(self, attr)
            if val is not None:
                try:
                    v = float(val)
                    # Snap to nearest valid IELTS band (0, 0.5, 1.0 ... 9.0)
                    snapped = round(v * 2) / 2
                    setattr(self, attr, snapped if 0 <= snapped <= 9 else None)
                except (TypeError, ValueError):
                    setattr(self, attr, None)
        # Cross-validate: section average should be close to overall
        parts = [self.listening, self.reading, self.writing, self.speaking]
        if self.overall and all(p is not None for p in parts):
            avg = sum(parts) / 4
            if abs(avg - self.overall) > 1.5:
                self.overall = round(avg * 2) / 2   # correct overall from parts
        return self

@dataclass
class TOEFLScore:
    total:      Optional[int] = None
    reading:    Optional[int] = None
    listening:  Optional[int] = None
    speaking:   Optional[int] = None
    writing:    Optional[int] = None
    test_date:  Optional[str] = None

    def validate(self):
        # Section max: reading 30, listening 30, speaking 30, writing 30 → total 120
        for attr, max_val in [("reading",30),("listening",30),("speaking",30),("writing",30)]:
            val = getattr(self, attr)
            if val is not None:
                try:
                    v = int(val)
                    setattr(self, attr, v if 0 <= v <= max_val else None)
                except (TypeError, ValueError):
                    setattr(self, attr, None)
        if self.total:
            parts = [self.reading, self.listening, self.speaking, self.writing]
            if all(p is not None for p in parts):
                calc = sum(parts)
                if abs(calc - self.total) > 5:
                    self.total = calc   # correct total from parts
        return self

@dataclass
class PTEScore:
    overall:    Optional[int] = None
    listening:  Optional[int] = None
    reading:    Optional[int] = None
    writing:    Optional[int] = None
    speaking:   Optional[int] = None
    test_date:  Optional[str] = None

    def validate(self):
        for attr in ("overall","listening","reading","writing","speaking"):
            val = getattr(self, attr)
            if val is not None:
                try:
                    v = int(val)
                    setattr(self, attr, v if 10 <= v <= 90 else None)
                except (TypeError, ValueError):
                    setattr(self, attr, None)
        return self

@dataclass
class PassportData:
    surname:     Optional[str] = None
    given_names: Optional[str] = None
    passport_no: Optional[str] = None
    nationality: Optional[str] = None
    dob:         Optional[str] = None
    expiry:      Optional[str] = None
    mrz:         Optional[str] = None

    def validate(self):
        if self.passport_no:
            # Sri Lankan passports: 1 letter + 7 digits
            if not re.match(r"^[A-Z]\d{7}$", self.passport_no.strip()):
                self.passport_no = None
        return self

@dataclass
class FinancialData:
    bank_name:   Optional[str] = None
    account_no:  Optional[str] = None
    closing_bal: Optional[str] = None
    currency:    Optional[str] = None
    period:      Optional[str] = None

    def validate(self):
        if self.closing_bal:
            # Strip commas, validate it looks like a number
            clean = self.closing_bal.replace(",", "")
            try:
                float(clean)
                self.closing_bal = clean
            except ValueError:
                self.closing_bal = None
        return self


# ─────────────────────────────────────────────────────────────────────────────
# FIELD EXTRACTORS  (kept identical to original)
# ─────────────────────────────────────────────────────────────────────────────

def _f(p, text):
    m = re.search(p, text, re.IGNORECASE)
    return m.group(1).strip() if m else None

def _fa(p, text):
    return re.findall(p, text, re.IGNORECASE)

def _parse_to_dataclass(doc_type: str, text: str):
    t = text.lower()

    if doc_type == "alevel":
        return ALevelResult(
            index_number = _f(r"\b(\d{7})\b", text),
            year         = _f(r"(20\d{2})", text),
            subjects     = _fa(r"(combined maths?|physics|chemistry|biology|economics|accounting|geography|ict)", t),
            grades       = _fa(r"\b([ABCSF])\b", text.upper()),
            z_score      = _f(r"z[\s\-]?score[\s:]+([0-9.]+)", t),
            stream       = _f(r"(science|mathematics|arts|technology|bio science)", t),
        ).validate()

    elif doc_type in ("bachelor", "master", "diploma"):
        return DegreeResult(
            name         = _f(r"(?:certify that|awarded to|conferred upon)\s+([A-Z\s]+)", text),
            degree       = _f(r"(bachelor|master|diploma)[^.\n]{0,60}", t),
            university   = _f(r"university of [\w\s]+|[\w\s]+ university", t),
            year         = _f(r"(20\d{2})", text),
            degree_class = _f(r"(first class|second class|upper second|lower second|distinction|merit|pass)", t),
            gpa          = _f(r"gpa[\s:]+([0-9.]+)", t),
        ).validate()

    elif doc_type == "ielts":
        def _band(p):
            v = _f(p, t)
            return float(v) if v else None
        return IELTSScore(
            overall    = _band(r"overall[\s\w]{0,20}?([0-9]\.[05])"),
            listening  = _band(r"listening[\s:]+([0-9]\.[05])"),
            reading    = _band(r"reading[\s:]+([0-9]\.[05])"),
            writing    = _band(r"writing[\s:]+([0-9]\.[05])"),
            speaking   = _band(r"speaking[\s:]+([0-9]\.[05])"),
            test_date  = _f(r"(\d{1,2}[\s/\-]\w+[\s/\-]20\d{2})", text),
            trf_number = _f(r"(?:trf|reference)[\s:]+([A-Z0-9\-]+)", t),
        ).validate()

    elif doc_type == "toefl":
        def _int(p):
            v = _f(p, t)
            return int(v) if v and v.isdigit() else None
        return TOEFLScore(
            total     = _int(r"total[\s:]+(\d{2,3})"),
            reading   = _int(r"reading[\s:]+(\d{1,2})"),
            listening = _int(r"listening[\s:]+(\d{1,2})"),
            speaking  = _int(r"speaking[\s:]+(\d{1,2})"),
            writing   = _int(r"writing[\s:]+(\d{1,2})"),
            test_date = _f(r"(\d{1,2}[\s/\-]\w+[\s/\-]20\d{2})", text),
        ).validate()

    elif doc_type == "pte":
        def _int(p):
            v = _f(p, t)
            return int(v) if v and v.isdigit() else None
        return PTEScore(
            overall   = _int(r"overall[\s:]+(\d{2,3})"),
            listening = _int(r"listening[\s:]+(\d{2,3})"),
            reading   = _int(r"reading[\s:]+(\d{2,3})"),
            writing   = _int(r"writing[\s:]+(\d{2,3})"),
            speaking  = _int(r"speaking[\s:]+(\d{2,3})"),
            test_date = _f(r"(\d{1,2}[\s/\-]\w+[\s/\-]20\d{2})", text),
        ).validate()

    elif doc_type == "passport":
        return PassportData(
            surname     = _f(r"surname[\s:]+([A-Z]+)", text),
            given_names = _f(r"given\s*names?[\s:]+([A-Z\s]+)", text),
            passport_no = _f(r"\b([A-Z]\d{7})\b", text),
            nationality = _f(r"nationality[\s:]+([A-Z\s]+)", text),
            dob         = _f(r"(?:date of birth|dob)[\s:]+([0-9/\-\s\w]+)", t),
            expiry      = _f(r"(?:date of expiry|expiry)[\s:]+([0-9/\-\s\w]+)", t),
            mrz         = _f(r"(P<<LKA[A-Z<]+)", text),
        ).validate()

    elif doc_type == "financial":
        return FinancialData(
            bank_name   = _f(r"(bank of ceylon|peoples bank|commercial bank|hatton national|sampath|nations trust|dfcc|seylan|ndb)", t),
            account_no  = _f(r"account\s*(?:number|no)[\s:]+([0-9\-]+)", t),
            closing_bal = _f(r"(?:closing|available|current)\s*balance[\s:]*(?:lkr|rs\.?)?[\s]*([0-9,]+(?:\.\d{2})?)", t),
            currency    = _f(r"\b(LKR|USD|GBP|AUD|SGD)\b", text),
            period      = _f(r"(?:statement period|from)[\s:]+([0-9/\-\s\w]+)", t),
        ).validate()

    return {}


# ─────────────────────────────────────────────────────────────────────────────
# OCR ERROR CORRECTOR  (kept from original)
# ─────────────────────────────────────────────────────────────────────────────

_WORD_FIXES = {
    "examlnation":"examination","certlficate":"certificate","unlversity":"university",
    "passporl":"passport","natlonality":"nationality","llstening":"listening",
    "wrlting":"writing","speaklng":"speaking","readlng":"reading",
    "candldate":"candidate","dlstrict":"district","agregate":"aggregate",
    "zscore":"z score","bandsoore":"band score","balence":"balance",
    "statment":"statement","certifled":"certified","postgratuate":"postgraduate",
    "bachelar":"bachelor","batchelor":"bachelor","mastar":"master","diplama":"diploma",
}

def _correct_ocr(text: str) -> str:
    return " ".join(_WORD_FIXES.get(w.lower(), w) for w in text.split())


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING  (updated — added bilateral + HoughLines deskew)
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        pil = Image.open(image_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    if w > 1200:
        s = 1200 / w
        img = cv2.resize(img, (1200, int(h*s)), interpolation=cv2.INTER_AREA)
    elif w < 800:
        s = 800 / w
        img = cv2.resize(img, (800, int(h*s)), interpolation=cv2.INTER_LANCZOS4)

    h, w = img.shape[:2]
    if w > h:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    # HoughLines deskew
    try:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        if lines is not None:
            angles = [(l[0][1]*180/np.pi - 90) for l in lines[:20]
                      if abs(l[0][1]*180/np.pi - 90) < 10]
            if angles:
                angle = float(np.median(angles))
                if abs(angle) > 0.5:
                    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                    gray = cv2.warpAffine(gray, M, (w, h),
                                          borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        pass

    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 21, 10)
    gray = cv2.medianBlur(gray, 3)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# EASYOCR WRAPPER  — CPU mode (updated from original)
# ─────────────────────────────────────────────────────────────────────────────

_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        print("[EasyOCR] Loading model ...")
        _reader = easyocr.Reader(
            lang_list=["en"],
            gpu=False,                           # ← CHANGED: was default (GPU if available)
            model_storage_directory=os.path.join(
                os.path.expanduser("~"), ".EasyOCR", "model"),
            download_enabled=True,
            verbose=False,
        )
        print("[EasyOCR] Ready.")
    return _reader


def _run_ocr(image_path: str) -> tuple[str, float]:
    img     = _preprocess(image_path)
    results = _get_reader().readtext(
        img, detail=1, paragraph=False,
        batch_size=1, workers=0, beamWidth=3,   # ← low-spec settings
    )
    words = [t for (_, t, c) in results if t.strip() and c > 0.25]
    confs = [c for (_, t, c) in results if t.strip() and c > 0.25]
    text  = _correct_ocr(" ".join(words))
    conf  = float(np.mean(confs)) if confs else 0.0
    return text, conf


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT PROCESSING AGENT  (main class — kept same public interface)
# ─────────────────────────────────────────────────────────────────────────────

class DocumentProcessingAgent:
    """
    Orchestration layer.  Same public interface as your original.
    Now uses ML classification instead of pure keyword scoring.
    """

    def process_image(self, image_path: str) -> dict:
        """OCR path: image → EasyOCR (CPU) → ML classify → dataclass → dict"""
        text, ocr_conf = _run_ocr(image_path)
        if not text.strip():
            return {"error": "No text extracted.", "doc_type": "unknown"}

        doc_type, ml_conf, method = classify_document(text)
        result_obj = _parse_to_dataclass(doc_type, text)

        return {
            "doc_type":              doc_type,
            "classification_method": method,        # ml / ml_low_conf / ml_low_conf_top2
            "ml_confidence":         round(ml_conf, 3),
            "ocr_confidence":        round(ocr_conf, 3),
            "data":                  asdict(result_obj) if hasattr(result_obj, "__dataclass_fields__") else result_obj,
        }

    def manual_entry(self, doc_kind: str, data: dict) -> dict:
        """
        Manual entry path — kept identical to your original.
        Accepts typed input, validates via dataclasses, returns same structure.

        Supported doc_kind values:
          alevel, bachelor, master, diploma,
          passport, financial, ielts, toefl, pte
        """
        return ManualEntryProcessor.process(doc_kind, data)


# ─────────────────────────────────────────────────────────────────────────────
# MANUAL ENTRY PROCESSOR  (kept identical to your original)
# ─────────────────────────────────────────────────────────────────────────────

class ManualEntryProcessor:

    @staticmethod
    def process(doc_kind: str, data: dict) -> dict:
        processors = {
            "alevel":    ManualEntryProcessor._alevel,
            "bachelor":  ManualEntryProcessor._degree,
            "master":    ManualEntryProcessor._degree,
            "diploma":   ManualEntryProcessor._degree,
            "passport":  ManualEntryProcessor._passport,
            "financial": ManualEntryProcessor._financial,
            "ielts":     ManualEntryProcessor._ielts,
            "toefl":     ManualEntryProcessor._toefl,
            "pte":       ManualEntryProcessor._pte,
        }
        fn = processors.get(doc_kind)
        if not fn:
            return {"error": f"Unknown doc_kind: {doc_kind}",
                    "supported": list(processors.keys())}
        result_obj = fn(data)
        return {
            "doc_type":              doc_kind,
            "classification_method": "manual_entry",
            "ml_confidence":         1.0,
            "ocr_confidence":        1.0,
            "data":                  asdict(result_obj),
        }

    @staticmethod
    def _alevel(d: dict) -> ALevelResult:
        return ALevelResult(
            index_number = d.get("index_number"),
            year         = str(d.get("year", "")),
            subjects     = d.get("subjects", []),
            grades       = d.get("grades", []),
            z_score      = str(d.get("z_score", "")) if d.get("z_score") else None,
            stream       = d.get("stream"),
        ).validate()

    @staticmethod
    def _degree(d: dict) -> DegreeResult:
        return DegreeResult(
            name         = d.get("name"),
            degree       = d.get("degree"),
            university   = d.get("university"),
            year         = str(d.get("year", "")),
            degree_class = d.get("class") or d.get("degree_class"),
            gpa          = str(d.get("gpa", "")) if d.get("gpa") else None,
        ).validate()

    @staticmethod
    def _ielts(d: dict) -> IELTSScore:
        def _flt(k): return float(d[k]) if d.get(k) is not None else None
        return IELTSScore(
            overall   = _flt("overall"),
            listening = _flt("listening"),
            reading   = _flt("reading"),
            writing   = _flt("writing"),
            speaking  = _flt("speaking"),
            test_date = d.get("test_date"),
            trf_number= d.get("trf_number"),
        ).validate()

    @staticmethod
    def _toefl(d: dict) -> TOEFLScore:
        def _int(k): return int(d[k]) if d.get(k) is not None else None
        return TOEFLScore(
            total     = _int("total"),
            reading   = _int("reading"),
            listening = _int("listening"),
            speaking  = _int("speaking"),
            writing   = _int("writing"),
            test_date = d.get("test_date"),
        ).validate()

    @staticmethod
    def _pte(d: dict) -> PTEScore:
        def _int(k): return int(d[k]) if d.get(k) is not None else None
        return PTEScore(
            overall   = _int("overall"),
            listening = _int("listening"),
            reading   = _int("reading"),
            writing   = _int("writing"),
            speaking  = _int("speaking"),
            test_date = d.get("test_date"),
        ).validate()

    @staticmethod
    def _passport(d: dict) -> PassportData:
        return PassportData(
            surname     = d.get("surname"),
            given_names = d.get("given_names"),
            passport_no = d.get("passport_no"),
            nationality = d.get("nationality", "SRI LANKAN"),
            dob         = d.get("dob"),
            expiry      = d.get("expiry"),
            mrz         = d.get("mrz"),
        ).validate()

    @staticmethod
    def _financial(d: dict) -> FinancialData:
        return FinancialData(
            bank_name   = d.get("bank_name"),
            account_no  = d.get("account_no"),
            closing_bal = str(d.get("closing_bal", "")) if d.get("closing_bal") else None,
            currency    = d.get("currency", "LKR"),
            period      = d.get("period"),
        ).validate()


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Document Processing Agent — Self Test ===\n")

    agent = DocumentProcessingAgent()

    # Test manual entry path (all doc types)
    test_cases = [
        ("ielts",    {"overall":7.0,"listening":7.5,"reading":8.0,"writing":6.5,"speaking":7.0,"test_date":"12 Jan 2024"}),
        ("toefl",    {"total":105,"reading":28,"listening":27,"speaking":24,"writing":26}),
        ("pte",      {"overall":65,"listening":68,"reading":72,"writing":60,"speaking":63}),
        ("alevel",   {"index_number":"1234567","year":2023,"subjects":["Physics","Chemistry"],"grades":["A","B"],"z_score":1.85,"stream":"bio science"}),
        ("bachelor", {"name":"K.D. Perera","degree":"BSc Computer Science","university":"University of Colombo","year":2022,"class":"Second Class Upper","gpa":3.42}),
        ("passport", {"surname":"PERERA","given_names":"KASUN DILSHAN","passport_no":"N1234567","nationality":"SRI LANKAN","dob":"15/06/1998","expiry":"14/06/2028"}),
        ("financial",{"bank_name":"Commercial Bank","account_no":"1234567890","closing_bal":"1500000.00","currency":"LKR"}),
    ]

    all_ok = True
    for doc_kind, data in test_cases:
        result = agent.manual_entry(doc_kind, data)
        ok = result.get("classification_method") == "manual_entry"
        status = "✓" if ok else "✗"
        if not ok: all_ok = False
        print(f"  {status} {doc_kind:12s}  →  {result['data']}")
        print()

    print(f"\nML classifier test:")
    test_texts = [
        ("IELTS overall band 7.0 listening 7.5 reading 8.0 writing 6.5", "ielts"),
        ("TOEFL iBT total score 100 ETS reading listening speaking writing", "toefl"),
        ("passport nationality LKA date of expiry immigration P<<LKA", "passport"),
        ("bank statement closing balance LKR commercial bank certified", "financial"),
        ("advanced level z score district rank gce al examination", "alevel"),
    ]
    for text, expected in test_texts:
        got, conf, method = classify_document(text)
        ok = "✓" if got == expected else "✗"
        print(f"  {ok} {expected:12s} → {got:12s}  conf={conf:.2f}  method={method}")