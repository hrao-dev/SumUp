import streamlit as st
from newspaper import Article
from transformers import pipeline
import PyPDF2
import spacy
from keybert import KeyBERT
from rouge_score import rouge_scorer
import textstat

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SumUp — Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600&family=DM+Mono:wght@400;500&display=swap');

/* ── Design tokens ── */
:root {
    --bg:           #0A0C10;
    --bg-deep:      #060709;
    --surface:      #10141C;
    --surface-2:    #141820;
    --surface-3:    #1A1F2B;
    --border:       #1E2436;
    --border-mid:   #2A3650;
    --border-light: #252D40;

    --accent:       #4A7CFA;
    --accent-soft:  #0D1428;
    --accent-mid:   #2A4EB0;
    --accent-dark:  #3560D8;
    --accent-glow:  rgba(74,124,250,0.20);
    --accent-text:  #9DB8FF;

    --success:      #2DD4A0;
    --success-soft: #081E18;

    --info-bg:      #0D1220;
    --info-border:  #1A2545;
    --info-text:    #5A7AB0;

    --text-primary:   #D8E0F0;
    --text-secondary: #6D7D99;
    --text-muted:     #323D54;
    --text-inverse:   #060709;

    --shadow-sm:     0 1px 4px rgba(0,0,0,0.5);
    --shadow-accent: 0 4px 22px rgba(74,124,250,0.28);

    --r-xs:   3px;
    --r-sm:   6px;
    --r-md:   8px;
    --r-lg:   12px;
    --r-pill: 999px;
}

/* ── Base ── */
*, *::before, *::after {
    font-family: 'DM Sans', system-ui, sans-serif !important;
    box-sizing: border-box;
}
html, body, [class*="css"] {
    font-family: 'DM Sans', system-ui, sans-serif !important;
    background: var(--bg) !important;
}
.stApp { background: var(--bg) !important; }
#MainMenu, footer, .stDeployButton { display: none !important; }
[data-testid="stToolbar"]  { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] {
    background: var(--bg) !important;
    border-bottom: 1px solid var(--border) !important;
}
.block-container {
    padding: 0 0 60px 0 !important;
    max-width: 100% !important;
}
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
section.main > div {
    background: var(--bg) !important;
}
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 2px; }

/* ── Header band ── */
.hdr {
    background: var(--bg-deep);
    border-bottom: 1px solid var(--border);
    padding: 32px 48px 26px;
    position: relative;
    margin-bottom: 0;
}
.hdr::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg,
        #4A7CFA 0%, #7C6AFA 50%, #2DD4A0 100%);
}
.hdr-eye {
    font-size: .6rem; font-weight: 500;
    letter-spacing: .2em; text-transform: uppercase;
    color: var(--text-muted);
    font-family: 'DM Mono', monospace !important;
    margin: 0 0 10px; display: block;
}
.hdr-title {
    font-family: 'DM Sans', system-ui, sans-serif !important;
    font-size: 2.2rem; font-weight: 600;
    color: var(--text-primary); line-height: 1.1;
    margin: 0 0 8px; letter-spacing: -0.04em;
}
.hdr-title em {
    font-style: normal;
    color: var(--accent);
}
.hdr-sub {
    font-size: .875rem; font-weight: 400;
    color: var(--text-secondary); line-height: 1.7;
    margin: 0 0 16px; max-width: 520px;
}
.chips { display: flex; flex-wrap: wrap; gap: 5px; }
.chip {
    font-size: .58rem; font-weight: 500;
    letter-spacing: .1em; text-transform: uppercase;
    padding: 2px 8px; border-radius: var(--r-pill);
    border: 1px solid; display: inline-block; line-height: 1.7;
    font-family: 'DM Mono', monospace !important;
}
.cg { background: var(--accent-soft); color: var(--accent-text); border-color: var(--accent-mid); }
.ct { background: var(--success-soft); color: var(--success); border-color: #0A3828; }
.ca { background: var(--surface-3); color: var(--text-secondary); border-color: var(--border-light); }
.cv { background: var(--surface-2); color: var(--text-muted); border-color: var(--border); }

/* ── Content area ── */
.content { padding: 24px 40px 0; }

/* ── Section heading ── */
.sec-head {
    font-size: .58rem; font-weight: 500;
    letter-spacing: .18em; text-transform: uppercase;
    color: var(--text-muted); margin: 0 0 8px;
    display: block;
    font-family: 'DM Mono', monospace !important;
}

/* ── Ruled divider ── */
.rule {
    border: none;
    border-top: 1px solid var(--border);
    margin: 20px 0;
}

/* ── Accent rules ── */
.accent-g { border: none; border-top: 2px solid var(--accent);  margin: 0 0 12px; }
.accent-t { border: none; border-top: 2px solid var(--success); margin: 0 0 12px; }
.accent-a { border: none; border-top: 2px solid #7C6AFA;        margin: 0 0 12px; }
.accent-v { border: none; border-top: 2px solid #2DD4A0;        margin: 0 0 12px; }

/* ── Textarea ── */
div[data-testid="stTextArea"] textarea {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-md) !important;
    font-family: 'DM Sans', system-ui, sans-serif !important;
    font-size: .88rem !important;
    color: var(--text-primary) !important;
    line-height: 1.7 !important;
    transition: border-color .15s !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
    outline: none !important;
}
div[data-testid="stTextArea"] textarea::placeholder {
    color: var(--text-muted) !important;
}
div[data-testid="stTextArea"] label {
    font-size: .58rem !important;
    font-weight: 500 !important;
    letter-spacing: .16em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    font-family: 'DM Mono', monospace !important;
}

/* ── Select box ── */
div[data-testid="stSelectbox"] > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-md) !important;
    font-family: 'DM Sans', system-ui, sans-serif !important;
    font-size: .85rem !important;
    color: var(--text-primary) !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
div[data-testid="stSelectbox"] label {
    font-size: .58rem !important;
    font-weight: 500 !important;
    letter-spacing: .16em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    font-family: 'DM Mono', monospace !important;
}

/* ── File uploader ── */
div[data-testid="stFileUploader"] section {
    background: var(--accent-soft) !important;
    border: 1.5px dashed var(--accent-mid) !important;
    border-radius: var(--r-lg) !important;
    transition: border-color .15s, background .15s, box-shadow .15s !important;
}
div[data-testid="stFileUploader"] section:hover {
    border-color: var(--accent) !important;
    background: var(--accent-soft) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
div[data-testid="stFileUploader"] label {
    font-size: .58rem !important;
    font-weight: 500 !important;
    letter-spacing: .16em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: var(--accent-soft) !important;
    border: 1.5px dashed var(--accent-mid) !important;
    border-radius: var(--r-lg) !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] p {
    color: var(--text-secondary) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploaderDropzoneInstructions"] > div > small {
    display: none !important;
}
[data-testid="stFileUploaderFile"] {
    background: var(--surface-3) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-sm) !important;
}
[data-testid="stFileUploaderFileName"] {
    color: var(--text-primary) !important;
}

/* ── Buttons ── */
div[data-testid="stButton"] > button {
    font-family: 'DM Sans', system-ui, sans-serif !important;
    font-weight: 500 !important;
    border-radius: var(--r-md) !important;
    width: 100% !important;
    transition: all .15s !important;
    letter-spacing: -.01em !important;
}
div[data-testid="stButton"] > button[kind="primary"] {
    background: var(--accent) !important;
    color: var(--text-inverse) !important;
    border: none !important;
    padding: 11px 20px !important;
    font-size: .84rem !important;
    box-shadow: var(--shadow-accent) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: var(--accent-dark) !important;
    box-shadow: 0 6px 26px rgba(74,124,250,0.4) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:active {
    transform: translateY(0) !important;
}
div[data-testid="stButton"] > button[kind="secondary"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-secondary) !important;
    font-size: .82rem !important;
    padding: 9px 20px !important;
}
div[data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: var(--accent-mid) !important;
    color: var(--accent-text) !important;
    background: var(--accent-soft) !important;
}

/* ── st.metric ── */
div[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: var(--r-lg) !important;
    padding: 14px 18px !important;
}
div[data-testid="stMetric"] label {
    font-size: .58rem !important;
    font-weight: 500 !important;
    letter-spacing: .14em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    font-family: 'DM Mono', monospace !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 1.3rem !important;
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-size: .72rem !important;
}

/* ── Expander ── */
div[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-md) !important;
}
div[data-testid="stExpander"] summary {
    font-size: .82rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    padding: 10px 16px !important;
}

/* ── Alerts ── */
div[data-testid="stAlert"] {
    border-radius: var(--r-md) !important;
    font-size: .85rem !important;
}
[data-baseweb="notification"][kind="positive"] {
    background: var(--success-soft) !important;
    border-color: var(--success) !important;
    color: var(--success) !important;
    border-radius: var(--r-md) !important;
}
[data-baseweb="notification"][kind="positive"] svg { fill: var(--success) !important; }

/* ── Inline tag pill ── */
.tag {
    display: inline-block;
    font-size: .72rem; font-weight: 400;
    padding: 2px 9px; border-radius: var(--r-pill);
    margin: 0 3px 4px 0;
    background: var(--accent-soft);
    border: 1px solid var(--accent-mid);
    color: var(--accent-text);
    font-family: 'DM Mono', monospace !important;
}

/* ── Monospace value badge ── */
.val-t {
    display: inline-block;
    font-family: 'DM Mono', monospace !important;
    font-size: .76rem;
    padding: 1px 6px; border-radius: var(--r-xs);
    background: var(--accent-soft);
    border: 1px solid var(--accent-mid);
    color: var(--accent-text);
}
.val-a {
    display: inline-block;
    font-family: 'DM Mono', monospace !important;
    font-size: .76rem;
    padding: 1px 6px; border-radius: var(--r-xs);
    background: var(--surface-3);
    border: 1px solid var(--border-light);
    color: var(--text-secondary);
}

/* ── Sentence highlight ── */
mark.hl {
    background: rgba(74,124,250,0.18);
    border-bottom: 2px solid var(--accent);
    border-radius: 2px;
    padding: 0 2px;
    color: var(--text-primary);
    cursor: help;
}

/* ── Result output text ── */
.out {
    font-size: .9rem; font-weight: 400;
    color: var(--text-primary); line-height: 1.85;
    font-family: 'DM Sans', system-ui, sans-serif !important;
}
.out p { margin: 0 0 .6em; }

/* ── Placeholder ── */
.ph {
    font-size: .84rem; font-style: italic;
    color: var(--text-muted);
    font-family: 'DM Sans', system-ui, sans-serif !important;
}

/* ── Stat block ── */
.stat {
    display: flex; align-items: baseline;
    gap: 8px; flex-wrap: wrap;
    margin-bottom: 10px;
}
.stat-lbl {
    font-size: .75rem; font-weight: 500;
    color: var(--text-secondary); min-width: 100px;
    font-family: 'DM Sans', system-ui, sans-serif !important;
}
.stat-note {
    font-size: .72rem; color: var(--text-muted);
    font-family: 'DM Sans', system-ui, sans-serif !important;
}
.stat-delta {
    font-size: .72rem; font-weight: 500;
    color: var(--accent-text);
    font-family: 'DM Mono', monospace !important;
}

/* ── Streamlit markdown text visibility ── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    color: var(--text-primary) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    import subprocess, sys
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    summ = pipeline("summarization", model="harao-ml/flant5-finetuned-summarize")
    kw   = KeyBERT()
    nlp  = spacy.load("en_core_web_sm")
    return summ, kw, nlp

summarizer, kw_model, nlp = load_models()

# ── NLP functions ─────────────────────────────────────────────────────────────
def split_text(text, max_tokens=512):
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield ' '.join(words[i:i + max_tokens])

def clean_text(text):
    import re
    # Collapse all whitespace including newlines from PDF extraction
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join(w for w in text.split() if len(w) < 100)

def fetch_article(url):
    try:
        a = Article(url)
        a.download(); a.parse()
        return (
            a.title or "Untitled",
            ", ".join(a.authors) if a.authors else "Unknown",
            a.publish_date.strftime('%B %d, %Y') if a.publish_date else "Unknown",
            a.text,
        )
    except Exception as e:
        return None, None, None, f"Error: {e}"

def generate_summary(content, min_len, max_len):
    if not content.strip(): return ""
    chunks = list(split_text(clean_text(content)))
    return ''.join([
        summarizer(c, min_length=min_len, max_length=max_len, do_sample=False)[0]['summary_text']
        for c in chunks if c.strip()
    ]) if chunks else ''

def extract_insights(text):
    kwds = [k for k, _ in kw_model.extract_keywords(
        text, keyphrase_ngram_range=(1,2), stop_words='english', top_n=8)]
    doc  = nlp(text)
    ents = {}
    for e in doc.ents:
        if e.label_ not in ents: ents[e.label_] = []
        if e.text not in ents[e.label_]: ents[e.label_].append(e.text)
    lmap = {"PERSON":"People","ORG":"Organizations","GPE":"Locations",
            "DATE":"Dates","MONEY":"Money","EVENT":"Events",
            "PRODUCT":"Products","LAW":"Laws"}
    return kwds, ents, lmap

def compute_rouge(src, summ):
    sc = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    s  = sc.score(src, summ)
    return s['rouge1'], s['rouge2'], s['rougeL']

def highlight_sentences(src, summ, top_n=3):
    sc   = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    # Normalize whitespace — PDFs often have newlines after every word.
    # Collapse all whitespace runs (including \n, \r, \t) to single spaces.
    import re
    src_clean = re.sub(r'\s+', ' ', src).strip()

    doc  = nlp(src_clean[:8000])
    # Filter: must be >30 chars and contain at least 4 words to be a real sentence
    sens = [
        s.text.strip() for s in doc.sents
        if len(s.text.strip()) > 30 and len(s.text.strip().split()) >= 4
    ]
    if not sens: return []
    scored = [(s, sc.score(summ, s)['rouge1'].fmeasure) for s in sens]
    thresh = sorted([v for _,v in scored], reverse=True)[min(top_n-1, len(scored)-1)]
    out, n = [], 0
    for s, v in scored:
        if v >= thresh and n < top_n and v > 0:
            out.append((s, True, v)); n += 1
        else:
            out.append((s, False, 0.0))
    return out

def reading_level(src, summ):
    def gl(g):
        if g<=6:  return "Elementary"
        if g<=8:  return "Middle school"
        if g<=10: return "High school"
        if g<=13: return "College"
        return "Graduate"
    def el(e):
        if e>=90: return "Very easy"
        if e>=70: return "Easy"
        if e>=60: return "Standard"
        if e>=50: return "Fairly difficult"
        if e>=30: return "Difficult"
        return "Very difficult"
    sg, ug = textstat.flesch_kincaid_grade(src), textstat.flesch_kincaid_grade(summ)
    se, ue = textstat.flesch_reading_ease(src),  textstat.flesch_reading_ease(summ)
    sw, uw = len(src.split()), len(summ.split())
    comp   = round((1 - uw/sw)*100) if sw > 0 else 0
    d      = sg - ug
    dl     = f"↓ {abs(d):.1f} easier" if d>0.5 else f"↑ {abs(d):.1f} harder" if d<-0.5 else "similar"
    return dict(sg=sg, ug=ug, se=se, ue=ue, sw=sw, uw=uw, comp=comp,
                dl=dl, gls=gl(sg), glu=gl(ug), els=el(se), elu=el(ue))

def ubar(v, mx=1.0, w=10):
    f = round(min(v/mx, 1.0)*w)
    return "█"*f + "░"*(w-f)

LENGTH_MAP = {"Brief":(30,80), "Balanced":(60,180), "Detailed":(120,300)}

# ── Session state ─────────────────────────────────────────────────────────────
RESULT_KEYS = ["summary","content","insights","rouge","highlight","reading","url_meta","error"]
INPUT_KEYS  = ["text_input","length_select"]

for k, v in dict(
    summary=None, content=None, insights=None, rouge=None,
    highlight=None, reading=None, url_meta=None, error=None,
    do_clear=False, do_summarize=False,
    pending_file=None, pending_text="", pending_length="Balanced",
    file_uploader_key=0,
).items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hdr">
  <span class="hdr-eye">Document Intelligence Platform</span>
  <h1 class="hdr-title">Sum<em>Up</em></h1>
  <p class="hdr-sub">Distill any document, article, or text into a structured summary — with
  keyword extraction, named entity recognition, ROUGE scoring, sentence highlighting,
  and reading level analysis.</p>
  <div class="chips">
    <span class="chip cg">FLAN-T5</span>
    <span class="chip cg">KeyBERT</span>
    <span class="chip ct">spaCy NER</span>
    <span class="chip ca">ROUGE</span>
    <span class="chip ca">Reading Level</span>
    <span class="chip cv">Highlighting</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
if "do_clear" not in st.session_state:
    st.session_state.do_clear = False

with st.sidebar:
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        background: #060709 !important;
        border-right: 1px solid #1E2436 !important;
        min-width: 300px !important;
        max-width: 340px !important;
    }
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] > div > div,
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding: 24px 18px 40px !important;
    }
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] button {
        font-family: 'DM Sans', system-ui, sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar logo / wordmark
    st.markdown("""
    <div style='display:flex;align-items:center;gap:9px;
                margin-bottom:18px;padding-bottom:14px;
                border-bottom:1px solid #1E2436;'>
        <div style='width:30px;height:30px;
                    background:#0D1428;
                    border:1px solid #2A4EB0;
                    border-radius:6px;
                    display:flex;align-items:center;justify-content:center;
                    font-family:"DM Mono",monospace;
                    font-size:.72rem;font-weight:500;
                    color:#9DB8FF;letter-spacing:.04em;flex-shrink:0;'>SU</div>
        <div>
            <div style='font-size:.95rem;font-weight:600;
                        color:#D8E0F0;letter-spacing:-.025em;line-height:1.2;'>SumUp</div>
            <div style='font-size:.62rem;color:#323D54;
                        font-family:"DM Mono",monospace;
                        letter-spacing:.04em;margin-top:2px;'>document intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    text_val = st.text_area(
        "Paste text or a URL",
        value=st.session_state.get("pending_text", ""),
        placeholder="e.g. https://example.com/article  or  paste any long-form text here…",
        height=210,
        key="text_input",
        help="Paste raw text, a news article URL, or any document content",
    )
    st.session_state.pending_text = text_val

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    file_val = st.file_uploader(
        "Or upload a PDF / TXT file",
        type=["pdf","txt"],
        key=f"file_upload_{st.session_state.file_uploader_key}",
        help="PDF and TXT supported",
    )
    st.session_state.pending_file = file_val

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<hr style='border:none;border-top:1px solid #1E2436;margin:0 0 14px'>",
        unsafe_allow_html=True,
    )

    length_val = st.selectbox(
        "Summary length",
        options=["Brief", "Balanced", "Detailed"],
        index=["Brief","Balanced","Detailed"].index(
            st.session_state.get("pending_length", "Balanced")
        ),
        key="length_select",
        help="Brief: 30–80 tokens · Balanced: 60–180 tokens · Detailed: 120–300 tokens",
    )
    st.session_state.pending_length = length_val

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<hr style='border:none;border-top:1px solid #1E2436;margin:0 0 18px'>",
        unsafe_allow_html=True,
    )

    btn_col, clr_col = st.columns([1, 1], gap="small")

    with btn_col:
        if st.button(
            "Summarize",
            type="primary",
            use_container_width=True,
            key="btn_summarize",
        ):
            st.session_state.do_summarize = True

    with clr_col:
        if st.button(
            "Clear",
            type="secondary",
            use_container_width=True,
            key="btn_clear",
        ):
            st.session_state.do_clear = True

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<hr style='border:none;border-top:1px solid #1E2436;margin:0 0 12px'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:.68rem;color:#323D54;line-height:1.65;font-family:\"DM Mono\",monospace;'>"
        "Fine-tuned FLAN-T5 · "
        "<a href='https://huggingface.co/harao-ml/flant5-finetuned-summarize' "
        "target='_blank' style='color:#4A7CFA;text-decoration:none;'>"
        "harao-ml/flant5-finetuned-summarize ↗</a>"
        "</p>",
        unsafe_allow_html=True,
    )

# ── Clear handler ─────────────────────────────────────────────────────────────
if st.session_state.do_clear:
    st.session_state.do_clear     = False
    st.session_state.do_summarize = False
    for k in RESULT_KEYS:
        st.session_state[k] = None
    st.session_state.pending_text   = ""
    st.session_state.pending_file   = None
    st.session_state.pending_length = "Balanced"
    st.session_state.file_uploader_key += 1
    for k in INPUT_KEYS:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

# ── Summarize handler ─────────────────────────────────────────────────────────
if st.session_state.do_summarize:
    st.session_state.do_summarize = False

    content  = None
    err      = None
    url_meta = None

    active_file   = st.session_state.pending_file
    active_text   = st.session_state.pending_text.strip()
    active_length = st.session_state.pending_length
    min_len, max_len = LENGTH_MAP[active_length]

    if active_text:
        if active_text.startswith("http://") or active_text.startswith("https://"):
            with st.spinner("Fetching article…"):
                title, author, pub_date, fetched = fetch_article(active_text)
            if not fetched or (isinstance(fetched, str) and fetched.startswith("Error")):
                err = f"Could not fetch URL — {fetched}"
            else:
                content  = fetched
                url_meta = dict(
                    title=title, author=author,
                    pub_date=pub_date, url=active_text,
                )
        else:
            content = active_text

    elif active_file is not None:
        try:
            if active_file.name.endswith(".pdf"):
                reader  = PyPDF2.PdfReader(active_file)
                content = "".join(p.extract_text() or "" for p in reader.pages)
            else:
                content = active_file.read().decode("utf-8")
            if not content.strip():
                err = "No extractable text found in the uploaded file."
        except Exception as e:
            err = f"Error reading file: {e}"

    else:
        err = "Please paste some text, enter a URL, or upload a file first."

    if err:
        st.session_state.error = err
        for k in ["summary","content","insights","rouge","highlight","reading","url_meta"]:
            st.session_state[k] = None
    elif content:
        st.session_state.error = None
        with st.spinner("Summarizing…"):
            summ = generate_summary(content, min_len, max_len)
        with st.spinner("Extracting keywords and entities…"):
            kwds, ents, lmap = extract_insights(summ)
        with st.spinner("Computing ROUGE scores…"):
            r1, r2, rL = compute_rouge(content, summ)
        with st.spinner("Highlighting key sentences…"):
            hl = highlight_sentences(content, summ)
        with st.spinner("Analysing reading level…"):
            rl = reading_level(content, summ)
        st.session_state.update(dict(
            summary=summ, content=content,
            insights=(kwds, ents, lmap),
            rouge=(r1, r2, rL),
            highlight=hl, reading=rl,
            url_meta=url_meta,
        ))

# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Output area
# ══════════════════════════════════════════════════════════════════════════════
main = st.container()

with main:
    st.markdown('<div class="content">', unsafe_allow_html=True)

    if st.session_state.error:
        st.error(st.session_state.error)

    # ── Empty state ───────────────────────────────────────────────────────────
    if not st.session_state.summary and not st.session_state.error:
        st.markdown("""
        <div style='
            margin: 40px auto 48px;
            max-width: 480px;
            text-align: center;
            padding: 40px 36px;
            background: #10141C;
            border: 1px solid #1E2436;
            border-radius: 10px;
        '>
            <div style='
                width:44px;height:44px;
                background:#0D1428;
                border:1px solid #2A4EB0;
                border-radius:10px;
                display:flex;align-items:center;justify-content:center;
                font-family:"DM Mono",monospace;
                font-size:.9rem;font-weight:500;color:#9DB8FF;
                margin:0 auto 16px;
            '>SU</div>
            <p style='
                font-family:"DM Sans",system-ui,sans-serif;
                font-size:1.1rem;
                font-weight:600;
                color:#D8E0F0;
                margin: 0 0 8px;
                letter-spacing:-.02em;
            '>Ready to summarize</p>
            <p style='
                font-size: .84rem;
                color: #6D7D99;
                margin: 0 0 24px;
                line-height: 1.7;
            '>
                Use the panel on the left to paste text, drop in a URL,
                or upload a PDF or TXT file. Then click
                <strong style="color:#4A7CFA">Summarize</strong>
            </p>
            <div style='display:flex;flex-direction:column;gap:8px;text-align:left;'>
                <div style='
                    display:flex;align-items:flex-start;gap:10px;
                    background:#14181F;border:1px solid #1E2436;
                    border-radius:6px;padding:10px 12px;
                '>
                    <div>
                        <p style='font-size:.78rem;font-weight:500;color:#D8E0F0;margin:0 0 2px'>
                            Paste text or a URL
                        </p>
                        <p style='font-size:.72rem;color:#6D7D99;margin:0;line-height:1.5'>
                            Article links, research content, any long-form text
                        </p>
                    </div>
                </div>
                <div style='
                    display:flex;align-items:flex-start;gap:10px;
                    background:#14181F;border:1px solid #1E2436;
                    border-radius:6px;padding:10px 12px;
                '>
                    <div>
                        <p style='font-size:.78rem;font-weight:500;color:#D8E0F0;margin:0 0 2px'>
                            Upload a file
                        </p>
                        <p style='font-size:.72rem;color:#6D7D99;margin:0;line-height:1.5'>
                            PDF or TXT — up to full document length
                        </p>
                    </div>
                </div>
                <div style='
                    display:flex;align-items:flex-start;gap:10px;
                    background:#0D1428;
                    border:1px solid #2A4EB0;
                    border-radius:6px;padding:10px 12px;
                '>
                    <div>
                        <p style='font-size:.78rem;font-weight:500;color:#9DB8FF;margin:0 0 2px'>
                            Click Summarize
                        </p>
                        <p style='font-size:.72rem;color:#4A6080;margin:0;line-height:1.5'>
                            Runs summarization, NER, ROUGE scoring,
                            sentence highlighting &amp; reading level analysis
                        </p>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── 1. SUMMARY ────────────────────────────────────────────────────────────
    st.markdown('<hr class="accent-g">', unsafe_allow_html=True)
    st.markdown('<span class="sec-head">Summary</span>', unsafe_allow_html=True)

    if st.session_state.summary:
        if st.session_state.url_meta:
            m = st.session_state.url_meta
            st.markdown(
                f"**{m['title']}**  \n"
                f"*{m['author']} · {m['pub_date']}*"
            )
            st.markdown("---")
        st.markdown(st.session_state.summary)
        if st.session_state.url_meta:
            st.markdown(f"[↗ Read full article]({st.session_state.url_meta['url']})")

    # ── 2. INSIGHTS + ROUGE ───────────────────────────────────────────────────
    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown('<hr class="accent-t">', unsafe_allow_html=True)
        st.markdown('<span class="sec-head">Information Extraction</span>', unsafe_allow_html=True)

        if st.session_state.insights:
            kwds, ents, lmap = st.session_state.insights

            st.markdown(
                "<p style='font-size:.58rem;font-weight:500;letter-spacing:.14em;"
                "text-transform:uppercase;color:#323D54;margin:0 0 8px;"
                "font-family:\"DM Mono\",monospace;'>Key Phrases</p>",
                unsafe_allow_html=True,
            )
            if kwds:
                st.markdown(
                    " ".join(f'<span class="tag">{k}</span>' for k in kwds),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("<span class='ph'>None detected</span>", unsafe_allow_html=True)

            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            st.markdown(
                "<p style='font-size:.58rem;font-weight:500;letter-spacing:.14em;"
                "text-transform:uppercase;color:#323D54;margin:0 0 8px;"
                "font-family:\"DM Mono\",monospace;'>Named Entities</p>",
                unsafe_allow_html=True,
            )
            if ents:
                for lb, items in ents.items():
                    name = lmap.get(lb, lb)
                    st.markdown(
                        f"<p style='font-size:.85rem;margin:0 0 5px;line-height:1.5;"
                        f"color:var(--text-primary);'>"
                        f"<span style='font-weight:500;color:#9DB8FF'>{name}</span>"
                        f"<span style='color:#323D54'> — </span>"
                        f"<span style='color:#D8E0F0'>{', '.join(items)}</span></p>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown("<span class='ph'>None detected</span>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='ph'>Keywords and entities will appear here.</p>",
                        unsafe_allow_html=True)

    with col_b:
        st.markdown('<hr class="accent-a">', unsafe_allow_html=True)
        st.markdown('<span class="sec-head">ROUGE Evaluation</span>', unsafe_allow_html=True)

        if st.session_state.rouge:
            r1, r2, rL = st.session_state.rouge

            def rrow(label, sc):
                return (
                    f"<p style='font-size:.85rem;margin:0 0 10px;line-height:1.5;'>"
                    f"<span style='font-weight:500;color:#9DB8FF;"
                    f"display:inline-block;width:30px;font-family:\"DM Mono\",monospace;'>{label}</span>"
                    f"&nbsp;<span class='val-t'>{sc.fmeasure:.3f}</span>"
                    f"&nbsp;<span style='font-size:.72rem;color:#323D54;"
                    f"font-family:\"DM Mono\",monospace;'>{ubar(sc.fmeasure)}</span>"
                    f"<br><span style='font-size:.7rem;color:#323D54;"
                    f"margin-left:34px;font-family:\"DM Mono\",monospace;'>"
                    f"Precision&thinsp;<span class='val-a'>{sc.precision:.3f}</span>"
                    f"&nbsp;&nbsp;Recall&thinsp;<span class='val-a'>{sc.recall:.3f}</span>"
                    f"</span></p>"
                )

            st.markdown(rrow("R-1", r1), unsafe_allow_html=True)
            st.markdown(rrow("R-2", r2), unsafe_allow_html=True)
            st.markdown(rrow("R-L", rL), unsafe_allow_html=True)
            st.markdown(
                "<p style='font-size:.7rem;color:#323D54;font-style:italic;margin-top:6px;"
                "font-family:\"DM Mono\",monospace;'>"
                "Higher F1 = more source content preserved</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<p class='ph'>ROUGE scores will appear here.</p>",
                        unsafe_allow_html=True)

    # ── 3. SENTENCE HIGHLIGHTING ──────────────────────────────────────────────
    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown('<hr class="accent-v">', unsafe_allow_html=True)
    st.markdown('<span class="sec-head">Source Sentence Highlighting</span>',
                unsafe_allow_html=True)

    if st.session_state.highlight:
        st.markdown(
            "<p style='font-size:.72rem;color:#323D54;font-style:italic;margin:0 0 12px;"
            "font-family:\"DM Mono\",monospace;'>"
            "<mark class='hl' style='font-style:normal;font-size:.7rem;padding:1px 5px'>"
            "Highlighted</mark>&nbsp; sentences contributed most to the summary</p>",
            unsafe_allow_html=True,
        )
        parts = []
        for sent, is_hl, score in st.session_state.highlight:
            esc = sent.replace("<","&lt;").replace(">","&gt;")
            if is_hl:
                parts.append(f'<mark class="hl" title="Relevance: {score:.3f}">{esc}</mark>')
            else:
                parts.append(f'<span style="color:#6D7D99">{esc}</span>')
        st.markdown(
            f"<div class='out'>{' '.join(parts)}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<p class='ph'>Source sentence highlights will appear here.</p>",
                    unsafe_allow_html=True)

    # ── 4. READING LEVEL ──────────────────────────────────────────────────────
    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    st.markdown('<hr class="accent-a">', unsafe_allow_html=True)
    st.markdown('<span class="sec-head">Reading Level &amp; Compression</span>',
                unsafe_allow_html=True)

    if st.session_state.reading:
        r = st.session_state.reading

        m1, m2, m3, m4 = st.columns(4, gap="medium")
        with m1:
            st.metric(
                "Source Grade",
                f"{r['sg']:.1f}",
                delta=None,
                help=r['gls'],
            )
        with m2:
            st.metric(
                "Summary Grade",
                f"{r['ug']:.1f}",
                delta=r['dl'],
                delta_color="normal",
                help=r['glu'],
            )
        with m3:
            st.metric(
                "Reading Ease",
                f"{r['ue']:.0f}",
                help=r['elu'],
            )
        with m4:
            st.metric(
                "Compression",
                f"{r['comp']}%",
                help=f"{r['sw']:,} → {r['uw']:,} words",
            )

        st.markdown(
            f"<p style='font-size:.7rem;color:#323D54;margin-top:10px;"
            f"font-family:\"DM Mono\",monospace;'>"
            f"Source: {r['gls']} ({r['els']}) &nbsp;·&nbsp; "
            f"Summary: {r['glu']} ({r['elu']}) &nbsp;·&nbsp; "
            f"Flesch-Kincaid: lower grade = more accessible</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<p class='ph'>Reading level comparison will appear here.</p>",
                    unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
