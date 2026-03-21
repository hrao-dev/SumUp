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
# Strategy: style only Streamlit's own rendered elements.
# No custom HTML card divs — they break due to Streamlit's injected wrappers.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --ink:       #161510;
    --ink-2:     #4a4840;
    --ink-3:     #9a9890;
    --line:      #e6e3da;
    --line-2:    #d0cdc4;
    --bg:        #f8f7f4;
    --bg-2:      #ffffff;
    --green:     #2a6348;
    --green-lt:  #ebf5ef;
    --green-bd:  #b8ddc8;
    --teal:      #176070;
    --teal-lt:   #eaf4f6;
    --teal-bd:   #b0d8e0;
    --amber:     #7c5210;
    --amber-lt:  #fdf4e6;
    --amber-bd:  #f0d490;
    --violet:    #3e38a0;
    --violet-lt: #eeedfb;
    --violet-bd: #c4c2ee;
    --f-head:    'DM Serif Display', Georgia, serif;
    --f-body:    'Inter', system-ui, sans-serif;
    --f-mono:    'JetBrains Mono', monospace;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: var(--f-body) !important;
}
.stApp { background: var(--bg) !important; }
#MainMenu, footer, header, .stDeployButton { display: none !important; }
.block-container {
    padding: 0 0 60px 0 !important;
    max-width: 100% !important;
}
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--line-2); border-radius: 2px; }

/* ── Header band ── */
.hdr {
    background: var(--bg-2);
    border-bottom: 1px solid var(--line);
    padding: 36px 48px 28px;
    position: relative;
    margin-bottom: 0;
}
.hdr::after {
    content:'';
    position:absolute;
    bottom:0; left:0; right:0; height:3px;
    background: linear-gradient(90deg,
        #2a6348 0%, #176070 33%, #7c5210 66%, #3e38a0 100%);
}
.hdr-eye {
    font-size: .65rem; font-weight: 600;
    letter-spacing: .22em; text-transform: uppercase;
    color: var(--green); margin: 0 0 8px; display: block;
}
.hdr-title {
    font-family: var(--f-head) !important;
    font-size: 2.6rem; font-weight: 400;
    color: var(--ink); line-height: 1.06;
    margin: 0 0 10px; letter-spacing: -.01em;
}
.hdr-title em { font-style: italic; color: var(--green); }
.hdr-sub {
    font-size: .875rem; font-weight: 400;
    color: var(--ink-2); line-height: 1.65;
    margin: 0 0 18px; max-width: 540px;
}
.chips { display:flex; flex-wrap:wrap; gap:6px; }
.chip {
    font-size: .62rem; font-weight: 600;
    letter-spacing: .1em; text-transform: uppercase;
    padding: 3px 10px; border-radius: 20px;
    border: 1px solid; display: inline-block; line-height: 1.6;
}
.cg { background:var(--green-lt); color:var(--green); border-color:var(--green-bd); }
.ct { background:var(--teal-lt);  color:var(--teal);  border-color:var(--teal-bd);  }
.ca { background:var(--amber-lt); color:var(--amber); border-color:var(--amber-bd); }
.cv { background:var(--violet-lt);color:var(--violet);border-color:var(--violet-bd);}

/* ── Content area ── */
.content { padding: 28px 40px 0; }

/* ── Section heading used above output blocks ── */
.sec-head {
    font-size: .62rem; font-weight: 600;
    letter-spacing: .2em; text-transform: uppercase;
    color: var(--ink-3); margin: 0 0 6px;
    display: block;
}

/* ── Ruled divider ── */
.rule {
    border: none;
    border-top: 1px solid var(--line);
    margin: 22px 0;
}

/* ── Accent rule — colored top border as section indicator ── */
.accent-g { border-top: 2px solid var(--green); margin: 0 0 14px; border-bottom:none; }
.accent-t { border-top: 2px solid var(--teal);  margin: 0 0 14px; border-bottom:none; }
.accent-a { border-top: 2px solid var(--amber); margin: 0 0 14px; border-bottom:none; }
.accent-v { border-top: 2px solid var(--violet);margin: 0 0 14px; border-bottom:none; }

/* ── Textarea ── */
div[data-testid="stTextArea"] textarea {
    background: var(--bg-2) !important;
    border: 1px solid var(--line-2) !important;
    border-radius: 8px !important;
    font-family: var(--f-body) !important;
    font-size: .9rem !important;
    color: var(--ink) !important;
    line-height: 1.7 !important;
    transition: border-color .15s !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: var(--green) !important;
    box-shadow: 0 0 0 3px rgba(42,99,72,.1) !important;
    outline: none !important;
}
div[data-testid="stTextArea"] textarea::placeholder {
    color: var(--ink-3) !important;
}

/* ── Select box (length) ── */
div[data-testid="stSelectbox"] > div > div {
    background: var(--bg-2) !important;
    border: 1px solid var(--line-2) !important;
    border-radius: 8px !important;
    font-family: var(--f-body) !important;
    font-size: .88rem !important;
    color: var(--ink) !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--green) !important;
    box-shadow: 0 0 0 3px rgba(42,99,72,.1) !important;
}
div[data-testid="stSelectbox"] label {
    font-size: .62rem !important;
    font-weight: 600 !important;
    letter-spacing: .18em !important;
    text-transform: uppercase !important;
    color: var(--ink-3) !important;
}

/* ── File uploader ── */
div[data-testid="stFileUploader"] section {
    background: var(--bg-2) !important;
    border: 1.5px dashed var(--line-2) !important;
    border-radius: 8px !important;
    transition: border-color .15s, background .15s !important;
}
div[data-testid="stFileUploader"] section:hover {
    border-color: var(--green) !important;
    background: var(--green-lt) !important;
}
div[data-testid="stFileUploader"] label {
    font-size: .62rem !important;
    font-weight: 600 !important;
    letter-spacing: .18em !important;
    text-transform: uppercase !important;
    color: var(--ink-3) !important;
}

/* ── Buttons ── */
div[data-testid="stButton"] > button {
    font-family: var(--f-body) !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    width: 100% !important;
    transition: all .15s !important;
    letter-spacing: .03em !important;
}
div[data-testid="stButton"] > button[kind="primary"] {
    background: var(--green) !important;
    color: #fff !important;
    border: none !important;
    padding: 12px 20px !important;
    font-size: .85rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.15), 0 2px 8px rgba(42,99,72,.2) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #1e4d38 !important;
    box-shadow: 0 2px 6px rgba(0,0,0,.15), 0 4px 14px rgba(42,99,72,.28) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:active {
    transform: translateY(0) !important;
}
div[data-testid="stButton"] > button[kind="secondary"] {
    background: var(--bg-2) !important;
    border: 1px solid var(--line-2) !important;
    color: var(--ink-2) !important;
    font-size: .82rem !important;
    padding: 9px 20px !important;
}
div[data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: var(--green) !important;
    color: var(--green) !important;
    background: var(--green-lt) !important;
}

/* ── st.metric ── */
div[data-testid="stMetric"] {
    background: var(--bg-2) !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
}
div[data-testid="stMetric"] label {
    font-size: .65rem !important;
    font-weight: 600 !important;
    letter-spacing: .15em !important;
    text-transform: uppercase !important;
    color: var(--ink-3) !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: var(--f-mono) !important;
    font-size: 1.35rem !important;
    color: var(--ink) !important;
    font-weight: 500 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-size: .75rem !important;
}

/* ── Expander ── */
div[data-testid="stExpander"] {
    background: var(--bg-2) !important;
    border: 1px solid var(--line) !important;
    border-radius: 8px !important;
}
div[data-testid="stExpander"] summary {
    font-size: .82rem !important;
    font-weight: 600 !important;
    color: var(--ink-2) !important;
    padding: 10px 16px !important;
}

/* ── Info / success / warning boxes ── */
div[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-size: .88rem !important;
}

/* ── Inline tag pill ── */
.tag {
    display: inline-block;
    font-size: .73rem; font-weight: 500;
    padding: 2px 9px; border-radius: 20px;
    margin: 0 3px 4px 0;
    background: var(--green-lt);
    border: 1px solid var(--green-bd);
    color: var(--green);
    font-family: var(--f-body);
}

/* ── Monospace value badge ── */
.val-t {
    display: inline-block;
    font-family: var(--f-mono); font-size: .78rem;
    padding: 1px 7px; border-radius: 4px;
    background: var(--teal-lt);
    border: 1px solid var(--teal-bd);
    color: var(--teal);
}
.val-a {
    display: inline-block;
    font-family: var(--f-mono); font-size: .78rem;
    padding: 1px 7px; border-radius: 4px;
    background: var(--amber-lt);
    border: 1px solid var(--amber-bd);
    color: var(--amber);
}

/* ── Sentence highlight ── */
mark.hl {
    background: #c6ebd6;
    border-bottom: 2px solid #68c88e;
    border-radius: 2px;
    padding: 0 2px;
    color: var(--ink);
    cursor: help;
}

/* ── Result output text ── */
.out {
    font-size: .92rem; font-weight: 400;
    color: var(--ink); line-height: 1.8;
    font-family: var(--f-body);
}
.out p { margin: 0 0 .6em; }

/* ── Placeholder ── */
.ph {
    font-size: .85rem; font-style: italic;
    color: var(--ink-3); font-family: var(--f-body);
}

/* ── Stat block for reading level ── */
.stat {
    display: flex; align-items: baseline;
    gap: 8px; flex-wrap: wrap;
    margin-bottom: 10px;
}
.stat-lbl {
    font-size: .75rem; font-weight: 600;
    color: var(--ink-2); min-width: 100px;
    font-family: var(--f-body);
}
.stat-note {
    font-size: .72rem; color: var(--ink-3);
    font-family: var(--f-body);
}
.stat-delta {
    font-size: .72rem; font-weight: 600;
    color: var(--green); font-family: var(--f-body);
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
    text = ' '.join(text.split())
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
    doc  = nlp(src[:8000])
    sens = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 20]
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

# Initialize clear flag
if "do_clear" not in st.session_state:
    st.session_state.do_clear = False

with st.sidebar:
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e6e3da !important;
        min-width: 320px !important;
        max-width: 360px !important;
    }
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] > div > div,
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding: 186px 20px 40px !important;
    }
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] button {
        font-family: 'Inter', system-ui, sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-bottom:18px'>
        <p style='font-size:1rem;font-weight:600;color:#161510;margin:0'>
            Start here
        </p>
    </div>
    <hr style='border:none;border-top:1px solid #e6e3da;margin:0 0 18px'>
    """, unsafe_allow_html=True)

    # ── All inputs — read every rerun, store in session state ─────────────────
    # Using session state storage means the Summarize button always reads
    # the latest values regardless of which rerun triggered it.

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

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<hr style='border:none;border-top:1px solid #e6e3da;margin:0 0 16px'>",
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

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<hr style='border:none;border-top:1px solid #e6e3da;margin:0 0 20px'>",
        unsafe_allow_html=True,
    )

    # ── Buttons — side by side, Summarize dominant ────────────────────────────
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

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<hr style='border:none;border-top:1px solid #e6e3da;margin:0 0 14px'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:.72rem;color:#9a9890;line-height:1.65'>"
        "Powered by a fine-tuned FLAN-T5 model · "
        "<a href='https://huggingface.co/harao-ml/flant5-finetuned-summarize' "
        "target='_blank' style='color:#2a6348;text-decoration:none'>"
        "harao-ml/flant5-finetuned-summarize ↗</a>"
        "</p>",
        unsafe_allow_html=True,
    )

# ── Clear handler ─────────────────────────────────────────────────────────────
if st.session_state.do_clear:
    st.session_state.do_clear     = False
    st.session_state.do_summarize = False
    # Reset all results
    for k in RESULT_KEYS:
        st.session_state[k] = None
    # Reset pending input mirrors
    st.session_state.pending_text   = ""
    st.session_state.pending_file   = None
    st.session_state.pending_length = "Balanced"
    # Increment file uploader key — forces Streamlit to render a brand new
    # file_uploader widget, which is the only reliable way to clear it visually
    st.session_state.file_uploader_key += 1
    # Delete static widget keys so they reset to defaults
    for k in INPUT_KEYS:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

# ── Summarize handler ─────────────────────────────────────────────────────────
# Always reads from session state mirrors — consistent for both input types
if st.session_state.do_summarize:
    st.session_state.do_summarize = False

    content  = None
    err      = None
    url_meta = None

    # Read from session state mirrors — not from widget variables directly
    active_file   = st.session_state.pending_file
    active_text   = st.session_state.pending_text.strip()
    active_length = st.session_state.pending_length
    min_len, max_len = LENGTH_MAP[active_length]

    # ── Resolve content ───────────────────────────────────────────────────────
    # Priority: text/URL wins if the user has typed anything.
    # File only used when text box is empty.
    # This prevents a stale uploaded file from shadowing freshly typed input.
    if active_text:
        # Text or URL input — ignore any uploaded file
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
        # File upload — only reached when text box is empty
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

    # ── Run pipeline ──────────────────────────────────────────────────────────
    if err:
        st.session_state.error = err
        # Clear any stale results so the error shows cleanly
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
# MAIN — Output area (full width, stacked sections)
# ══════════════════════════════════════════════════════════════════════════════
main = st.container()

with main:
    st.markdown('<div class="content">', unsafe_allow_html=True)

    if st.session_state.error:
        st.error(st.session_state.error)

    # ── Empty state call-to-action ────────────────────────────────────────────
    if not st.session_state.summary and not st.session_state.error:
        st.markdown("""
        <div style='
            margin: 40px auto 48px;
            max-width: 520px;
            text-align: center;
            padding: 48px 40px;
            background: #ffffff;
            border: 1px solid #e6e3da;
            border-radius: 14px;
        '>
            <div style='
                font-size: 2.4rem;
                margin-bottom: 16px;
                line-height: 1;
            '>📄</div>
            <p style='
                font-family: DM Serif Display, Georgia, serif;
                font-size: 1.45rem;
                font-weight: 400;
                color: #161510;
                margin: 0 0 10px;
                line-height: 1.25;
            '>Ready to summarize</p>
            <p style='
                font-size: .88rem;
                color: #4a4840;
                margin: 0 0 24px;
                line-height: 1.65;
            '>
                Use the panel on the left to paste text, drop in a URL,
                or upload a PDF or TXT file. Then click
                <strong style="color:#2a6348">Summarize</strong>
            </p>
            <div style='
                display: flex;
                flex-direction: column;
                gap: 10px;
                text-align: left;
            '>
                <div style='
                    display: flex; align-items: flex-start; gap: 12px;
                    background: #f8f7f4; border-radius: 8px; padding: 12px 14px;
                '>
                    <span style='font-size:1rem;margin-top:1px'></span>
                    <div>
                        <p style='font-size:.8rem;font-weight:600;color:#161510;margin:0 0 2px'>
                            Paste text or a URL
                        </p>
                        <p style='font-size:.75rem;color:#9a9890;margin:0;line-height:1.5'>
                            Article links, research content, any long-form text
                        </p>
                    </div>
                </div>
                <div style='
                    display: flex; align-items: flex-start; gap: 12px;
                    background: #f8f7f4; border-radius: 8px; padding: 12px 14px;
                '>
                    <span style='font-size:1rem;margin-top:1px'></span>
                    <div>
                        <p style='font-size:.8rem;font-weight:600;color:#161510;margin:0 0 2px'>
                            Upload a file
                        </p>
                        <p style='font-size:.75rem;color:#9a9890;margin:0;line-height:1.5'>
                            PDF or TXT — up to full document length
                        </p>
                    </div>
                </div>
                <div style='
                    display: flex; align-items: flex-start; gap: 12px;
                    background: #eef6f1; border: 1px solid #b8ddc8;
                    border-radius: 8px; padding: 12px 14px;
                '>
                    <span style='font-size:1rem;margin-top:1px'></span>
                    <div>
                        <p style='font-size:.8rem;font-weight:600;color:#2a6348;margin:0 0 2px'>
                            Click Summarize
                        </p>
                        <p style='font-size:.75rem;color:#4a7a60;margin:0;line-height:1.5'>
                            Runs summarization, NER, ROUGE scoring,
                            sentence highlighting & reading level analysis
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
                "<p style='font-size:.68rem;font-weight:600;letter-spacing:.12em;"
                "text-transform:uppercase;color:#9a9890;margin:0 0 8px'>Key Phrases</p>",
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
                "<p style='font-size:.68rem;font-weight:600;letter-spacing:.12em;"
                "text-transform:uppercase;color:#9a9890;margin:0 0 8px'>Named Entities</p>",
                unsafe_allow_html=True,
            )
            if ents:
                for lb, items in ents.items():
                    name = lmap.get(lb, lb)
                    st.markdown(
                        f"<p style='font-size:.87rem;margin:0 0 5px;line-height:1.5'>"
                        f"<span style='font-weight:600;color:#4a4840'>{name}</span>"
                        f"<span style='color:#9a9890'> — </span>"
                        f"<span style='color:#161510'>{', '.join(items)}</span></p>",
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
                    f"<p style='font-size:.87rem;margin:0 0 10px;line-height:1.5'>"
                    f"<span style='font-weight:600;color:#4a4840;display:inline-block;width:30px'>{label}</span>"
                    f"&nbsp;<span class='val-t'>{sc.fmeasure:.3f}</span>"
                    f"&nbsp;<span style='font-size:.75rem;color:#9a9890;font-family:var(--f-mono)'>"
                    f"{ubar(sc.fmeasure)}</span>"
                    f"<br><span style='font-size:.72rem;color:#9a9890;margin-left:34px'>"
                    f"Precision&thinsp;<span class='val-t'>{sc.precision:.3f}</span>"
                    f"&nbsp;&nbsp;Recall&thinsp;<span class='val-t'>{sc.recall:.3f}</span>"
                    f"</span></p>"
                )

            st.markdown(rrow("R-1", r1), unsafe_allow_html=True)
            st.markdown(rrow("R-2", r2), unsafe_allow_html=True)
            st.markdown(rrow("R-L", rL), unsafe_allow_html=True)
            st.markdown(
                "<p style='font-size:.72rem;color:#9a9890;font-style:italic;margin-top:6px'>"
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
            "<p style='font-size:.75rem;color:#9a9890;font-style:italic;margin:0 0 12px'>"
            "<mark class='hl' style='font-style:normal;font-size:.72rem;padding:1px 5px'>"
            "Highlighted</mark>&nbsp; sentences contributed most to the summary</p>",
            unsafe_allow_html=True,
        )
        parts = []
        for sent, is_hl, score in st.session_state.highlight:
            esc = sent.replace("<","&lt;").replace(">","&gt;")
            if is_hl:
                parts.append(f'<mark class="hl" title="Relevance: {score:.3f}">{esc}</mark>')
            else:
                parts.append(f'<span>{esc}</span>')
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
    st.markdown('<span class="sec-head">Reading Level & Compression</span>',
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
            f"<p style='font-size:.75rem;color:#9a9890;margin-top:10px'>"
            f"Source: {r['gls']} ({r['els']}) &nbsp;·&nbsp; "
            f"Summary: {r['glu']} ({r['elu']}) &nbsp;·&nbsp; "
            f"Flesch-Kincaid: lower grade = more accessible</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<p class='ph'>Reading level comparison will appear here.</p>",
                    unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)