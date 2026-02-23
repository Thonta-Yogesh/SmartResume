"""
SmartResume AI — Intelligent Resume Scanner for Recruiters
RAG-powered semantic search using NLP embeddings.
"""
import os
import streamlit as st
from resume_engine import ResumeEngine


# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="SmartResume AI — Resume Scanner",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─── Initialize Engine (cached) ──────────────────────────────
@st.cache_resource
def get_engine():
    return ResumeEngine()

engine = get_engine()


# ─── Premium CSS — Dark Glassmorphism Theme ───────────────────
st.markdown("""
<style>
    /* ═══════════════════════════════════════════════════════
       FONTS & GLOBAL RESET
       ═══════════════════════════════════════════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: rgba(255,255,255,0.03);
        --bg-card-hover: rgba(255,255,255,0.06);
        --glass-border: rgba(255,255,255,0.06);
        --glass-border-hover: rgba(255,255,255,0.12);
        --text-primary: #f0f0f5;
        --text-secondary: #8b8fa3;
        --text-muted: #5a5e72;
        --accent-primary: #7c5cfc;
        --accent-secondary: #5eead4;
        --accent-warm: #f59e0b;
        --gradient-main: linear-gradient(135deg, #7c5cfc 0%, #5eead4 100%);
        --gradient-warm: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
        --gradient-cool: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
        --shadow-lg: 0 20px 60px rgba(0,0,0,0.4);
        --shadow-glow: 0 0 40px rgba(124,92,252,0.15);
        --radius-sm: 10px;
        --radius-md: 16px;
        --radius-lg: 24px;
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: var(--bg-primary) !important;
        color: var(--text-primary);
    }
    #MainMenu, footer, header { visibility: hidden; }

    /* Scrollbar Styling */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(124,92,252,0.3);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(124,92,252,0.5);
    }

    /* Sidebar toggle (expand arrow) — always visible */
    button[data-testid="stExpandSidebarButton"] {
        color: #a78bfa !important;
        background: rgba(124,92,252,0.18) !important;
        border: 1px solid rgba(124,92,252,0.35) !important;
        border-radius: 10px !important;
        transition: var(--transition) !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 999999 !important;
    }
    button[data-testid="stExpandSidebarButton"]:hover {
        background: rgba(124,92,252,0.3) !important;
        border-color: rgba(124,92,252,0.55) !important;
        transform: scale(1.1) !important;
    }
    button[data-testid="stExpandSidebarButton"] svg {
        fill: #a78bfa !important;
        stroke: #a78bfa !important;
    }
    /* Collapse arrow inside sidebar header */
    button[data-testid="stSidebarCollapseButton"] {
        color: #a78bfa !important;
    }
    button[data-testid="stSidebarCollapseButton"]:hover {
        background: rgba(124,92,252,0.12) !important;
    }

    /* ═══════════════════════════════════════════════════════
       SIDEBAR — Sleek Dark Panel
       ═══════════════════════════════════════════════════════ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d14 0%, #111119 40%, #0d0d14 100%) !important;
        border-right: 1px solid rgba(124,92,252,0.1);
    }
    section[data-testid="stSidebar"] * {
        color: #c8c8e0 !important;
    }
    section[data-testid="stSidebar"] .stButton button {
        background: var(--gradient-cool) !important;
        color: #fff !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.55rem 1.2rem !important;
        letter-spacing: 0.02em !important;
        transition: var(--transition) !important;
        box-shadow: 0 4px 15px rgba(99,102,241,0.25) !important;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 8px 30px rgba(99,102,241,0.4) !important;
        filter: brightness(1.1) !important;
    }
    section[data-testid="stSidebar"] .stButton button:active {
        transform: translateY(0) scale(0.99) !important;
    }

    /* Sidebar file uploader */
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
        border: 1px dashed rgba(124,92,252,0.25) !important;
        border-radius: var(--radius-md) !important;
        background: rgba(124,92,252,0.04) !important;
        transition: var(--transition) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
        border-color: rgba(124,92,252,0.45) !important;
        background: rgba(124,92,252,0.08) !important;
    }

    /* ─── Sidebar Brand ─── */
    .sb-brand {
        text-align: center;
        padding: 1.5rem 1rem 1rem;
        margin-bottom: 0.5rem;
        position: relative;
    }
    .sb-brand::after {
        content: '';
        position: absolute;
        bottom: 0; left: 15%; right: 15%;
        height: 1px;
        background: linear-gradient(90deg,
            transparent 0%, rgba(124,92,252,0.4) 50%, transparent 100%);
    }
    .sb-brand .brand-icon {
        font-size: 2.2rem;
        display: block;
        margin-bottom: 0.3rem;
        filter: drop-shadow(0 0 12px rgba(124,92,252,0.5));
    }
    .sb-brand h2 {
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: var(--gradient-main);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .sb-brand p {
        font-size: 0.7rem;
        color: var(--text-muted) !important;
        margin: 0.3rem 0 0;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        font-weight: 500;
    }

    /* Sidebar Divider */
    .sb-div {
        border: none;
        height: 1px;
        background: linear-gradient(90deg,
            transparent, rgba(124,92,252,0.15), transparent);
        margin: 1.2rem 0;
    }

    /* Feature Pills */
    .feat-row { text-align: center; margin-bottom: 0.6rem; }
    .feat-pill {
        display: inline-block;
        background: rgba(124,92,252,0.08);
        border: 1px solid rgba(124,92,252,0.15);
        border-radius: 20px;
        padding: 0.25rem 0.65rem;
        font-size: 0.68rem;
        color: #a78bfa !important;
        margin: 0.15rem;
        font-weight: 500;
        letter-spacing: 0.01em;
        transition: var(--transition);
    }
    .feat-pill:hover {
        background: rgba(124,92,252,0.15);
        border-color: rgba(124,92,252,0.3);
        transform: translateY(-1px);
    }

    /* Resume List Items */
    .resume-item {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: var(--radius-sm);
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.4rem;
        transition: var(--transition);
    }
    .resume-item:hover {
        background: rgba(124,92,252,0.06);
        border-color: rgba(124,92,252,0.15);
        transform: translateX(3px);
    }
    .resume-item .ri-name {
        font-weight: 600;
        font-size: 0.82rem;
        color: var(--text-primary) !important;
    }
    .resume-item .ri-meta {
        font-size: 0.68rem;
        color: var(--text-muted) !important;
        margin-top: 0.15rem;
    }

    /* ═══════════════════════════════════════════════════════
       HERO SECTION
       ═══════════════════════════════════════════════════════ */
    .hero-section {
        text-align: center !important;
        padding: 2.5rem 1rem 1rem;
        position: relative;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50px; left: 50%;
        transform: translateX(-50%);
        width: 500px; height: 500px;
        background: radial-gradient(circle,
            rgba(124,92,252,0.08) 0%,
            transparent 70%);
        pointer-events: none;
        z-index: 0;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(124,92,252,0.1);
        border: 1px solid rgba(124,92,252,0.2);
        border-radius: 25px;
        padding: 0.35rem 1rem;
        font-size: 0.75rem;
        color: #a78bfa;
        font-weight: 500;
        letter-spacing: 0.03em;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }
    .hero-badge .pulse-dot {
        width: 7px; height: 7px;
        background: #5eead4;
        border-radius: 50%;
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(94,234,212,0.4); }
        50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(94,234,212,0); }
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin: 0 0 0.6rem;
        position: relative;
        z-index: 1;
    }
    .hero-title .gradient-text {
        background: var(--gradient-main);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s ease-in-out infinite;
        background-size: 200% auto;
    }
    @keyframes shimmer {
        0%, 100% { background-position: 0% center; }
        50% { background-position: 200% center; }
    }

    .hero-subtitle {
        color: #9ca3be;
        font-size: 0.95rem;
        font-weight: 400;
        max-width: 520px;
        margin: 0 auto;
        line-height: 1.7;
        position: relative;
        z-index: 1;
        letter-spacing: 0.01em;
        text-align: center !important;
        display: block;
        width: 100%;
    }
    .hero-subtitle strong {
        color: #c4c9e0;
        font-weight: 500;
    }

    /* ═══════════════════════════════════════════════════════
       STATS CARDS — Glassmorphism
       ═══════════════════════════════════════════════════════ */
    .stats-grid {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 1.8rem auto 2rem;
        flex-wrap: wrap;
        max-width: 750px;
    }
    .stat-card {
        flex: 1;
        min-width: 140px;
        max-width: 180px;
        background: var(--bg-card);
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-md);
        padding: 1.2rem 1rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: var(--transition);
        cursor: default;
    }
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: var(--gradient-main);
        opacity: 0;
        transition: opacity 0.3s;
    }
    .stat-card:hover {
        background: var(--bg-card-hover);
        border-color: var(--glass-border-hover);
        transform: translateY(-4px);
        box-shadow: var(--shadow-glow);
    }
    .stat-card:hover::before { opacity: 1; }
    .stat-card .s-icon {
        font-size: 1.4rem;
        margin-bottom: 0.3rem;
        display: block;
    }
    .stat-card .s-val {
        font-size: 1.4rem;
        font-weight: 800;
        color: var(--text-primary);
        margin: 0.2rem 0;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.02em;
    }
    .stat-card .s-lbl {
        font-size: 0.7rem;
        color: var(--text-muted);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ═══════════════════════════════════════════════════════
       SEARCH SECTION
       ═══════════════════════════════════════════════════════ */
    .search-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.8rem;
    }
    .search-header h3 {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
    }
    .search-header .search-line {
        flex: 1; height: 1px;
        background: linear-gradient(90deg,
            rgba(124,92,252,0.3), transparent);
    }

    /* Streamlit text area override */
    .stApp .stTextArea textarea {
        background: var(--bg-card) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        padding: 1rem !important;
        transition: var(--transition) !important;
        resize: none !important;
    }
    .stApp .stTextArea textarea:focus {
        border-color: rgba(124,92,252,0.4) !important;
        box-shadow: 0 0 0 3px rgba(124,92,252,0.1),
                    0 0 30px rgba(124,92,252,0.05) !important;
    }
    .stApp .stTextArea textarea::placeholder {
        color: var(--text-muted) !important;
    }
    .stApp .stTextArea label {
        color: var(--text-secondary) !important;
    }

    /* Search controls row */
    .search-controls {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin-top: 0.8rem;
    }
    .search-controls .control-label {
        font-size: 0.78rem;
        color: var(--text-muted);
        font-weight: 500;
        white-space: nowrap;
        letter-spacing: 0.02em;
    }

    /* Select box — compact inline */
    .stApp .stSelectbox {
        max-width: 100px;
    }
    .stApp .stSelectbox > div > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        min-height: 42px !important;
    }
    .stApp .stSelectbox label {
        color: var(--text-secondary) !important;
        font-size: 0.78rem !important;
    }

    /* Main search button — premium gradient */
    .stApp .stButton button[kind="primary"] {
        background: var(--gradient-cool) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        padding: 0.65rem 2rem !important;
        letter-spacing: 0.03em !important;
        transition: var(--transition) !important;
        box-shadow: 0 4px 20px rgba(99,102,241,0.3) !important;
        white-space: nowrap !important;
    }
    .stApp .stButton button[kind="primary"]:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 8px 35px rgba(99,102,241,0.45) !important;
        filter: brightness(1.1) !important;
    }
    .stApp .stButton button[kind="primary"]:active {
        transform: translateY(0) scale(0.99) !important;
    }

    /* ═══════════════════════════════════════════════════════
       RESULT CARDS — Premium Glass Design
       ═══════════════════════════════════════════════════════ */
    .results-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .results-count {
        background: rgba(124,92,252,0.12);
        color: #a78bfa;
        padding: 0.2rem 0.7rem;
        border-radius: 15px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .result-card {
        background: var(--bg-card);
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-lg);
        padding: 1.6rem;
        margin-bottom: 1rem;
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: var(--gradient-main);
        opacity: 0;
        transition: opacity 0.3s;
    }
    .result-card:hover {
        background: var(--bg-card-hover);
        border-color: var(--glass-border-hover);
        transform: translateY(-3px);
        box-shadow: var(--shadow-glow);
    }
    .result-card:hover::before { opacity: 1; }

    /* Result Header */
    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 0.8rem;
    }
    .result-rank {
        font-size: 2rem;
        margin-right: 0.8rem;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
    }
    .result-name-group {
        flex: 1;
    }
    .result-name {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
        letter-spacing: -0.01em;
    }
    .result-file {
        font-size: 0.78rem;
        color: var(--text-muted);
        font-family: 'JetBrains Mono', monospace;
        margin-top: 0.15rem;
    }

    /* Match Badge */
    .match-badge {
        padding: 0.35rem 1rem;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.02em;
        white-space: nowrap;
    }
    .match-badge.tier-s {
        background: linear-gradient(135deg, rgba(94,234,212,0.15), rgba(94,234,212,0.05));
        color: #5eead4;
        border: 1px solid rgba(94,234,212,0.3);
        box-shadow: 0 0 20px rgba(94,234,212,0.1);
    }
    .match-badge.tier-a {
        background: linear-gradient(135deg, rgba(124,92,252,0.15), rgba(124,92,252,0.05));
        color: #a78bfa;
        border: 1px solid rgba(124,92,252,0.3);
    }
    .match-badge.tier-b {
        background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(245,158,11,0.05));
        color: #fbbf24;
        border: 1px solid rgba(245,158,11,0.3);
    }
    .match-badge.tier-c {
        background: rgba(255,255,255,0.04);
        color: var(--text-secondary);
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Result Meta Row */
    .result-meta {
        display: flex;
        gap: 1.5rem;
        margin: 0.6rem 0;
        font-size: 0.82rem;
        color: var(--text-secondary);
        flex-wrap: wrap;
    }
    .result-meta span {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }

    /* Skill Pills */
    .skills-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.3rem;
        margin: 0.7rem 0;
    }
    .skill-pill {
        display: inline-block;
        background: rgba(124,92,252,0.08);
        border: 1px solid rgba(124,92,252,0.15);
        border-radius: 20px;
        padding: 0.2rem 0.65rem;
        font-size: 0.72rem;
        color: #a78bfa;
        font-weight: 500;
        transition: var(--transition);
    }
    .skill-pill:hover {
        background: rgba(124,92,252,0.15);
        border-color: rgba(124,92,252,0.3);
        transform: translateY(-1px);
    }

    /* Chunk Preview */
    .chunk-preview {
        background: rgba(124,92,252,0.03);
        border-left: 3px solid rgba(124,92,252,0.3);
        padding: 0.9rem 1.1rem;
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 0.8rem;
        line-height: 1.65;
        max-height: 110px;
        overflow-y: auto;
        font-family: 'Inter', sans-serif;
    }
    .chunk-preview strong {
        color: var(--text-primary);
        font-weight: 600;
    }

    /* Action Buttons */
    .action-row {
        display: flex;
        gap: 0.5rem;
        margin-top: 1rem;
        flex-wrap: wrap;
        align-items: center;
    }
    .action-btn {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.4rem 0.9rem;
        border-radius: 10px;
        font-size: 0.78rem;
        font-weight: 600;
        text-decoration: none;
        transition: var(--transition);
        cursor: pointer;
        border: none;
        letter-spacing: 0.01em;
    }
    .action-btn:hover {
        transform: translateY(-2px);
        filter: brightness(1.1);
    }
    .action-btn.email-btn {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: #fff !important;
        box-shadow: 0 3px 12px rgba(99,102,241,0.3);
    }
    .action-btn.phone-btn {
        background: linear-gradient(135deg, #10b981, #059669);
        color: #fff !important;
        box-shadow: 0 3px 12px rgba(16,185,129,0.3);
    }
    .action-btn.download-btn {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: #fff !important;
        box-shadow: 0 3px 12px rgba(245,158,11,0.3);
    }

    /* Download button override */
    .stApp .stDownloadButton button {
        background: rgba(245,158,11,0.1) !important;
        color: #fbbf24 !important;
        border: 1px solid rgba(245,158,11,0.2) !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        transition: var(--transition) !important;
    }
    .stApp .stDownloadButton button:hover {
        background: rgba(245,158,11,0.18) !important;
        border-color: rgba(245,158,11,0.35) !important;
        transform: translateY(-1px) !important;
    }

    /* ═══════════════════════════════════════════════════════
       EMPTY STATE
       ═══════════════════════════════════════════════════════ */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
    }
    .empty-state .empty-icon {
        font-size: 3rem;
        margin-bottom: 0.8rem;
        filter: grayscale(0.5) opacity(0.6);
    }
    .empty-state p {
        color: var(--text-muted);
        font-size: 0.9rem;
    }

    /* ═══════════════════════════════════════════════════════
       SEPARATOR
       ═══════════════════════════════════════════════════════ */
    .section-sep {
        border: none;
        height: 1px;
        background: linear-gradient(90deg,
            transparent, rgba(124,92,252,0.12), transparent);
        margin: 1.5rem 0;
    }

    /* ═══════════════════════════════════════════════════════
       FOOTER
       ═══════════════════════════════════════════════════════ */
    .footer {
        text-align: center;
        padding: 1.5rem 1rem 1rem;
        margin-top: 2rem;
    }
    .footer-brand {
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--text-muted);
        letter-spacing: 0.03em;
    }
    .footer-brand span {
        background: var(--gradient-main);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .footer-tech {
        display: flex;
        justify-content: center;
        gap: 0.8rem;
        margin-top: 0.5rem;
        flex-wrap: wrap;
    }
    .footer-tag {
        font-size: 0.68rem;
        color: var(--text-muted);
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-weight: 500;
    }

    /* ═══════════════════════════════════════════════════════
       STREAMLIT OVERRIDES
       ═══════════════════════════════════════════════════════ */
    .stApp hr { border-color: rgba(255,255,255,0.05) !important; }
    .stApp .stMarkdown h3, .stApp .stMarkdown h5 {
        color: var(--text-primary) !important;
    }
    .stApp .stAlert { border-radius: var(--radius-md) !important; }
    .stApp .stSpinner > div { color: var(--accent-primary) !important; }
    .stApp .stProgress > div > div {
        background-color: var(--accent-primary) !important;
    }

    /* Warning / Info boxes */
    .stApp [data-testid="stNotification"] {
        background: rgba(124,92,252,0.06) !important;
        border: 1px solid rgba(124,92,252,0.15) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
    }

    /* Tabs */
    .stApp .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stApp .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
    }
    .stApp .stTabs [aria-selected="true"] {
        background: rgba(124,92,252,0.1) !important;
        color: #a78bfa !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <span class="brand-icon">🧠</span>
        <h2>SmartResume</h2>
        <p>AI-Powered Recruiter Suite</p>
    </div>
    """, unsafe_allow_html=True)

    # Feature pills
    st.markdown("""
    <div class="feat-row">
        <span class="feat-pill">📄 PDF / DOCX / TXT</span>
        <span class="feat-pill">🧬 768d Embeddings</span>
        <span class="feat-pill">🔍 Semantic Search</span>
        <span class="feat-pill">🤖 RAG Pipeline</span>
        <span class="feat-pill">🧠 spaCy NER</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)

    # Upload section
    st.markdown("##### 📤  Upload Resumes")
    uploaded_files = st.file_uploader(
        "Drop resume files here",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        if st.button("⚡ Process Uploaded Resumes", use_container_width=True):
            for ufile in uploaded_files:
                try:
                    progress_bar = st.progress(0, text=f"Processing {ufile.name}...")
                    def update_progress(val, msg):
                        progress_bar.progress(val, text=msg)
                    engine.process_resume(ufile.name, ufile.read(), progress_callback=update_progress)
                    st.success(f"✅ {ufile.name}")
                except Exception as e:
                    st.error(f"❌ {ufile.name}: {e}")
            st.cache_resource.clear()
            st.rerun()

    # Local resumes
    st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)
    resumes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resumes")
    if os.path.exists(resumes_dir):
        local_files = [f for f in os.listdir(resumes_dir) if f.lower().endswith(('.pdf', '.docx', '.txt'))]
        if local_files:
            st.markdown(f"##### 📁  Local Resumes · {len(local_files)} files")
            if st.button("⚡ Process All Local Resumes", use_container_width=True):
                progress_bar = st.progress(0, text="Starting...")
                for i, fname in enumerate(local_files):
                    fpath = os.path.join(resumes_dir, fname)
                    try:
                        with open(fpath, "rb") as f:
                            file_bytes = f.read()
                        def update_progress(val, msg):
                            overall = (i + val) / len(local_files)
                            progress_bar.progress(overall, text=f"[{i+1}/{len(local_files)}] {fname}: {msg}")
                        engine.process_resume(fname, file_bytes, progress_callback=update_progress)
                    except Exception as e:
                        st.warning(f"⚠️ {fname}: {e}")
                progress_bar.progress(1.0, text="All done!")
                st.success(f"✅ Processed {len(local_files)} resume files!")
                st.cache_resource.clear()
                st.rerun()

    # Database summary
    st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)
    all_resumes = engine.get_all_resumes()
    st.markdown(f"##### 🗄️  Resume Database · {len(all_resumes)}")

    if all_resumes:
        for r in all_resumes:
            skills_preview = ", ".join(r["skills"][:3])
            if len(r["skills"]) > 3:
                skills_preview += f" +{len(r['skills'])-3}"
            exp_str = f"{r['experience_years']:.0f}yr" if r["experience_years"] > 0 else ""
            st.markdown(f"""
            <div class="resume-item">
                <div class="ri-name">👤 {r['candidate_name']}</div>
                <div class="ri-meta">📄 {r['file_name']}{' · 📅 ' + exp_str if exp_str else ''}</div>
                <div class="ri-meta">🔧 {skills_preview if skills_preview else 'Processing...'}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)
        if st.button("🗑️  Clear All Resumes", use_container_width=True):
            engine.clear_all()
            st.cache_resource.clear()
            st.rerun()
    else:
        st.info("No resumes yet — upload files above to get started!")


# ─── Main Content ─────────────────────────────────────────────

# Hero Section
resume_count = engine.resume_count
st.markdown(f"""
<div class="hero-section">
    <div class="hero-badge">
        <span class="pulse-dot"></span>
        AI-Powered Resume Intelligence
    </div>
    <h1 class="hero-title">
        <span class="gradient-text">SmartResume</span> AI
    </h1>
    <p class="hero-subtitle">
        Describe your <strong>ideal candidate</strong> and let our NLP engine surface the
        best matches — powered by <strong>semantic embeddings</strong>, spaCy NER,
        and Retrieval-Augmented Generation.
    </p>
</div>
""", unsafe_allow_html=True)

# Stats Row
st.markdown(f"""
<div class="stats-grid">
    <div class="stat-card">
        <span class="s-icon">📄</span>
        <div class="s-val">{resume_count}</div>
        <div class="s-lbl">Resumes</div>
    </div>
    <div class="stat-card">
        <span class="s-icon">🧬</span>
        <div class="s-val">768d</div>
        <div class="s-lbl">Embeddings</div>
    </div>
    <div class="stat-card">
        <span class="s-icon">🤖</span>
        <div class="s-val">RAG</div>
        <div class="s-lbl">Pipeline</div>
    </div>
    <div class="stat-card">
        <span class="s-icon">🧠</span>
        <div class="s-val">NER</div>
        <div class="s-lbl">spaCy NLP</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Search Section ───────────────────────────────────────────

st.markdown("""
<div class="search-header">
    <h3>🔍 Search Resumes</h3>
    <div class="search-line"></div>
</div>
""", unsafe_allow_html=True)

query = st.text_area(
    "Enter job description or search query",
    placeholder="e.g., Looking for a Python developer with 3+ years in machine learning, "
                "familiar with TensorFlow, Docker, and AWS. Strong communication skills preferred.",
    height=120,
    label_visibility="collapsed"
)

top_k = st.selectbox("Results to show", [5, 10, 15, 20], index=0)
col_l, col_btn, col_r = st.columns([3, 2, 3])
with col_btn:
    search_pressed = st.button("🔍  Find Best Matches", use_container_width=True, type="primary")


# ─── Results ──────────────────────────────────────────────────

if search_pressed and query.strip():
    if resume_count == 0:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📂</div>
            <p>No resumes in the database yet.<br>Upload resumes using the sidebar to get started.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("🧬 Running semantic search with NLP embeddings..."):
            results = engine.search(query.strip(), top_k=top_k)

        if not results:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🔍</div>
                <p>No matching resumes found. Try a different job description.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="results-title">
                🏆 Top Matches
                <span class="results-count">{len(results)} found</span>
            </div>
            """, unsafe_allow_html=True)

            for idx, r in enumerate(results):
                score = r["match_score"]
                # Tier badges
                if score >= 50:
                    tier = "tier-s"
                elif score >= 40:
                    tier = "tier-a"
                elif score >= 30:
                    tier = "tier-b"
                else:
                    tier = "tier-c"

                rank_emoji = ["🥇", "🥈", "🥉"][idx] if idx < 3 else f"#{idx+1}"

                skills_html = "".join(
                    f'<span class="skill-pill">{s}</span>' for s in r["skills"][:10]
                )
                exp_str = (
                    f"{r['experience_years']:.0f} years"
                    if r["experience_years"] > 0
                    else "N/A"
                )

                chunk_preview = (
                    r["best_chunk"][:280] + "..."
                    if len(r["best_chunk"]) > 280
                    else r["best_chunk"]
                )

                # Action buttons
                actions_html = '<div class="action-row">'
                if r["email"]:
                    first_name = r["candidate_name"].split()[0]
                    actions_html += (
                        f'<a href="mailto:{r["email"]}" class="action-btn email-btn" '
                        f'target="_blank">📧 Email {first_name}</a>'
                    )
                if r["phone"]:
                    phone_clean = (
                        r["phone"].replace(" ", "").replace("-", "")
                        .replace("(", "").replace(")", "")
                    )
                    actions_html += (
                        f'<a href="tel:{phone_clean}" class="action-btn phone-btn">'
                        f'📞 {r["phone"]}</a>'
                    )
                actions_html += '</div>'

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-header">
                        <div style="display:flex; align-items:center;">
                            <span class="result-rank">{rank_emoji}</span>
                            <div class="result-name-group">
                                <p class="result-name">{r['candidate_name']}</p>
                                <div class="result-file">📄 {r['file_name']}</div>
                            </div>
                        </div>
                        <div class="match-badge {tier}">{score}%</div>
                    </div>
                    <div class="result-meta">
                        <span>📅 {exp_str} experience</span>
                        <span>🎯 Doc {r['full_doc_score']}% · Chunk {r['chunk_score']}%</span>
                    </div>
                    <div class="skills-row">
                        {skills_html if skills_html else '<span style="color:var(--text-muted);font-size:0.8rem;">No specific skills detected</span>'}
                    </div>
                    <div class="chunk-preview">
                        <strong>Best matching section</strong><br>
                        {chunk_preview}
                    </div>
                    {actions_html}
                </div>
                """, unsafe_allow_html=True)

                # Download button
                fname, fbytes = engine.get_resume_file_bytes(r["resume_id"])
                if fbytes:
                    mime = (
                        "application/pdf"
                        if fname.lower().endswith(".pdf")
                        else "application/octet-stream"
                    )
                    st.download_button(
                        label=f"📥 Download {fname}",
                        data=fbytes,
                        file_name=fname,
                        mime=mime,
                        key=f"dl_{r['resume_id']}_{idx}",
                    )

elif search_pressed and not query.strip():
    st.warning("Please enter a job description or search query.")


# ─── Footer ───────────────────────────────────────────────────
st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <div class="footer-brand">
        Powered by <span>SmartResume AI</span>
    </div>
    <div class="footer-tech">
        <span class="footer-tag">Sentence Transformers</span>
        <span class="footer-tag">all-mpnet-base-v2</span>
        <span class="footer-tag">spaCy NER</span>
        <span class="footer-tag">NLTK</span>
        <span class="footer-tag">Cosine Similarity</span>
        <span class="footer-tag">RAG Pipeline</span>
    </div>
</div>
""", unsafe_allow_html=True)
