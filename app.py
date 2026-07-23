import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import tensorflow as tf
import time

st.set_page_config(
    page_title="BananaNet — Ripeness Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,700;0,9..144,900;1,9..144,400&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=DM+Mono:wght@300;400;500&display=swap');

/* ══════════════════════════════════════════
   ROOT TOKENS
══════════════════════════════════════════ */
:root {
  --cream:     #FDFAF5;
  --cream2:    #F7F3EC;
  --white:     #FFFFFF;
  --ink:       #1A1612;
  --ink2:      #3D3730;
  --ink3:      #6B6358;
  --line:      rgba(26,22,18,0.08);
  --line2:     rgba(26,22,18,0.14);

  --lime:      #B5E550;
  --lime-dim:  rgba(181,229,80,0.15);
  --lime-glow: rgba(181,229,80,0.35);
  --amber:     #F5A623;
  --amber-dim: rgba(245,166,35,0.12);
  --coral:     #E85D3F;
  --coral-dim: rgba(232,93,63,0.1);
  --sky:       #2F7BE8;
  --sky-dim:   rgba(47,123,232,0.1);
  --emerald:   #1DAE72;

  --serif: 'Fraunces', Georgia, serif;
  --sans:  'DM Sans', sans-serif;
  --mono:  'DM Mono', monospace;

  --r-sm: 12px;
  --r-md: 18px;
  --r-lg: 26px;
  --r-xl: 36px;

  --shadow-sm: 0 2px 8px rgba(26,22,18,0.06);
  --shadow-md: 0 8px 32px rgba(26,22,18,0.08);
  --shadow-lg: 0 20px 60px rgba(26,22,18,0.10);
}

/* ══════════════════════════════════════════
   GLOBAL RESET & BASE
══════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
  font-family: var(--sans) !important;
  background-color: var(--cream) !important;
  color: var(--ink) !important;
}

/* Mesh gradient background */
.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 80% 60% at 10% 0%,   rgba(181,229,80,0.12)  0%, transparent 60%),
    radial-gradient(ellipse 60% 50% at 90% 10%,  rgba(47,123,232,0.08)  0%, transparent 55%),
    radial-gradient(ellipse 70% 70% at 50% 100%, rgba(245,166,35,0.06)  0%, transparent 60%),
    var(--cream);
  z-index: 0;
  pointer-events: none;
}

/* ══════════════════════════════════════════
   SCROLLBAR
══════════════════════════════════════════ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--cream2); }
::-webkit-scrollbar-thumb { background: var(--line2); border-radius: 3px; }

/* ══════════════════════════════════════════
   LAYOUT CONTAINERS
══════════════════════════════════════════ */
[data-testid="stMainBlockContainer"] {
  background: transparent !important;
  padding: 2.5rem 3rem !important;
  position: relative;
  z-index: 1;
}

/* ══════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════ */
[data-testid="stSidebar"] {
  background: transparent !important;
  border-right: none !important;
}
[data-testid="stSidebarContent"] {
  background: transparent !important;
  padding: 2rem 1rem !important;
}

/* Sidebar glass card wrapper */
.sb-brand {
  background: rgba(255,255,255,0.85);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid rgba(255,255,255,0.9);
  border-radius: var(--r-lg);
  padding: 22px 20px;
  margin-bottom: 18px;
  box-shadow: var(--shadow-md);
}
.sb-brand-icon {
  width: 44px; height: 44px;
  background: var(--ink);
  border-radius: var(--r-sm);
  display: flex; align-items: center; justify-content: center;
  font-size: 22px;
  margin-bottom: 12px;
  position: relative;
}
.sb-brand-icon::after {
  content: '';
  position: absolute;
  inset: -1px;
  border-radius: calc(var(--r-sm) + 1px);
  background: linear-gradient(135deg, var(--lime), var(--amber));
  z-index: -1;
}
.sb-brand-name {
  font-family: var(--calibri);
  font-size: 19px; font-weight: 700;
  color: var(--ink);
  letter-spacing: -0.5px;
  line-height: 1.1;
}
.sb-brand-sub {
  font-family: var(--mono);
  font-size: 10px; font-weight: 400;
  color: var(--emerald);
  letter-spacing: 1.5px;
  text-transform: uppercase;
  margin-top: 5px;
  display: flex; align-items: center; gap: 5px;
}
.sb-brand-sub::before {
  content: '';
  width: 6px; height: 6px;
  background: var(--emerald);
  border-radius: 50%;
  box-shadow: 0 0 6px var(--emerald);
  animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
  0%,100% { opacity: 1; }
  50%      { opacity: 0.4; }
}

/* Nav section */
.sb-nav-wrap {
  background: rgba(255,255,255,0.7);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255,255,255,0.85);
  border-radius: var(--r-lg);
  padding: 14px 10px;
  box-shadow: var(--shadow-sm);
}
.sb-nav-label {
  font-family: var(--mono);
  font-size: 9px; font-weight: 500;
  color: var(--ink3);
  text-transform: uppercase;
  letter-spacing: 2px;
  padding: 6px 10px 8px;
}

div[data-testid="stSidebar"] button {
  text-align: left !important;
  justify-content: flex-start !important;
  padding: 11px 14px !important;
  border-radius: var(--r-sm) !important;
  font-size: 13.5px !important;
  font-weight: 500 !important;
  font-family: var(--sans) !important;
  letter-spacing: 0.1px !important;
  margin-bottom: 2px !important;
  transition: all 0.2s ease !important;
  border: 1px solid transparent !important;
}
div[data-testid="stSidebar"] button[kind="secondary"] {
  background: transparent !important;
  color: var(--ink3) !important;
}
div[data-testid="stSidebar"] button[kind="secondary"]:hover {
  background: rgba(26,22,18,0.04) !important;
  color: var(--ink) !important;
}
div[data-testid="stSidebar"] button[kind="primary"] {
  background: var(--ink) !important;
  color: var(--lime) !important;
  font-weight: 700 !important;
  border-color: transparent !important;
  box-shadow: var(--shadow-sm) !important;
}

/* Sidebar bottom */
.sb-bottom {
  background: rgba(255,255,255,0.6);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(255,255,255,0.8);
  border-radius: var(--r-lg);
  padding: 18px 18px;
  margin-top: 16px;
  box-shadow: var(--shadow-sm);
}
.sb-bottom-title {
  font-family: var(--mono);
  font-size: 9px; font-weight: 500;
  color: var(--ink3);
  text-transform: uppercase;
  letter-spacing: 2px;
  margin-bottom: 12px;
}
.sb-bottom-row {
  font-family: var(--sans);
  font-size: 12.5px; font-weight: 500;
  color: var(--ink2);
  margin-bottom: 8px;
  display: flex; align-items: center; gap: 8px;
}
.sb-dot-lime   { color: var(--lime);    font-size: 10px; }
.sb-dot-sky    { color: var(--sky);     font-size: 10px; }
.sb-dot-amber  { color: var(--amber);   font-size: 10px; }

/* ══════════════════════════════════════════
   PAGE HEADER
══════════════════════════════════════════ */
.pg-eyebrow {
  font-family: var(--mono);
  font-size: 10px; font-weight: 500;
  color: var(--ink3);
  text-transform: uppercase;
  letter-spacing: 3px;
  margin-bottom: 10px;
  display: flex; align-items: center; gap: 10px;
}
.pg-eyebrow::after {
  content: '';
  flex: 0 0 28px;
  height: 1px;
  background: var(--ink3);
  opacity: 0.4;
}
.pg-title {
  font-family: var(--serif);
  font-size: 52px; font-weight: 900;
  color: var(--ink);
  line-height: 1.05;
  letter-spacing: -2px;
  margin-bottom: 14px;
}
.pg-title em {
  font-style: italic;
  font-weight: 400;
  color: var(--ink2);
}
.pg-sub {
  font-family: var(--sans);
  font-size: 15px; font-weight: 400;
  color: var(--ink3);
  line-height: 1.75;
  max-width: 560px;
  margin-bottom: 2.5rem;
}

/* ══════════════════════════════════════════
   GLASS CARD
══════════════════════════════════════════ */
.card {
  background: rgba(255,255,255,0.88) !important;
  backdrop-filter: blur(20px) !important;
  -webkit-backdrop-filter: blur(20px) !important;
  border: 1px solid rgba(255,255,255,0.95) !important;
  border-radius: var(--r-xl) !important;
  padding: 28px !important;
  margin-bottom: 20px !important;
  box-shadow: var(--shadow-md) !important;
  position: relative;
  overflow: hidden;
  transition: box-shadow 0.25s ease, transform 0.2s ease;
}
.card:hover {
  box-shadow: var(--shadow-lg) !important;
  transform: translateY(-2px);
}
.card-title {
  font-family: var(--mono);
  font-size: 9.5px; font-weight: 500;
  color: var(--ink3);
  text-transform: uppercase;
  letter-spacing: 2.5px;
  margin-bottom: 20px;
  padding-bottom: 14px;
  border-bottom: 1px solid var(--line);
  display: flex; align-items: center; gap: 8px;
}

/* ══════════════════════════════════════════
   METRIC CARDS
══════════════════════════════════════════ */
.metric-card {
  background: rgba(255,255,255,0.9);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255,255,255,0.95);
  border-radius: var(--r-xl);
  padding: 26px 28px;
  margin-bottom: 18px;
  box-shadow: var(--shadow-md);
  position: relative;
  overflow: hidden;
  transition: transform 0.2s ease, box-shadow 0.25s ease;
}
.metric-card:hover {
  transform: translateY(-3px);
  box-shadow: var(--shadow-lg);
}
.metric-orb {
  position: absolute;
  top: -30px; right: -30px;
  width: 100px; height: 100px;
  border-radius: 50%;
  filter: blur(40px);
  opacity: 0.5;
}
.mc-lime   { background: var(--lime); }
.mc-sky    { background: var(--sky); }
.mc-amber  { background: var(--amber); }
.metric-label {
  font-family: var(--mono);
  font-size: 9.5px; font-weight: 500;
  color: var(--ink3);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-bottom: 10px;
}
.metric-num {
  font-family: var(--serif);
  font-size: 38px; font-weight: 900;
  letter-spacing: -1.5px;
  line-height: 1;
  margin-bottom: 6px;
}
.mn-lime  { color: #3A8500; }
.mn-sky   { color: var(--sky); }
.mn-ink   { color: var(--ink); }
.metric-sub {
  font-family: var(--sans);
  font-size: 12px; font-weight: 400;
  color: var(--ink3);
}

/* ══════════════════════════════════════════
   STATUS BADGES
══════════════════════════════════════════ */
.badge {
  display: inline-flex; align-items: center; gap: 10px;
  padding: 11px 22px;
  border-radius: 100px;
  font-family: var(--sans);
  font-size: 17px; font-weight: 700;
  letter-spacing: -0.3px;
  margin: 10px 0 18px 0;
}
.badge-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
}
.badge-unripe  {
  background: rgba(29,174,114,0.10);
  color: #117A4A;
  border: 1.5px solid rgba(29,174,114,0.25);
}
.badge-unripe .badge-dot  { background: #1DAE72; box-shadow: 0 0 8px rgba(29,174,114,0.5); }
.badge-ripe    {
  background: rgba(181,229,80,0.12);
  color: #4A6B00;
  border: 1.5px solid rgba(181,229,80,0.35);
}
.badge-ripe .badge-dot    { background: #7AAE00; box-shadow: 0 0 8px rgba(181,229,80,0.6); }
.badge-overripe {
  background: rgba(232,93,63,0.09);
  color: #9A2E14;
  border: 1.5px solid rgba(232,93,63,0.25);
}
.badge-overripe .badge-dot { background: var(--coral); box-shadow: 0 0 8px rgba(232,93,63,0.4); }

/* ══════════════════════════════════════════
   META / INFO
══════════════════════════════════════════ */
.meta {
  font-family: var(--mono);
  font-size: 12px; font-weight: 400;
  color: var(--ink3);
  display: flex; align-items: center; gap: 12px;
  margin-bottom: 14px;
  flex-wrap: wrap;
}
.meta strong { color: var(--ink2); }
.meta-sep {
  width: 3px; height: 3px;
  background: var(--line2);
  border-radius: 50%;
}
.info-box {
  background: rgba(47,123,232,0.06);
  border: 1px solid rgba(47,123,232,0.18);
  border-left: 3px solid var(--sky);
  border-radius: 0 var(--r-sm) var(--r-sm) 0;
  padding: 13px 16px;
  font-family: var(--mono);
  font-size: 12px; color: var(--sky);
}
.img-wrap {
  background: var(--cream2);
  border: 1px solid var(--line);
  border-radius: var(--r-md);
  padding: 6px;
  margin-bottom: 18px;
  overflow: hidden;
}

/* ══════════════════════════════════════════
   PIPELINE STEPS
══════════════════════════════════════════ */
.pipe {
  display: flex; gap: 16px;
  margin-bottom: 22px; align-items: flex-start;
}
.pipe-num {
  width: 30px; height: 30px; flex-shrink: 0;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-family: var(--mono);
  font-size: 11px; font-weight: 500;
  margin-top: 1px;
}
.pn1 { background: var(--lime-dim);  color: #3A6A00; border: 1px solid rgba(181,229,80,0.4); }
.pn2 { background: var(--sky-dim);   color: var(--sky);   border: 1px solid rgba(47,123,232,0.3); }
.pn3 { background: var(--amber-dim); color: #9A6200;      border: 1px solid rgba(245,166,35,0.3); }
.pipe-body-title { font-family: var(--sans); font-size: 14px; font-weight: 700; color: var(--ink); margin-bottom: 3px; }
.pipe-body-sub   { font-family: var(--mono); font-size: 11.5px; color: var(--ink3); line-height: 1.55; }

/* Code chip */
.code-chip {
  display: inline-block;
  background: var(--cream2);
  border: 1px solid var(--line2);
  border-radius: var(--r-sm);
  padding: 10px 14px;
  font-family: var(--mono);
  font-size: 12px; color: var(--ink2);
  margin-top: 16px;
  line-height: 1.6;
}
.kw { color: var(--sky); }
.fn { color: var(--coral); }
.st { color: var(--emerald); }

/* ══════════════════════════════════════════
   BUTTONS
══════════════════════════════════════════ */
div.stButton > button:first-child {
  background: var(--ink) !important;
  color: var(--lime) !important;
  border-radius: var(--r-md) !important;
  padding: 12px 28px !important;
  font-weight: 700 !important;
  font-size: 13px !important;
  font-family: var(--sans) !important;
  border: none !important;
  width: 100% !important;
  margin-top: 12px !important;
  letter-spacing: 0.5px !important;
  text-transform: uppercase !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 6px 24px rgba(26,22,18,0.18) !important;
}
div.stButton > button:first-child:hover {
  background: var(--ink2) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 10px 32px rgba(26,22,18,0.22) !important;
}

/* ══════════════════════════════════════════
   PROGRESS BAR
══════════════════════════════════════════ */
div[data-testid="stProgress"] > div > div > div {
  background: linear-gradient(90deg, var(--lime), var(--emerald)) !important;
  border-radius: 4px !important;
}
div[data-testid="stProgress"] > div {
  background: var(--cream2) !important;
  border-radius: 4px !important;
}

/* ══════════════════════════════════════════
   FORM ELEMENTS & DROPDOWN CRITICAL FIXES (WARNA TERANG)
══════════════════════════════════════════ */
label[data-testid="stWidgetLabel"] p {
  color: var(--ink3) !important;
  font-weight: 500 !important;
  font-size: 11.5px !important;
  font-family: var(--mono) !important;
  text-transform: uppercase !important;
  letter-spacing: 1.2px !important;
}

/* 1. Paksa Kotak Utama Dropdown Berwarna Putih Terang */
div[data-baseweb="select"] > div {
  background-color: var(--white) !important;
  border-radius: var(--r-sm) !important;
  border: 1px solid var(--line2) !important;
  box-shadow: var(--shadow-sm) !important;
}

/* 2. Bersihkan Efek Hitam Pada Lapisan Dalam Kotak Dropdown */
div[data-baseweb="select"] div,
div[data-baseweb="select"] span,
div[data-testid="stSelectboxSelectedValue"] {
  background-color: transparent !important;
  color: var(--ink) !important; /* Teks pilihan berwarna gelap */
  font-family: var(--sans) !important;
  font-size: 14px !important;
}

/* 3. Warnai Ikon Panah Kecil Dropdown Agar Terlihat */
div[data-baseweb="select"] svg {
  fill: var(--ink2) !important;
}

/* Efek Focus Saat Dropdown Diklik */
div[data-baseweb="select"] > div:focus-within {
  border-color: rgba(181,229,80,0.6) !important;
  box-shadow: 0 0 0 3px rgba(181,229,80,0.15) !important;
}

/* 4. Lapisan Menu Melayang Saat Dropdown Terbuka (List Pilihan) */
div[role="listbox"], 
div[data-baseweb="popover"] {
  background-color: var(--white) !important;
  border: 1px solid var(--line2) !important;
  border-radius: var(--r-sm) !important;
}

/* Baris Opsi Pilihan di Dalam List */
div[role="listbox"] li,
div[data-baseweb="popover"] li,
li[role="option"] {
  font-family: var(--sans) !important;
  color: var(--ink) !important;
  font-size: 14px !important;
  background-color: var(--white) !important;
}

/* Efek Hover Saat Kursor Menyorot Pilihan */
div[role="listbox"] li:hover,
div[data-baseweb="popover"] li:hover,
li[role="option"]:hover {
  background-color: var(--cream2) !important;
  color: var(--ink) !important;
}

/* ══════════════════════════════════════════
   FILE UPLOADER
══════════════════════════════════════════ */
div[data-testid="stFileUploader"] section {
  background: rgba(255,255,255,0.7) !important;
  border: 2px dashed var(--line2) !important;
  border-radius: var(--r-md) !important;
  padding: 30px !important;
  transition: border-color 0.2s, background 0.2s !important;
}
div[data-testid="stFileUploader"] section:hover {
  border-color: rgba(181,229,80,0.5) !important;
  background: rgba(181,229,80,0.03) !important;
}
div[data-testid="stFileUploader"] section div {
  color: var(--ink3) !important;
  font-family: var(--sans) !important;
  font-size: 13px !important;
}
div[data-testid="stFileUploader"] section button {
  background: var(--white) !important;
  color: var(--ink) !important;
  border: 1px solid var(--line2) !important;
  border-radius: var(--r-sm) !important;
  font-weight: 600 !important;
  font-family: var(--sans) !important;
  font-size: 13px !important;
  box-shadow: var(--shadow-sm) !important;
}

/* ══════════════════════════════════════════
   DATAFRAME / TABLE
══════════════════════════════════════════ */
div[data-testid="stDataFrame"] {
  border-radius: var(--r-md) !important;
  overflow: hidden !important;
  border: 1px solid var(--line) !important;
  box-shadow: var(--shadow-sm) !important;
}
div[data-testid="stDataFrame"] * {
  font-family: var(--mono) !important;
  font-size: 12px !important;
}

table {
  font-family: var(--mono) !important;
  font-size: 13px !important;
  border-collapse: collapse !important;
  width: 100% !important;
  background: var(--white) !important;
  border-radius: var(--r-md);
  overflow: hidden;
  box-shadow: var(--shadow-sm);
}
thead th {
  background: var(--cream2) !important;
  color: var(--ink3) !important;
  font-size: 9.5px !important;
  text-transform: uppercase !important;
  letter-spacing: 1.5px !important;
  padding: 14px 20px !important;
  border-bottom: 1px solid var(--line) !important;
  font-weight: 500 !important;
}
tbody td {
  padding: 14px 20px !important;
  border-bottom: 1px solid var(--line) !important;
  color: var(--ink2) !important;
  font-weight: 400 !important;
}
tbody tr:last-child td { border-bottom: none !important; }
tbody tr:hover td {
  background: var(--cream2) !important;
}

/* ══════════════════════════════════════════
   FOOTER
══════════════════════════════════════════ */
.footer {
  display: flex; justify-content: space-between; align-items: center;
  border-top: 1px solid var(--line);
  padding-top: 24px;
  margin-top: 3.5rem;
  flex-wrap: wrap; gap: 12px;
}
.footer-item {
  font-family: var(--mono);
  font-size: 10px; font-weight: 400;
  color: var(--ink3);
  letter-spacing: 1.5px;
  text-transform: uppercase;
}

/* ══════════════════════════════════════════
   GRAPHVIZ (invert dark nodes to light)
══════════════════════════════════════════ */
.stGraphVizChart svg {
  background: var(--cream2) !important;
  border-radius: var(--r-md);
  padding: 12px;
}

/* ══════════════════════════════════════════
   MISC FIXES
══════════════════════════════════════════ */
p { font-family: var(--sans) !important; }
h1,h2,h3,h4 { font-family: var(--serif) !important; }

div[data-testid="stAlert"] {
  border-radius: var(--r-md) !important;
  font-family: var(--mono) !important;
  font-size: 12.5px !important;
}

li[role="option"] {
  font-family: var(--sans) !important;
  color: var(--ink) !important;
}
li[role="option"]:hover { background: var(--cream2) !important; }

/* Caption */
div[data-testid="stCaptionContainer"] p {
  font-family: var(--mono) !important;
  font-size: 11px !important;
  color: var(--ink3) !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# CORE LOGIC
# ══════════════════════════════════════════════
@st.cache_resource
def load_model_file(model_name):
    try:
        tf.keras.backend.clear_session()
        from tensorflow.keras.applications.resnet50 import preprocess_input

        def preprocess_resnet(x):
            return preprocess_input(x)

        if "ResNet50" in model_name:
            return tf.keras.models.load_model(
                'models/resnet50_banana.h5',
                custom_objects={'preprocess_resnet': preprocess_resnet},
                compile=False
            )
        else:
            return tf.keras.models.load_model('models/vgg16_banana.h5', compile=False)
    except Exception as e:
        st.error(f"Debug error: {e}")
        return None

def preprocess_image(image):
    img_array = np.array(image)
    if img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    elif len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img_array, (224, 224))
    return np.expand_dims(img_resized, axis=0).astype(np.float32)

if 'menu' not in st.session_state:
    st.session_state.menu = "Beranda"
if 'history' not in st.session_state:
    st.session_state.history = []

# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown('''
    <div class="sb-brand">
      <div class="sb-brand-name">Banana Deteksi</div>
    </div>
    ''', unsafe_allow_html=True)

    # ── INFO PENELITIAN ──
    st.markdown('''
    <div style="background:rgba(181,229,80,0.06);border:1px solid rgba(181,229,80,0.2);
                border-radius:12px;padding:12px 14px;margin-bottom:14px;">
      <div style="font-family:var(--mono);font-size:9px;color:var(--ink3);
                  text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;">
        Info Penelitian
      </div>
      <div style="font-family:var(--sans);font-size:11.5px;color:var(--ink2);line-height:1.6;">
        <b>Program Studi</b><br>Teknik Informatika<br>
        <span style="color:var(--ink3)">Universitas Dinamika Bangsa</span>
      </div>
      <div style="margin-top:8px;font-family:var(--sans);font-size:11px;color:var(--ink3);">
        Dzahabiyya R.K · M. Azwan S. · M. Ichsan A.<br>
        <span style="color:var(--ink3);font-family:var(--mono);font-size:10px;">2026</span>
      </div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="sb-nav-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="sb-nav-label">Navigasi Laporan</div>', unsafe_allow_html=True)

    pages = [
        ("Beranda",                ""),
        ("Upload & Prediksi AI",   ""),
     
    ]
    for pname, icon in pages:
        active = st.session_state.menu == pname
        label = f"▸  {pname}" if active else f"{icon}  {pname}"
        if st.button(label, key=f"nav_{pname}", use_container_width=True,
                     type="primary" if active else "secondary"):
            st.session_state.menu = pname
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('''
    <div class="sb-bottom">
      <div class="sb-bottom-title">Peneliti</div>
      <div class="sb-bottom-row"><span class="sb-dot-lime">◆</span> Dzahabiyya Rasikhah K.</div>
      <div class="sb-bottom-row"><span class="sb-dot-lime">◆</span> M. Azwan Salsi</div>
      <div class="sb-bottom-row"><span class="sb-dot-sky">◆</span> M. Ichsan Ali</div>
      <div style="margin-top:12px;border-top:1px solid var(--line);padding-top:10px;">
        <div class="sb-bottom-title">Model Tersedia</div>
        <div class="sb-bottom-row"><span class="sb-dot-amber">◆</span> VGG16 Transfer Learning</div>
        <div class="sb-bottom-row"><span class="sb-dot-amber">◆</span> ResNet50 Transfer Learning</div>
      </div>
    </div>
    ''', unsafe_allow_html=True)

current_page = st.session_state.menu

# ══════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════
def section_header(eyebrow, title, sub):
    st.markdown(f'<div class="pg-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="pg-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="pg-sub">{sub}</div>', unsafe_allow_html=True)

def card_open(title_html):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="card-title">{title_html}</div>', unsafe_allow_html=True)

def card_close():
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PAGE: BERANDA
# ══════════════════════════════════════════════
if current_page == "Beranda":
    section_header(
        "Proyek Penelitian 2026",
        "Sistem Klasifikasi<br><em>Kematangan Pisang.</em>",
        "Perancangan sistem klasifikasi tingkat kematangan buah pisang berdasarkan fitur warna dan tekstur menggunakan algoritma ResNet50 dan VGG16 berbasis web."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('''
        <div class="metric-card">
          <div class="metric-orb mc-lime"></div>
          <div class="metric-label">ResNet50 — Akurasi</div>
          <div class="metric-num mn-lime">54.69<span style="font-size:20px;font-weight:300;color:var(--ink3);">%</span></div>
          
        </div>''', unsafe_allow_html=True)
    with c2:
        st.markdown('''
        <div class="metric-card">
          <div class="metric-orb mc-sky"></div>
          <div class="metric-label">VGG16 — Akurasi</div>
          <div class="metric-num mn-sky">96.46<span style="font-size:20px;font-weight:300;color:var(--ink3);">%</span></div>
        </div>''', unsafe_allow_html=True)
    with c3:
        n = len(st.session_state.history)
        st.markdown(f'''
        <div class="metric-card">
          <div class="metric-orb mc-amber"></div>
          <div class="metric-label">Session Logs</div>
          <div class="metric-num mn-ink">{n}<span style="font-size:20px;font-weight:300;color:var(--ink3);"> img</span></div>
          <div class="metric-sub">Citra diproses saat ini</div>
        </div>''', unsafe_allow_html=True)

    c_l, c_r = st.columns([6, 5])
    with c_l:
        card_open("📁 Distribusi Dataset — 3 Kelas Kematangan")
        st.markdown('<p style="font-family:var(--sans);font-size:13.5px;color:var(--ink3);margin-bottom:16px;line-height:1.7;">1.500 citra pisang varietas lokal Jambi + Kaggle Banana Ripeness, terbagi 70% latih · 20% uji · 10% validasi.</p>', unsafe_allow_html=True)
        df_dist = pd.DataFrame([500, 500, 500], index=['Unripe (Mentah)', 'Ripe (Matang)', 'Overripe (Terlalu Matang)'], columns=['Jumlah Citra'])
        st.bar_chart(df_dist, color='#B5E550', use_container_width=True)
        card_close()

    with c_r:
        card_open("⚙️ Pipeline Sistem")
        st.markdown('''
        <div class="pipe"><div class="pipe-num pn1">01</div><div>
          <div class="pipe-body-title">Akuisisi Citra</div>
          <div class="pipe-body-sub">Upload gambar pisang — JPG/PNG, auto-resize 224×224px.</div>
        </div></div>
        <div class="pipe"><div class="pipe-num pn2">02</div><div>
          <div class="pipe-body-title">Ekstraksi Fitur CNN</div>
          <div class="pipe-body-sub">VGG16 atau ResNet50 + fitur warna RGB/HSV dan tekstur GLCM.</div>
        </div></div>
        <div class="pipe"><div class="pipe-num pn3">03</div><div>
          <div class="pipe-body-title">Klasifikasi Softmax</div>
          <div class="pipe-body-sub">Output: Unripe / Ripe / Overripe + Confidence Rate.</div>
        </div></div>
        ''', unsafe_allow_html=True)
        card_close()

# ══════════════════════════════════════════════
# PAGE: UPLOAD & PREDIKSI AI
# ══════════════════════════════════════════════
elif current_page == "Upload & Prediksi AI":
    section_header(
        "Inference Engine",
        "Upload &<br><em>Prediksi AI.</em>",
        "Pilih arsitektur model, unggah citra buah pisang, dan jalankan klasifikasi tingkat kematangan secara otomatis."
    )

    col1, col2 = st.columns([5, 6])
    with col1:
        card_open("01 — Konfigurasi Model")
        model_choice = st.selectbox("Arsitektur Neural Network", ["VGG16 (Stable)", "ResNet50 (Advanced)"])
        uploaded_file = st.file_uploader("Upload Citra Pisang", type=['jpg','jpeg','png'])
        btn_trigger = st.button("⚡  Jalankan Prediksi AI")
        card_close()

    with col2:
        card_open("02 — Hasil Klasifikasi AI")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            from datetime import datetime
            upload_time = datetime.now()
            upload_str  = upload_time.strftime("%d %B %Y, %H:%M:%S")

            st.markdown('<div class="img-wrap">', unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(f'''
            <div style="display:flex;align-items:center;gap:10px;
                        background:rgba(181,229,80,0.08);
                        border:1px solid rgba(181,229,80,0.28);
                        border-radius:10px;padding:10px 14px;margin-bottom:14px;">
              <span style="font-size:16px;">📅</span>
              <div>
                <div style="font-family:var(--mono);font-size:9px;color:var(--ink3);
                            text-transform:uppercase;letter-spacing:1.5px;margin-bottom:2px;">
                  Waktu Upload
                </div>
                <div style="font-family:var(--mono);font-size:12px;font-weight:600;color:#3A6A00;">
                  {upload_str}
                </div>
              </div>
            </div>
            ''', unsafe_allow_html=True)

            clean_model_name = "ResNet50" if "ResNet50" in model_choice else "VGG16"
            model_filename = f"models/{clean_model_name.lower()}_banana.h5"
            import os
            model_exists = os.path.exists(model_filename)

            if not model_exists:
                st.markdown(f'''
                <div style="background:#FFF0F0;border:1.5px solid #E85D3F;border-left:4px solid #E85D3F;
                            border-radius:12px;padding:16px 20px;margin-top:8px;">
                  <div style="font-family:var(--mono);font-size:11px;font-weight:600;color:#9A2E14;
                              text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">⚠ Model File Not Found</div>
                  <div style="font-family:var(--mono);font-size:12px;color:#6B2010;line-height:1.6;">
                    File <code style="background:#FFE0D8;padding:2px 6px;border-radius:4px;">{model_filename}</code> tidak ditemukan.<br><br>
                    Pastikan folder <code style="background:#FFE0D8;padding:2px 6px;border-radius:4px;">models/</code> ada di direktori yang sama dengan <code style="background:#FFE0D8;padding:2px 6px;border-radius:4px;">app.py</code>.
                  </div>
                  <pre style="background:#FFE0D8;padding:10px;border-radius:6px;margin-top:10px;font-size:11px;color:#6B2010;">project/
├── app.py
└── models/
    ├── vgg16_banana.h5
    └── resnet50_banana.h5</pre>
                </div>
                ''', unsafe_allow_html=True)
            else:
                model = load_model_file(clean_model_name)
                if model is None:
                    st.markdown(f'''
                    <div style="background:#FFF8E6;border:1.5px solid #F5A623;border-left:4px solid #F5A623;
                                border-radius:12px;padding:16px 20px;margin-top:8px;">
                      <div style="font-family:var(--mono);font-size:11px;font-weight:600;color:#7A4E00;
                                  text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">⚠ Gagal Memuat Model</div>
                      <div style="font-family:var(--mono);font-size:12px;color:#5A3A00;line-height:1.6;">
                        File ditemukan namun gagal di-load. Kemungkinan file corrupt atau versi TensorFlow tidak kompatibel.
                      </div>
                    </div>
                    ''', unsafe_allow_html=True)
                elif btn_trigger:
                    with st.spinner('Menganalisis citra...'):
                        try:
                            t0 = time.time()
                            input_tensor = preprocess_image(image)
                            prediction = model.predict(input_tensor)
                            predicted_index = np.argmax(prediction)
                            confidence = np.max(prediction) * 100
                            inference_time = int((time.time() - t0) * 1000)
                            classes = ['Overripe','Ripe','Unripe']
                            predicted_class = classes[predicted_index]

                            st.markdown("<p style='font-family:var(--mono);font-size:9.5px;color:var(--ink3);text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;'>Classification Result</p>", unsafe_allow_html=True)
                            cls_map = {
                                'Unripe':   ('badge-unripe',   'Mentah'),
                                'Ripe':     ('badge-ripe',     'Matang'),
                                'Overripe': ('badge-overripe', 'Terlalu Matang'),
                            }
                            cls, label_id = cls_map[predicted_class]
                            st.markdown(f"<div class='badge {cls}'><span class='badge-dot'></span>{predicted_class} — {label_id}</div>", unsafe_allow_html=True)
                            st.markdown(f'''
                            <div class="meta">
                              <strong>Confidence: {confidence:.2f}%</strong>
                              <span class="meta-sep"></span>{clean_model_name}
                              <span class="meta-sep"></span>{inference_time} ms
                            </div>''', unsafe_allow_html=True)
                            st.progress(int(confidence))

                            st.session_state.history.append({
                                "Tanggal": upload_str,
                                "File": uploaded_file.name,
                                "Model": clean_model_name,
                                "Prediksi": predicted_class,
                                "Confidence": f"{confidence:.2f}%",
                                "Speed": f"{inference_time}ms",
                                "Tensor": f"{input_tensor.shape[1]}\u00d7{input_tensor.shape[2]}"
                            })
                        except Exception as e:
                            st.markdown(f'''
                            <div style="background:#FFF0F0;border:1.5px solid #E85D3F;border-left:4px solid #E85D3F;
                                        border-radius:12px;padding:16px 20px;margin-top:8px;">
                              <div style="font-family:var(--mono);font-size:11px;font-weight:600;color:#9A2E14;
                                          text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">⚠ Prediction Error</div>
                              <div style="font-family:var(--mono);font-size:12px;color:#6B2010;">{str(e)}</div>
                            </div>
                            ''', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">▸ Upload citra pisang di panel kiri untuk memulai prediksi AI.</div>', unsafe_allow_html=True)
        card_close()

    if st.session_state.history:
        card_open("📊 Log Riwayat Prediksi — Sesi Aktif")
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True)
        card_close()



# ══════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════
st.markdown('''
<div class="footer">
  <span class="footer-item">Dzahabiyya Rasikhah K.</span>
  <span class="footer-item">M. Azwan Salsi</span>
  <span class="footer-item">M. Ichsan Ali</span>
  <span class="footer-item">Teknik Informatika — UDBD</span>
  <span class="footer-item">2026</span>
</div>
''', unsafe_allow_html=True)