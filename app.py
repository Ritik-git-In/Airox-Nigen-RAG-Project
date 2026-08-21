"""Streamlit UI for the PDF RAG assistant.

Flow: log in with an email -> upload PDFs (indexed into your private vector
store) -> ask questions and get answers with citations.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os

# Chroma's anonymized telemetry adds startup/runtime overhead we don't need.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import threading
import time

import streamlit as st
import streamlit.components.v1 as components

# Only the lightweight modules are imported up front. The heavy ones
# (vectorstore -> chromadb, pipeline -> openai) are imported lazily inside the
# functions that need them, so a cold start renders the login page fast.
from rag import auth, config, security

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Look & feel                                                                 #
# --------------------------------------------------------------------------- #
def inject_styles() -> None:
    """Custom CSS: gradient backdrop, chat bubbles, animated buttons."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        /* Hide default Streamlit chrome: menu, footer, Deploy button. The
           native "Running..." status widget is left alone (see below) — it's
           the searching/thinking indicator, not decorative chrome. */
        #MainMenu, footer,
        [data-testid="stDeployButton"],
        [data-testid="stToolbarActions"] { display: none !important; }
        /* Visual header hiding and click-through. */
        header.stAppHeader,
        .stAppToolbar,
        [data-testid="stHeader"] {
            background-color: transparent !important;
            background-image: none !important;
            border: none !important;
            box-shadow: none !important;
            pointer-events: none !important;
        }
        [data-testid="stHeader"] > div, [data-testid="stHeader"] * {
            width: 100% !important;
            pointer-events: none !important;
        }
        /* Sidebar open/close uses Streamlit's own native control — a custom
           JS-driven replacement (a hand-built button + hand-built open/close
           CSS state) was tried and repeatedly broke (overlap bugs, then
           stopped opening at all). Streamlit's own button is guaranteed to
           work; it just needs pointer-events restored (it lives inside
           stHeader, which the rule above disables clicks on) and its own
           size back — the "stHeader *" rule above stretches it to the
           header's full width by default. Sized and positioned here to sit
           just left of the navbar title, not as a full-width bar. */
        [data-testid="stExpandSidebarButton"] {
            pointer-events: auto !important;
            position: fixed !important;
            top: 14px !important;
            left: 14px !important;
            width: 40px !important;
            min-width: 40px !important;
            max-width: 40px !important;
            height: 40px !important;
            margin: 0 !important;
            padding: 0 !important;
            border-radius: 12px !important;
            background: rgba(255,255,255,0.10) !important;
            border: 1px solid rgba(255,255,255,0.16) !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            z-index: 99999 !important; /* above the navbar (99998) */
            box-shadow: 0 10px 30px rgba(0,0,0,0.35) !important;
            transition: background .15s ease !important;
        }
        [data-testid="stExpandSidebarButton"]:hover {
            background: rgba(255,255,255,0.18) !important;
        }
        [data-testid="stExpandSidebarButton"] svg,
        [data-testid="stExpandSidebarButton"] span {
            width: auto !important;
        }
        /* Keep the native "Running..." indicator above the custom navbar
           (z-index 99998, further below) so it isn't hidden behind that
           solid-background bar. Only just above the navbar, not up near the
           sidebar toggle button's range (999999) — giving it that much
           priority risked it intercepting clicks meant for the button. */
        [data-testid="stStatusWidget"] {
            z-index: 99999 !important;
            /* Purely informational, never something a user clicks — make
               sure it can never sit on top of and intercept clicks meant
               for the sidebar toggle button or anything else. */
            pointer-events: none !important;
        }

        /* Tighter dark backdrop applied to all main containers so footers/panels match */
        html, body,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        .block-container {
            background: #050505 !important; /* user requested tighter black */
            background-image: none !important;
            min-height: 100vh;
        }

        /* DevTools-identified container (runtime-generated emotion class).
           Keep a specific override in case Streamlit inserts its own darker band. */
        .st-emotion-cache-1p8uksh.e15ve43o3,
        .stBottom.st-emotion-cache-1p2n2i4.e15ve43o2 {
            background-color: #050505 !important;
            background-image: none !important;
        }

        /* Pull everything up: Streamlit's default top padding is very large */
        [data-testid="stMainBlockContainer"], .block-container {
            padding-top: 2.4rem !important;
            /* Reserve space so the last message — even a long one, like a
               multi-item reference list — can scroll fully clear above the
               fixed question box instead of ending up hidden behind it.
               The chat input can grow to several lines as the user types a
               long question, and since it's pinned to the viewport that
               growth eats upward into the page instead of pushing content
               down on its own. This used to track the input's *live*
               height via a ResizeObserver running in a components.html()
               iframe (see git history) so the reservation stayed exact —
               but that iframe re-ran on every single Streamlit rerun (any
               click anywhere on the page, not just typing), adding real,
               noticeable overhead to every interaction. The textarea is
               already CSS-capped to max-height 12rem below, so its total
               height (including the send button, borders, padding) has a
               known worst case — a flat reservation sized for that worst
               case is exact enough without live JS tracking. */
            padding-bottom: 18rem !important;
            /* Streamlit's own default styling makes this element height:100%
               of its scrolling flex parent (stAppScrollToBottomContainer,
               bounded to 100vh above) — fine normally, since content that's
               shorter than the viewport just leaves blank space below it,
               but it silently caps how tall this box can grow past 100vh.
               With that cap in place the padding-bottom reserved above
               stays *inside* that fixed 100vh box instead of extending the
               scrollable area past it, so once real content plus the
               reservation together exceed one viewport's height, scrolling
               to this element's true end no longer clears the last message
               past the fixed input — confirmed live via devtools (the
               rendered box stayed pinned at 100vh regardless of how much
               taller its own content grew). Overriding height:auto alone
               isn't enough either: as a flex item inside a fixed-height
               column, flex-shrink defaults to 1, so the browser still
               shrinks it back down to fit the 800px parent regardless —
               also confirmed live, the computed height stayed exactly
               800px with height:auto alone. flex-shrink:0 stops that, so
               it truly sizes to its real content + padding and the
               scrollable parent's scrollHeight correctly grows to match —
               "scroll to the end" then actually means the real end.
               min-height keeps it filling at least one viewport when
               there's little content, same as before. */
            height: auto !important;
            min-height: 100% !important;
            flex-shrink: 0 !important;
        }

        /* Ensure header children stretch full width and don't get clipped */
        [data-testid="stHeader"] > div, [data-testid="stHeader"] * {
            width: 100% !important;
        }

        /* Bound the main pane to its own fixed-size, independently-scrolling
           region instead of letting the whole document grow and scroll.
           Previously the page itself scrolled, and every scroll-to-latest-
           message call had to reflow the *entire* page — including
           recalculating every fixed-position element against an ever-
           growing document height — which got measurably slower, and less
           reliable, the longer a chat session ran. Scrolling a viewport-
           sized, bounded element is a cheap, isolated operation regardless
           of how much history it holds.
           Streamlit renders this pane under TWO different data-testids
           depending on the page: plain "stMain" normally (e.g. the login
           page), but "stAppScrollToBottomContainer" on any page that uses
           st.chat_input — confirmed live via devtools, both carry the same
           "stMain" CSS class either way. The chat page is the one that
           actually needs this rule, so both selectors must be listed or it
           silently never applies where it matters.
           The navbar and chat input stay position:fixed (relative to the
           viewport, unaffected by an ancestor's overflow) exactly as before. */
        [data-testid="stMain"],
        [data-testid="stAppScrollToBottomContainer"] {
            height: 100vh !important;
            overflow-y: auto !important;
            overflow-x: hidden !important;
        }

        /* The application navbar is separate from Streamlit's header. Keep
           it fixed while the conversation and document views scroll. */
        .st-key-app-navbar {
            position: fixed !important;
            top: 0;
            left: 0;
            right: 0;
            z-index: 99998;
            width: 100%;
            margin: 0 !important;
            /* Left padding cleared for the sidebar-toggle button (fixed at
               left:14px, 40px wide) so the title starts to its right
               instead of sitting underneath it. */
            padding: 10px clamp(1.25rem, 5.2vw, 6.25rem) 8px 4.5rem !important;
            background: #050505;
            box-shadow: 0 10px 24px rgba(0,0,0,0.38);
        }
        /* A fixed element does not occupy document space. This prevents the
           first chat message from being hidden beneath the navbar. */
        .app-navbar-spacer { height: 76px; }

        @media (max-width: 640px) {
            .st-key-app-navbar {
                padding-right: 1rem !important;
            }
            .app-navbar-spacer { height: 82px; }
        }

        /* Header band */
        .app-header { padding: 0; }
        .app-title {
            font-size: 30px; font-weight: 700; line-height: 1.1;
            background: linear-gradient(90deg, #8b9bff 0%, #c08bff 60%, #ff9ecb 100%);
            -webkit-background-clip: text; background-clip: text; color: transparent;
        }
        .app-sub { color: #9aa3b8; font-size: 14px; margin-top: 4px; }
        /* Page heading under the navbar — smaller than the login hero */
        .app-header .app-title { font-size: 18px; }
        .app-header .app-sub { font-size: 12px; margin-top: 1px; }

        /* Chat — ChatGPT-style clean rows (no boxy cards) */
        [data-testid="stChatMessage"] {
            background: transparent;
            border: none;
            box-shadow: none;
            padding: 2px 0;
            margin-bottom: 8px;
        }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
            line-height: 1.65;
        }
        /* User turn: compact bubble that hugs its text, pushed to the right */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            width: fit-content;
            max-width: 80%;
            margin: 6px 0 16px auto;       /* left margin auto -> aligns right */
            flex-direction: row-reverse;    /* avatar sits on the right */
            background: rgba(124,108,246,0.16);
            border: 1px solid rgba(124,108,246,0.30);
            border-radius: 16px;
            padding: 4px 12px;
        }
        /* Assistant turn: open, full-width, no box */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
            background: transparent;
            border: none;
            padding: 2px 2px;
            margin-bottom: 18px;
        }
        /* Avatars: soft circular badges (recolour the default red user avatar) */
        [data-testid="stChatMessageAvatarUser"],
        [data-testid="stChatMessageAvatarAssistant"] {
            border-radius: 50%;
        }
        [data-testid="stChatMessageAvatarAssistant"] {
            background: linear-gradient(135deg, #7C6CF6, #c08bff) !important;
            color: #fff !important;
        }
        [data-testid="stChatMessageAvatarUser"] {
            background: linear-gradient(135deg, #2bb5a0, #3ad07a) !important;
            color: #fff !important;
        }

        /* Buttons */
        .stButton > button, .stFormSubmitButton > button {
            border-radius: 11px; border: none; font-weight: 600;
            background: linear-gradient(135deg, #6C5CE7, #8b5cf6);
            color: #fff;
            transition: transform .14s ease, box-shadow .14s ease, filter .14s ease;
        }
        .stButton > button:hover, .stFormSubmitButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 22px rgba(124,108,246,.42);
            filter: brightness(1.05);
        }

        /* Chat input area. Streamlit natively makes [data-testid="stBottom"]
           (the input's wrapper) position:sticky within the same scrolling
           flex column as the messages — confirmed live via devtools, and it
           IS an elegant idea in principle. In practice, tested live with a
           long conversation, it does not actually track the container's
           true bottom edge: it visibly sticks partway up the viewport,
           overlapping message content instead of the fixed input. That's a
           worse version of the exact bug being fixed here, so it isn't
           used. position:fixed (relative to the viewport, immune to any of
           the scroll container's own internal quirks) is the version that
           has tested reliably, both now and earlier this session — so
           stBottom's own box is collapsed to hide the empty space it would
           otherwise leave in normal flow, and the actual input is pulled
           out via fixed positioning. See the padding-bottom comment on
           stMainBlockContainer below for how clearance above it is
           reserved now that the message column's own height is bounded. */
        [data-testid="stBottom"],
        [data-testid="stBottomBlockContainer"] {
            padding: 0 !important;
            margin: 0 !important;
            min-height: 0 !important;
            height: 0 !important;
            overflow: visible !important;
        }
        [data-testid="stChatInput"] {
            position: fixed !important;
            left: clamp(1.25rem, 5.2vw, 6.25rem) !important;
            right: clamp(1.25rem, 5.2vw, 6.25rem) !important;
            bottom: 0 !important;
            border-radius: 14px 14px 0 0;
            /* Lower than expanded source panels (99999) so those can render
               above this input instead of being covered by it. */
            z-index: 99000 !important;
            border: 1px solid rgba(124,108,246,.45);
            border-bottom: none;
            background: #141a28;
            box-shadow: 0 10px 36px rgba(0,0,0,0.45);
            margin-top: 0 !important;
        }
        /* Cap how tall the input can grow (about 8 lines) — pasting a long
           block of text (a whole previous answer, a paragraph to ask about)
           would otherwise let it grow without limit, eventually consuming
           most of the viewport and leaving no room to see the conversation
           above it. Past this height the textarea scrolls internally
           instead, the same way ChatGPT/Claude cap their input box. */
        [data-testid="stChatInput"] textarea {
            max-height: 12rem !important;
            overflow-y: auto !important;
        }

        /* Keep the fixed input within the main pane while the sidebar is open. */
        body:has([data-testid="stSidebar"][aria-expanded="true"])
        [data-testid="stChatInput"] {
            left: calc(300px + clamp(1.25rem, 5.2vw, 6.25rem)) !important;
        }

        @media (max-width: 640px) {
            [data-testid="stChatInput"] {
                left: 1rem !important;
                right: 1rem !important;
            }
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: #000000;
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #cdd3e6; }

        /* Sources expander */
        [data-testid="stExpander"] {
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.07);
            background: rgba(255,255,255,0.02);
        }
        /* Match any element whose class contains the source-details key so
           per-message keys like "source-details-3" are picked up. No
           z-index here: this is normal scrolling page content and must
           stay BELOW the fixed navbar/chat input, not above them — giving
           it a higher z-index than the fixed chrome (as before) let its
           text render on top of the navbar/input when scrolled behind
           them, which looked like the fixed bars had gone transparent. */
        [class*="st-key-source-details"] {
            scroll-margin-bottom: 7rem;
        }

        /* When an expander is focused/opened, give the browser a scroll margin
           so it doesn't place the bottom of the expander underneath the
           fixed chat input. */
        [data-testid="stExpander"] > details,
        [data-testid="stExpander"] > details > summary {
            scroll-margin-bottom: 8rem;
        }

        /* Document search box: drop the "Press Enter to apply" hint
           Streamlit shows under every text_input — it's redundant clutter
           for a simple filter box. */
        .st-key-doc_search_wrap [data-testid="InputInstructions"] {
            display: none !important;
        }
        /* Google-style search box: the input and its clear button are two
           separate Streamlit elements side by side — remove the gap between
           their columns and round only the outer corners of each so they
           read as one continuous pill instead of two boxes. */
        .st-key-doc_search_wrap [data-testid="stHorizontalBlock"] {
            gap: 0 !important;
            align-items: center !important;
        }
        .st-key-doc_search_wrap [data-testid="stTextInputRootElement"] {
            border-radius: 999px 0 0 999px !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            border-right: none !important;
            background: rgba(255,255,255,0.04) !important;
            box-shadow: none !important;
        }
        .st-key-doc_search_wrap [data-testid="stTextInputRootElement"]:focus-within {
            border-color: rgba(255,255,255,0.30) !important;
        }
        .st-key-doc_search_wrap [data-testid="stTextInputIcon"] {
            color: #8b93a7 !important;
        }
        .st-key-doc_search_wrap [data-testid="stTextInput"] input {
            background: transparent !important;
        }
        .st-key-doc_search_wrap [data-testid="stBaseButton-secondary"] {
            border-radius: 0 999px 999px 0 !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            border-left: 1px solid rgba(255,255,255,0.10) !important;
            background: rgba(255,255,255,0.04) !important;
            color: #8b93a7 !important;
            height: 2.5rem !important;
            box-shadow: none !important;
        }
        .st-key-doc_search_wrap [data-testid="stBaseButton-secondary"]:hover {
            background: rgba(255,255,255,0.09) !important;
            color: #e8eaf2 !important;
            transform: none !important;
            box-shadow: none !important;
        }

        /* File uploader dropzone */
        [data-testid="stFileUploaderDropzone"] {
            border-radius: 12px;
            border: 1px dashed rgba(124,108,246,0.4);
            background: rgba(124,108,246,0.04);
        }

        /* Profile card (sidebar) */
        .profile-card {
            display: flex; align-items: center; gap: 12px;
            padding: 14px; border-radius: 16px; margin-bottom: 14px;
            background: linear-gradient(135deg, rgba(124,108,246,0.20), rgba(192,139,255,0.06));
            border: 1px solid rgba(124,108,246,0.32);
            box-shadow: 0 8px 26px rgba(0,0,0,0.30);
            animation: fadeInUp 0.5s ease both;
        }
        .avatar {
            width: 48px; height: 48px; flex: 0 0 48px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-weight: 700; font-size: 21px; color: #fff;
            background: linear-gradient(135deg, #7C6CF6, #c08bff);
            box-shadow: 0 4px 16px rgba(124,108,246,0.55);
        }
        /* User chip in the top header (right side) */
        .user-chip {
            display: flex; align-items: center; justify-content: flex-end;
            gap: 10px; margin-top: 10px;
        }
        .user-email {
            font-weight: 600; font-size: 13px; color: #e8eaf2;
            word-break: break-all; text-align: right;
        }
        .avatar-sm { width: 44px; height: 44px; flex: 0 0 44px; font-size: 18px; }

        /* Top navbar (DigiRocket-style: brand left, user menu right) */
        .nav-brand {
            font-size: 20px; font-weight: 700; color: #e8eaf2;
            display: flex; align-items: center; gap: 8px; padding-top: 2px;
        }
        .nav-brand .nav-accent {
            background: linear-gradient(90deg, #8b9bff, #c08bff);
            -webkit-background-clip: text; background-clip: text; color: transparent;
        }
        .nav-divider {
            border: none; height: 1px; margin: 4px 0 8px 0;
            background: linear-gradient(90deg, rgba(124,108,246,.40), rgba(255,255,255,.04));
        }
        /* Compact user pill = the popover trigger button */
        [data-testid="stPopover"] button {
            border-radius: 999px !important;
            background: rgba(124,108,246,0.14) !important;
            border: 1px solid rgba(124,108,246,0.32) !important;
            color: #e8eaf2 !important; font-weight: 600 !important;
            padding: 7px 16px !important;
            transition: background .15s ease;
        }
        [data-testid="stPopover"] button:hover {
            background: rgba(124,108,246,0.26) !important;
        }

        .profile-hi { font-size: 12px; color: #9aa3b8; }
        .profile-email {
            font-weight: 600; font-size: 13.5px; color: #e8eaf2; word-break: break-all;
        }
        .profile-status {
            font-size: 11px; color: #7ee0a8; margin-top: 3px;
            display: flex; align-items: center; gap: 5px;
        }
        .profile-status .dot {
            width: 7px; height: 7px; border-radius: 50%; background: #3ad07a;
            box-shadow: 0 0 8px #3ad07a; animation: pulse 1.6s ease-in-out infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50%      { opacity: .45; transform: scale(.8); }
        }

        /* Login card */
        .login-card {
            max-width: 560px; margin: 3vh auto 34px auto;
            padding: 30px 30px;
            background: rgba(255,255,255,0.035);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            box-shadow: 0 18px 60px rgba(0,0,0,0.45);
            animation: fadeInUp 0.5s ease both;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# State                                                                        #
# --------------------------------------------------------------------------- #
def _init_state() -> None:
    st.session_state.setdefault("user_email", None)
    st.session_state.setdefault("messages", [])  # list[dict(role, content, sources)]
    st.session_state.setdefault("upload_mode", None)  # "pdf" | "folder"
    # Bumped after each successful index so the file_uploader widgets below
    # get a fresh key and drop their previous selection instead of Streamlit
    # re-showing the just-indexed files (uploaders remember files by key).
    st.session_state.setdefault("uploader_nonce", 0)


@st.cache_resource(show_spinner=False)
def _start_backend_warmup() -> bool:
    """Pre-load the heavy backend once per server process, off the main thread.

    chromadb and the ONNX embedding model take several seconds to import,
    initialize, and (first run only) download. Doing it in a background thread
    while the user is still on the login page means their first upload or
    question doesn't pay that cost.
    """

    def _warm() -> None:
        try:
            from rag import embeddings, vectorstore

            vectorstore.warm_up()
            embeddings.warm_up()
        except Exception:
            pass  # warm-up is best-effort; real calls will surface any error

    threading.Thread(target=_warm, name="rag-warmup", daemon=True).start()
    return True


@st.cache_data(show_spinner=False)
def _sources_for(user_email: str) -> list[str]:
    """Cached list of a user's indexed documents.

    Reads the lightweight per-user catalog (which migrates itself from the
    vector store on first use) instead of scanning every chunk's metadata in
    Chroma — essential once drive scans index thousands of PDFs. The cache is
    cleared whenever documents are added or removed.
    """
    from rag import catalog

    return catalog.sources(user_email)


# --------------------------------------------------------------------------- #
# Views                                                                        #
# --------------------------------------------------------------------------- #
def login_view() -> None:
    """Account-based login: each user has their own email + password.

    Every account's documents are stored in a separate, private vector
    collection, so users only ever see their own PDFs.
    """
    st.markdown(
        """
        <div class="login-card">
          <div class="app-title">📄 PDF RAG Assistant</div>
          <div class="app-sub">
            Upload many PDFs and ask questions across them — every answer cites
            the exact file and page it came from. Your documents stay private to
            your account.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns([1, 2, 1])
    with cols[1]:
        sign_in_tab, sign_up_tab = st.tabs(["Sign in", "Create account"])

        with sign_in_tab:
            with st.form("signin"):
                email = st.text_input(
                    "Email", placeholder="you@example.com", autocomplete="email"
                )
                pw = st.text_input(
                    "Password", type="password", autocomplete="current-password"
                )
                ok = st.form_submit_button("Sign in →", use_container_width=True)
            if ok:
                if not security.is_valid_email(email):
                    st.error("Please enter a valid email address.")
                elif auth.verify_user(email, pw):
                    st.session_state.user_email = email.strip().lower()
                    st.session_state.messages = []
                    st.rerun()
                else:
                    st.error("Incorrect email or password.")

        with sign_up_tab:
            with st.form("signup"):
                email2 = st.text_input(
                    "Email",
                    placeholder="you@example.com",
                    key="su_email",
                    autocomplete="email",
                )
                pw1 = st.text_input(
                    f"Password (min {auth.MIN_PASSWORD_LEN} characters)",
                    type="password",
                    key="su_pw1",
                    autocomplete="new-password",
                )
                pw2 = st.text_input(
                    "Confirm password",
                    type="password",
                    key="su_pw2",
                    autocomplete="new-password",
                )
                ok2 = st.form_submit_button("Create account", use_container_width=True)
            if ok2:
                if pw1 != pw2:
                    st.error("The two passwords don't match.")
                else:
                    created, msg = auth.create_user(email2, pw1)
                    if created:
                        st.session_state.user_email = email2.strip().lower()
                        st.session_state.messages = []
                        st.success("Account created! Signing you in…")
                        st.rerun()
                    else:
                        st.error(msg)


def sidebar(user_email: str) -> None:
    """Directory upload area + the user's indexed documents."""
    with st.sidebar:
        st.subheader("Add documents")
        # Streamlit's native uploader cannot change between a file and a
        # directory picker after its plus button has been clicked. Put that
        # decision in a small plus menu first, then open the right picker.
        with st.popover("＋ Add", use_container_width=True):
            st.caption("Choose what you want to add")
            if st.button("Upload PDF", use_container_width=True, key="choose_pdf_upload"):
                st.session_state.upload_mode = "pdf"
                st.session_state.close_add_menu = True
                st.rerun()
            if st.button("Upload Folder", use_container_width=True, key="choose_folder_upload"):
                st.session_state.upload_mode = "folder"
                st.session_state.close_add_menu = True
                st.rerun()
            if st.button("Scan Drive", use_container_width=True, key="choose_drive_scan"):
                st.session_state.upload_mode = "drive"
                st.session_state.close_add_menu = True
                st.rerun()

        # ``st.popover`` has no API to close itself, so after an option is
        # chosen we nudge the (same-origin) parent page to dismiss it. The
        # embedded counter makes the HTML unique per click — without it,
        # Streamlit reuses the previous identical iframe and the script never
        # runs again (which is why only the first selection used to close).
        if st.session_state.pop("close_add_menu", False):
            st.session_state.add_menu_close_count = (
                st.session_state.get("add_menu_close_count", 0) + 1
            )
            components.html(
                f"""
                <script>
                // close request #{st.session_state.add_menu_close_count}
                setTimeout(function () {{
                    const doc = window.parent.document;
                    if (doc.querySelector('[data-baseweb="popover"]')) {{
                        const trigger = doc.querySelector(
                            '[data-testid="stSidebar"] [data-testid="stPopover"] button'
                        );
                        if (trigger) trigger.click();
                    }}
                    doc.dispatchEvent(new KeyboardEvent("keydown", {{key: "Escape", bubbles: true}}));
                    doc.dispatchEvent(new KeyboardEvent("keyup", {{key: "Escape", bubbles: true}}));
                }}, 100);
                </script>
                """,
                height=0,
            )

        nonce = st.session_state.uploader_nonce
        uploads = []
        if st.session_state.upload_mode == "pdf":
            st.caption("Upload one PDF")
            upload = st.file_uploader(
                "Choose a PDF", type=["pdf"], key=f"single_pdf_upload_{nonce}"
            )
            uploads = [upload] if upload else []
        elif st.session_state.upload_mode == "folder":
            st.caption("Upload a folder — PDFs in subfolders are included")
            uploads = st.file_uploader(
                "Choose a folder containing PDFs",
                type=["pdf"],
                accept_multiple_files="directory",
                key=f"folder_pdf_upload_{nonce}",
                help="All PDFs in the selected folder and its subfolders are included.",
            )
        elif st.session_state.upload_mode == "drive":
            st.caption(
                "Index every PDF on a connected drive — files are read in "
                "place, nothing is copied or uploaded."
            )
            _drive_scan_picker(user_email)
        else:
            st.caption("Use + Add to upload a PDF, a folder, or scan a whole drive.")

        if uploads and st.button(
            "Index selected PDFs", type="primary", use_container_width=True
        ):
            _index_uploads(user_email, uploads)

        st.divider()
        sources = _sources_for(user_email)
        st.subheader(f"Your documents ({len(sources):,})" if sources else "Your documents")
        if not sources:
            st.caption("No documents indexed yet.")
        else:
            # A drive scan can index thousands of PDFs; scanning the list
            # visually for one file name isn't practical at that scale.
            with st.container(key="doc_search_wrap"):
                search_col, clear_col = st.columns([0.85, 0.15])
                with search_col:
                    search = st.text_input(
                        "Search your documents",
                        placeholder="Search by file name…",
                        icon="🔍",
                        key="doc_search",
                        label_visibility="collapsed",
                    )
                with clear_col:
                    # A text_input only picks up an edit on blur/Enter, so
                    # backspacing it clear and clicking away can feel
                    # sluggish. A button click reruns immediately — a fast,
                    # one-click way back to the full list instead of waiting
                    # on that.
                    st.button(
                        "✕",
                        key="clear_doc_search_btn",
                        help="Clear search",
                        disabled=not search,
                        on_click=lambda: st.session_state.update(doc_search=""),
                    )
            query = search.strip().lower()
            visible = [s for s in sources if query in s.lower()] if query else sources
            if query and not visible:
                st.caption(f"No documents match “{search.strip()}”.")
            # Drive scans can index thousands of PDFs; rendering a row (plus
            # a delete button) for each would make every rerun crawl. Cap
            # the list by default, but let the user opt into the full list
            # for the *current* search — reset that choice whenever the
            # search changes so a cleared search doesn't suddenly render
            # thousands of rows.
            if st.session_state.get("doc_search_last") != query:
                st.session_state.doc_list_show_all = False
                st.session_state.doc_search_last = query
            _MAX_LISTED = 30
            limit = len(visible) if st.session_state.get("doc_list_show_all") else _MAX_LISTED
            for src in visible[:limit]:
                cols = st.columns([0.82, 0.18])
                cols[0].write(f"• {src}")
                if cols[1].button("🗑", key=f"del_{src}", help=f"Remove {src}"):
                    from rag import catalog, vectorstore

                    vectorstore.delete_source(user_email, src)
                    catalog.remove(user_email, src)
                    _sources_for.clear()
                    st.rerun()
            remaining = len(visible) - limit
            if remaining > 0:
                if st.button(
                    f"Show {remaining:,} more", key="show_more_docs", use_container_width=True
                ):
                    st.session_state.doc_list_show_all = True
                    st.rerun()
        if sources:
            with st.popover("🧹 Remove all documents", use_container_width=True):
                st.caption(
                    "Deletes your entire index. Files on your drives are "
                    "not touched."
                )
                if st.button(
                    "Yes, remove everything",
                    type="primary",
                    key="wipe_all_docs",
                    use_container_width=True,
                ):
                    from rag import catalog, vectorstore

                    vectorstore.reset_user(user_email)
                    catalog.delete_all(user_email)
                    _sources_for.clear()
                    st.rerun()


def _index_uploads(user_email: str, uploads) -> None:
    """Validate, save, and index PDFs selected through a folder upload.

    Security: caps the batch size, rejects oversized files and anything that
    isn't really a PDF, and writes to a sanitized, hash-namespaced path so a
    crafted file name can't escape the upload directory.
    """
    from rag import catalog, pipeline

    config.ensure_dirs()

    if len(uploads) > security.MAX_FILES_PER_UPLOAD:
        st.warning(
            f"Only the first {security.MAX_FILES_PER_UPLOAD} files will be "
            f"indexed (you selected {len(uploads)})."
        )
        uploads = uploads[: security.MAX_FILES_PER_UPLOAD]

    existing = set(_sources_for(user_email))  # also seeds the catalog if needed
    progress = st.progress(0.0, text="Starting…")
    total = len(uploads)
    # One session for the whole batch: each record is still committed to
    # disk immediately (see rag.catalog), just without reopening a fresh
    # SQLite connection per file.
    with catalog.session(user_email) as cat:
        for i, uploaded in enumerate(uploads, start=1):
            # Streamlit's directory uploader includes files from nested
            # folders. Keep their safe relative names so citations identify
            # the right PDF.
            display_name = security.sanitize_source_name(uploaded.name)
            data = bytes(uploaded.getbuffer())

            # --- validation gates ---
            if len(data) > security.MAX_FILE_MB * 1024 * 1024:
                st.error(f"{display_name}: larger than {security.MAX_FILE_MB} MB — skipped.")
                progress.progress(i / total)
                continue
            if not security.looks_like_pdf(data):
                st.error(f"{display_name}: doesn't look like a real PDF — skipped.")
                progress.progress(i / total)
                continue
            if (
                display_name not in existing
                and len(existing) >= security.MAX_DOCS_PER_USER
            ):
                st.error(
                    f"{display_name}: document limit reached "
                    f"({security.MAX_DOCS_PER_USER} per account) — skipped."
                )
                progress.progress(i / total)
                continue

            dest = config.UPLOAD_DIR / security.storage_name(user_email, display_name)
            progress.progress((i - 0.5) / total, text=f"Indexing {display_name}…")
            try:
                dest.write_bytes(data)
                result = pipeline.ingest_pdf(user_email, dest, source_name=display_name)
                record = {
                    "origin": "upload",
                    "size": len(data),
                    "mtime": 0,
                    "pages": result.pages,
                    "chunks": result.chunks,
                    "ocr": result.ocr_pages > 0,
                }
                cat.upsert(display_name, record)  # kept even at chunks==0 so a
                # re-upload of the same unreadable file doesn't retry forever
                if result.chunks == 0:
                    st.warning(
                        f"{display_name}: no extractable text found, even "
                        "after OCR — it may be blank or badly scanned."
                    )
                else:
                    existing.add(display_name)
                    msg = f"Indexed {result.source}: {result.pages} pages, {result.chunks} chunks"
                    if result.ocr_pages:
                        msg += f" ({result.ocr_pages} via OCR)"
                    st.toast(msg, icon="✅")
            except Exception as exc:  # surface, don't crash the app
                st.error(f"Failed to index {display_name}: {exc}")
            finally:
                # The chunked text now lives in the vector store; keeping the
                # raw PDF around would only grow the disk (it is never read
                # again).
                dest.unlink(missing_ok=True)
            progress.progress(i / total, text=f"Done {i}/{total}")
    _sources_for.clear()
    # New key on rerun -> a fresh, empty file_uploader instead of one that
    # still shows the files just indexed.
    st.session_state.uploader_nonce += 1
    time.sleep(0.4)
    progress.empty()
    st.rerun()


def _drive_scan_picker(user_email: str) -> None:
    """Drive selector + scan button.

    Runs the scan synchronously (see ``_run_drive_scan``) rather than in a
    background thread. A background thread's progress needs some way to
    redraw a page that's already been sent to the browser — a fragment's own
    auto-tick and a full-page reload loop were both tried here and neither
    reliably reached the browser. A synchronous loop updating ``st.progress``
    directly is the same pattern folder uploads already use in this app
    (see ``_index_uploads``) and it streams to the browser live, no redraw
    scheme needed.
    """
    from rag import drive_scan

    drives = drive_scan.list_drives()
    if not drives:
        st.info("No drives detected. Connect a drive and try again.")
        return
    drive = st.selectbox(
        "Choose a drive",
        drives,
        format_func=lambda d: d.describe(),
        key="drive_choice",
    )
    if st.button("🔍 Scan & index this drive", type="primary", use_container_width=True):
        _run_drive_scan(user_email, drive.root)


def _run_drive_scan(user_email: str, root: str) -> None:
    """Discover and index every PDF on ``root``, showing live progress.

    Files are read in place (never copied), and each one is committed to the
    catalog as soon as it's indexed — so if this is interrupted (closing the
    tab, clicking elsewhere, which abandons this script run like any other
    Streamlit rerun does), a later re-scan skips everything already done
    (matching size/mtime against the catalog) and only touches what's left.
    For an unattended, multi-terabyte, multi-day scan that must survive the
    browser closing entirely, use ``scripts/bulk_index.py`` instead — this
    button is for the common case of scanning a drive while watching it.
    """
    from rag import catalog, drive_scan, pipeline

    progress = st.progress(0.0, text=f"Scanning {root} …")
    stats: dict = {}
    pdfs: list = []
    for path in drive_scan.iter_pdfs(root, stats):
        pdfs.append(path)
        if len(pdfs) % 10 == 0:
            progress.progress(
                0.0,
                text=f"Scanning {root} … {len(pdfs):,} PDFs found "
                f"({stats.get('dirs', 0):,} folders searched)",
            )

    if not pdfs:
        progress.empty()
        st.info(f"No PDF files were found on {root}.")
        return

    total = len(pdfs)
    _sources_for(user_email)  # seeds the catalog for pre-catalog accounts
    max_bytes = security.MAX_DRIVE_FILE_MB * 1024 * 1024
    new = skipped = no_text = too_big = failed = ocr_used = 0
    with catalog.session(user_email) as cat:
        for i, path in enumerate(pdfs, start=1):
            source = drive_scan.source_name_for(root, path)
            progress.progress((i - 1) / total, text=f"Indexing {i:,}/{total:,} — {path.name}")
            try:
                fstat = path.stat()
            except OSError:
                failed += 1
            else:
                rec = cat.get(source)
                if (
                    rec
                    and rec.get("size") == fstat.st_size
                    and rec.get("mtime") == int(fstat.st_mtime)
                ):
                    skipped += 1
                elif fstat.st_size > max_bytes:
                    too_big += 1
                else:
                    try:
                        result = pipeline.ingest_pdf(user_email, path, source_name=source)
                        # Zero-chunk files are recorded too, so re-scans skip them.
                        cat.upsert(
                            source,
                            {
                                "origin": "drive",
                                "path": drive_scan.strip_extended(path),
                                "size": fstat.st_size,
                                "mtime": int(fstat.st_mtime),
                                "pages": result.pages,
                                "chunks": result.chunks,
                                "ocr": result.ocr_pages > 0,
                            },
                        )
                        if result.chunks == 0:
                            no_text += 1
                        else:
                            new += 1
                        if result.ocr_pages:
                            ocr_used += 1
                    except Exception:
                        failed += 1
            progress.progress(i / total, text=f"Indexed {i:,}/{total:,} — {path.name}")
    _sources_for.clear()
    progress.empty()
    st.toast(f"✅ Scan of {root} complete!", icon="✅")
    st.success(f"Scan of {root} complete.")
    st.caption(
        f"Newly indexed: {new:,} ({ocr_used:,} needed OCR) · already up to "
        f"date: {skipped:,} · no extractable text: {no_text:,} · larger "
        f"than {security.MAX_DRIVE_FILE_MB} MB: {too_big:,} · failed: {failed:,}"
    )


def _navbar(user_email: str) -> None:
    """Render the navigation as a fixed bar and reserve room beneath it."""
    # A keyed container gets the stable ``st-key-app-navbar`` CSS class.
    # It stays visible while the content below it scrolls normally.
    with st.container(key="app-navbar"):
        _navbar_content(user_email)
    st.markdown(
        '<div class="app-navbar-spacer" aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )


def _navbar_content(user_email: str) -> None:
    """Contents of the fixed navigation bar."""
    left, right = st.columns([0.74, 0.26], vertical_alignment="center")
    with left:
        st.markdown(
            '<div class="nav-brand">📄 PDF RAG '
            '<span class="nav-accent">Assistant</span></div>',
            unsafe_allow_html=True,
        )
    with right:
        with st.popover(f"👤  {user_email}", use_container_width=True):
            st.markdown("**Account**")
            st.write(user_email)
            st.caption("🟢 Online")
            if st.button("Sign out", key="signout_top", use_container_width=True):
                st.session_state.user_email = None
                st.session_state.messages = []
                st.session_state.upload_mode = None
                st.rerun()


def chat_view(user_email: str) -> None:
    """Render the conversation and handle new questions."""
    _navbar(user_email)

    if not config.kimi_is_configured():
        st.warning(
            "No Kimi API key found. Copy `.env.example` to `.env` and set "
            "`KIMI_API_KEY` so the assistant can compose answers. You can still "
            "upload and index PDFs without it."
        )

    # Replay history.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ``st.chat_input`` stays pinned to the bottom of the viewport (like
    # ChatGPT/Claude) while the conversation scrolls above it.
    question = st.chat_input(
        "Ask a question about your PDFs…",
        key="question_input",
    )

    if question and question.strip():
        _handle_question(user_email, question.strip())


def _handle_question(user_email: str, question: str) -> None:
    """Render the new exchange live and store it — the standard, single-pass
    Streamlit chat pattern. An earlier version forced two extra reruns per
    question to avoid a source-panel double-render bug; now that the source
    panel is disabled entirely that workaround serves no purpose, and it was
    the more likely cause of a separate glitch (Streamlit's own internal
    layout spacers ending up misplaced mid-message on very long answers).
    """
    from rag import pipeline

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    # Always called here, unconditionally, at this one fixed point — the
    # LLM call below takes several seconds, plenty of time for this to load
    # and run before the "searching..." spinner or the answer even render,
    # so neither ends up sitting behind the fixed input, invisible, while
    # they're on screen.
    _scroll_to_bottom()

    with st.chat_message("assistant"):
        if not _sources_for(user_email):
            text = "You haven't indexed any PDFs yet. Upload some from the sidebar first."
            st.markdown(text)
            st.session_state.messages.append({"role": "assistant", "content": text})
            _scroll_to_bottom()
            return
        with st.spinner("Searching your documents and thinking…"):
            answer = pipeline.ask(user_email, question)
        st.markdown(answer.text)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer.text}
        )
    _scroll_to_bottom()  # safety net in case the answer made the page taller


def _scroll_to_bottom() -> None:
    """Scroll the chat area to its true end right after a new answer renders.

    The fixed chat input only stays clear of content once scrolled all the
    way down (the reserved padding-bottom lives past the last message);
    without this, a fresh answer renders above the fold and looks like it's
    partly hidden behind the input until the user manually scrolls further.

    Targets the bounded main pane specifically rather than the whole
    document. Scrolling the whole page used to mean reflowing *everything*
    — every fixed-position element recalculated against a document height
    that keeps growing with the conversation — which got slower, and less
    reliable, the longer a session ran. The main pane is now a fixed 100vh
    region with its own scrollbar (see inject_styles()), so this is a
    cheap, isolated scroll regardless of how much history it holds.

    Streamlit uses "stAppScrollToBottomContainer" as this pane's testid on
    any page with st.chat_input (plain "stMain" elsewhere) — both are
    queried since only one is ever present on a given page.
    """
    components.html(
        """
        <script>
        (function () {
            const doc = window.top.document;
            const main = doc.querySelector('[data-testid="stAppScrollToBottomContainer"]')
                || doc.querySelector('[data-testid="stMain"]');
            if (main) {
                main.scrollTop = main.scrollHeight;
            }
        })();
        </script>
        """,
        height=0,
    )


def main() -> None:
    _init_state()
    _start_backend_warmup()
    inject_styles()
    if not st.session_state.user_email:
        login_view()
        return
    user_email = st.session_state.user_email
    sidebar(user_email)
    chat_view(user_email)


if __name__ == "__main__":
    main()
