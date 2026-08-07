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

        /* Hide default Streamlit chrome: menu, footer, Deploy button,
           and the "running… / Stop" status widget that flashes on rerun. */
        #MainMenu, footer,
        [data-testid="stStatusWidget"],
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
        [data-testid="stExpandSidebarButton"] {
            position: fixed !important;
            left: 10px !important;
            top: 10px !important;
            z-index: 999999 !important;
            width: 40px !important;
            height: 40px !important;
            pointer-events: auto !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            background-color: rgba(255,255,255,0.08) !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            border-radius: 12px !important;
            box-shadow: 0 10px 30px rgba(0,0,0,0.35) !important;
        }
        [data-testid="stExpandSidebarButton"] span {
            width: auto !important;
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
            padding-bottom: 6.0rem !important; /* more bottom space so inputs sit lower */
            min-height: calc(100vh - 80px) !important; /* keep the main area tall so the question box appears lower */
        }

        /* Ensure header children stretch full width and don't get clipped */
        [data-testid="stHeader"] > div, [data-testid="stHeader"] * {
            width: 100% !important;
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
            padding: 10px clamp(1.25rem, 5.2vw, 6.25rem) 8px !important;
            background: #050505;
            box-shadow: 0 10px 24px rgba(0,0,0,0.38);
        }
        /* A fixed element does not occupy document space. This prevents the
           first chat message from being hidden beneath the navbar. */
        .app-navbar-spacer { height: 76px; }

        @media (max-width: 640px) {
            .st-key-app-navbar {
                padding-left: 4.5rem !important;
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
            animation: fadeInUp 0.3s ease both;
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

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(12px); }
            to   { opacity: 1; transform: translateY(0); }
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

        /* Chat input */
        [data-testid="stChatInput"] {
            border-radius: 14px;
            border: 1px solid rgba(124,108,246,.45);
            background: rgba(20,26,40,0.95);
            box-shadow: 0 10px 36px rgba(0,0,0,0.45);
            margin-top: 40px !important; /* nudge the question box significantly lower within its container */
            transition: margin-top .18s ease;
        }

        /* If the question input sits inside a PDF display node, ensure it's pushed further down there too */
        .stApp [data-testid="stMainBlockContainer"] .stFileUploaderDropzone ~ div [data-testid="stChatInput"] {
            margin-top: 56px !important;
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


def inject_header_toggle() -> None:
    """Inject a small top-left toggle button that opens/closes the sidebar.

    The script creates a button in the page root and toggles the sidebar
    directly using the page DOM, then closes it when clicking outside.
    """
    components.html(
        """
        <script>
        (function () {
            const topDoc = window.top.document;
            if (!topDoc || !topDoc.body) return;

            const styleId = 'custom-sidebar-toggle-style';
            if (!topDoc.getElementById(styleId)) {
                const style = topDoc.createElement('style');
                style.id = styleId;
                style.textContent = `
                #custom-sidebar-toggle {
                    position: fixed !important;
                    top: 10px !important;
                    left: 10px !important;
                    z-index: 999999 !important;
                    width: 46px !important;
                    height: 46px !important;
                    border-radius: 16px !important;
                    background: rgba(255,255,255,0.12) !important;
                    border: 1px solid rgba(255,255,255,0.16) !important;
                    display: grid !important;
                    place-items: center !important;
                    cursor: pointer !important;
                    box-shadow: 0 20px 50px rgba(0,0,0,0.45) !important;
                    transition: transform .16s ease, background .16s ease !important;
                    pointer-events: auto !important;
                }
                #custom-sidebar-toggle:hover {
                    transform: translateY(-1px) !important;
                    background: rgba(255,255,255,0.20) !important;
                }
                #custom-sidebar-toggle span {
                    font-size: 20px !important;
                    color: #eef2ff !important;
                    line-height: 1 !important;
                }
                [data-testid="stExpandSidebarButton"] {
                    display: none !important;
                }
                [data-testid="stSidebar"],
                .stSidebar.st-emotion-cache-l7lcii {
                    transform: translateX(var(--custom-sidebar-x, -300px)) !important;
                    min-width: var(--custom-sidebar-min-width, 0px) !important;
                    max-width: var(--custom-sidebar-max-width, 0px) !important;
                    width: var(--custom-sidebar-width, 0px) !important;
                    overflow: var(--custom-sidebar-overflow, hidden) !important;
                    visibility: var(--custom-sidebar-visibility, hidden) !important;
                    opacity: var(--custom-sidebar-opacity, 0) !important;
                    transition: transform 320ms ease !important, min-width 320ms ease !important, max-width 320ms ease !important, width 320ms ease !important, opacity 320ms ease !important;
                }
                [data-testid="stSidebar"][aria-expanded="true"],
                .stSidebar.st-emotion-cache-l7lcii[aria-expanded="true"] {
                    --custom-sidebar-x: 0px !important;
                    --custom-sidebar-min-width: 300px !important;
                    --custom-sidebar-max-width: 300px !important;
                    --custom-sidebar-width: 300px !important;
                    --custom-sidebar-overflow: visible !important;
                    --custom-sidebar-visibility: visible !important;
                    --custom-sidebar-opacity: 1 !important;
                }
                [data-testid="stSidebar"][aria-expanded="false"],
                .stSidebar.st-emotion-cache-l7lcii[aria-expanded="false"] {
                    --custom-sidebar-x: -300px !important;
                    --custom-sidebar-min-width: 0px !important;
                    --custom-sidebar-max-width: 0px !important;
                    --custom-sidebar-width: 0px !important;
                    --custom-sidebar-overflow: hidden !important;
                    --custom-sidebar-visibility: hidden !important;
                    --custom-sidebar-opacity: 0 !important;
                }
                `;
                topDoc.head.appendChild(style);
            }

            function getSidebar() {
                const docs = [topDoc].concat(
                    Array.from(topDoc.querySelectorAll('iframe')).map((frame) => {
                        try {
                            return frame.contentDocument || frame.contentWindow?.document || null;
                        } catch (err) {
                            return null;
                        }
                    }).filter(Boolean)
                );
                for (const doc of docs) {
                    const sidebar = doc.querySelector('[data-testid="stSidebar"]');
                    if (sidebar) return {sidebar, doc};
                }
                return null;
            }

            function isSidebarOpen(found) {
                if (!found) return false;
                const cs = found.doc.defaultView.getComputedStyle(found.sidebar);
                const rect = found.sidebar.getBoundingClientRect();
                return rect.left >= 0 && cs.visibility !== 'hidden' && cs.opacity !== '0';
            }

            function setSidebarOpen(open) {
                const found = getSidebar();
                if (!found) return;
                const {sidebar} = found;
                if (open) {
                    sidebar.style.setProperty('--custom-sidebar-x', '0px', 'important');
                    sidebar.style.setProperty('--custom-sidebar-min-width', '300px', 'important');
                    sidebar.style.setProperty('--custom-sidebar-max-width', '300px', 'important');
                    sidebar.style.setProperty('--custom-sidebar-width', '300px', 'important');
                    sidebar.style.setProperty('--custom-sidebar-overflow', 'visible', 'important');
                    sidebar.style.removeProperty('visibility');
                    sidebar.style.removeProperty('opacity');
                    sidebar.setAttribute('aria-expanded', 'true');
                } else {
                    sidebar.style.setProperty('--custom-sidebar-x', '-300px', 'important');
                    sidebar.style.setProperty('--custom-sidebar-min-width', '0px', 'important');
                    sidebar.style.setProperty('--custom-sidebar-max-width', '0px', 'important');
                    sidebar.style.setProperty('--custom-sidebar-width', '0px', 'important');
                    sidebar.style.setProperty('--custom-sidebar-overflow', 'hidden', 'important');
                    sidebar.setAttribute('aria-expanded', 'false');
                }
            }

            function toggleSidebar(ev) {
                if (ev) {
                    ev.preventDefault();
                    ev.stopPropagation();
                }
                const found = getSidebar();
                if (!found) return;
                setSidebarOpen(!isSidebarOpen(found));
            }

            let toggleButton = topDoc.getElementById('custom-sidebar-toggle');
            if (!toggleButton) {
                toggleButton = topDoc.createElement('button');
                toggleButton.id = 'custom-sidebar-toggle';
                toggleButton.type = 'button';
                toggleButton.title = 'Toggle menu';
                toggleButton.innerHTML = '<span aria-hidden="true">☰</span>';
                toggleButton.addEventListener('click', toggleSidebar);
                topDoc.body.appendChild(toggleButton);
            } else {
                const replacement = toggleButton.cloneNode(true);
                replacement.removeAttribute('onclick');
                replacement.onclick = null;
                replacement.addEventListener('click', toggleSidebar);
                toggleButton.parentNode.replaceChild(replacement, toggleButton);
                toggleButton = replacement;
            }

            function hideNativeToggle() {
                const native = topDoc.querySelector('[data-testid="stExpandSidebarButton"]');
                if (native) native.style.setProperty('display', 'none', 'important');
            }
            hideNativeToggle();

            topDoc.addEventListener('click', function (event) {
                const found = getSidebar();
                if (!found) return;
                if (toggleButton.contains(event.target) || found.sidebar.contains(event.target)) return;
                if (isSidebarOpen(found)) {
                    setSidebarOpen(false);
                }
            }, true);

            topDoc.addEventListener('keydown', function (event) {
                if (event.key === 'Escape') {
                    const found = getSidebar();
                    if (found && isSidebarOpen(found)) {
                        setSidebarOpen(false);
                    }
                }
            });

            new MutationObserver(function () {
                if (!topDoc.getElementById('custom-sidebar-toggle')) {
                    topDoc.body.appendChild(toggleButton);
                }
                hideNativeToggle();
            }).observe(topDoc.body, { childList: true, subtree: true });
        })();
        </script>
        """,
        height=0,
    )


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

        uploads = []
        if st.session_state.upload_mode == "pdf":
            st.caption("Upload one PDF")
            upload = st.file_uploader(
                "Choose a PDF", type=["pdf"], key="single_pdf_upload"
            )
            uploads = [upload] if upload else []
        elif st.session_state.upload_mode == "folder":
            st.caption("Upload a folder — PDFs in subfolders are included")
            uploads = st.file_uploader(
                "Choose a folder containing PDFs",
                type=["pdf"],
                accept_multiple_files="directory",
                key="folder_pdf_upload",
                help="All PDFs in the selected folder and its subfolders are included.",
            )
        elif st.session_state.upload_mode == "drive":
            st.caption(
                "Index every PDF on a connected drive — files are read in "
                "place, nothing is copied or uploaded."
            )
            from rag import drive_scan

            drives = drive_scan.list_drives()
            if not drives:
                st.info("No drives detected. Connect a drive and try again.")
            else:
                drive = st.selectbox(
                    "Choose a drive",
                    drives,
                    format_func=lambda d: d.describe(),
                    key="drive_choice",
                )
                if st.button(
                    "🔍 Scan & index this drive",
                    type="primary",
                    use_container_width=True,
                ):
                    # Picked up by main() after the sidebar renders; the scan
                    # itself runs in the main area where there's room for
                    # progress reporting.
                    st.session_state.drive_scan_root = drive.root
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
        # Drive scans can index thousands of PDFs; rendering a row (plus a
        # delete button) for each would make every rerun crawl. Cap the list.
        _MAX_LISTED = 30
        for src in sources[:_MAX_LISTED]:
            cols = st.columns([0.82, 0.18])
            cols[0].write(f"• {src}")
            if cols[1].button("🗑", key=f"del_{src}", help=f"Remove {src}"):
                from rag import catalog, vectorstore

                vectorstore.delete_source(user_email, src)
                catalog.remove(user_email, src)
                _sources_for.clear()
                st.rerun()
        if len(sources) > _MAX_LISTED:
            st.caption(f"…and {len(sources) - _MAX_LISTED:,} more.")
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
    cat = catalog.load(user_email) or {}
    progress = st.progress(0.0, text="Starting…")
    total = len(uploads)
    for i, uploaded in enumerate(uploads, start=1):
        # Streamlit's directory uploader includes files from nested folders.
        # Keep their safe relative names so citations identify the right PDF.
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
            if result.chunks == 0:
                st.warning(
                    f"{display_name}: no extractable text found — it may be a "
                    "scanned image. (OCR isn't enabled yet.)"
                )
            else:
                existing.add(display_name)
                cat[display_name] = {
                    "origin": "upload",
                    "size": len(data),
                    "mtime": 0,
                    "pages": result.pages,
                    "chunks": result.chunks,
                }
                st.toast(
                    f"Indexed {result.source}: {result.pages} pages, "
                    f"{result.chunks} chunks",
                    icon="✅",
                )
        except Exception as exc:  # surface, don't crash the app
            st.error(f"Failed to index {display_name}: {exc}")
        finally:
            # The chunked text now lives in the vector store; keeping the raw
            # PDF around would only grow the disk (it is never read again).
            dest.unlink(missing_ok=True)
        progress.progress(i / total, text=f"Done {i}/{total}")
    catalog.save(user_email, cat)
    _sources_for.clear()
    time.sleep(0.4)
    progress.empty()
    st.rerun()


def _drive_scan_view(user_email: str, root: str) -> None:
    """Discover and index every PDF on ``root``, with live progress.

    Built for terabyte-scale drives: files are read in place (never copied),
    and progress is checkpointed to the catalog every few files — so stopping
    midway is safe, and a re-scan skips every file that is already indexed
    and unchanged (same size + mtime), only touching what's new or modified.
    """
    from rag import catalog, drive_scan, pipeline

    st.markdown(f"### 🔍 Scanning Drive {root}")
    st.caption(
        "You can stop anytime — scanning again later resumes where it left "
        "off, and only new or changed PDFs are re-indexed."
    )

    # --- Phase 1: discover every PDF on the drive ---
    found_ph = st.empty()
    stats: dict = {}
    pdfs: list = []
    last_update = 0.0
    with st.spinner(f"Scanning all PDF files from Drive {root} — please wait…"):
        for path in drive_scan.iter_pdfs(root, stats):
            pdfs.append(path)
            now = time.time()
            if now - last_update > 0.3:  # throttle UI updates
                found_ph.info(
                    f"Scanning **{root}** … {len(pdfs):,} PDFs found "
                    f"({stats.get('dirs', 0):,} folders searched)"
                )
                last_update = now
    found_ph.success(
        f"Scan complete: found {len(pdfs):,} PDF file(s) on {root} "
        f"({stats.get('dirs', 0):,} folders searched)."
    )
    if not pdfs:
        st.info("No PDF files were found on this drive.")
        return

    # --- Phase 2: index only what's new or changed ---
    _sources_for(user_email)  # seeds the catalog for pre-catalog accounts
    cat = catalog.load(user_email) or {}
    max_bytes = security.MAX_DRIVE_FILE_MB * 1024 * 1024
    progress = st.progress(0.0, text="Preparing to index…")
    new = skipped = no_text = too_big = failed = 0
    dirty = 0
    last_update = 0.0
    for i, path in enumerate(pdfs, start=1):
        source = drive_scan.source_name_for(root, path)
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
                    cat[source] = {
                        "origin": "drive",
                        "path": drive_scan.strip_extended(path),
                        "size": fstat.st_size,
                        "mtime": int(fstat.st_mtime),
                        "pages": result.pages,
                        "chunks": result.chunks,
                    }
                    if result.chunks == 0:
                        no_text += 1
                    else:
                        new += 1
                    dirty += 1
                    if dirty % 25 == 0:
                        catalog.save(user_email, cat)  # checkpoint for resume
                except Exception:
                    failed += 1
        now = time.time()
        if now - last_update > 0.25 or i == len(pdfs):  # throttle UI updates
            progress.progress(
                i / len(pdfs), text=f"Indexing {i:,}/{len(pdfs):,} — {path.name}"
            )
            last_update = now
    catalog.save(user_email, cat)
    _sources_for.clear()

    st.success(
        "All PDF files have been scanned successfully. "
        "You can now ask any question related to the documents."
    )
    st.caption(
        f"Newly indexed: {new:,} · already up to date: {skipped:,} · "
        f"no extractable text: {no_text:,} · larger than "
        f"{security.MAX_DRIVE_FILE_MB} MB: {too_big:,} · failed: {failed:,}"
    )
    if st.button("💬 Start asking questions", type="primary"):
        st.rerun()


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
            if msg.get("sources"):
                _render_sources(msg["sources"])

    # ``st.chat_input`` stays pinned to the bottom of the viewport (like
    # ChatGPT/Claude) while the conversation scrolls above it.
    question = st.chat_input(
        "Ask a question about your PDFs…",
        key="question_input",
    )

    if question and question.strip():
        question = question.strip()
        _handle_question(user_email, question)

def _handle_question(user_email: str, question: str) -> None:
    from rag import pipeline

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not _sources_for(user_email):
            text = "You haven't indexed any PDFs yet. Upload some from the sidebar first."
            st.markdown(text)
            st.session_state.messages.append({"role": "assistant", "content": text})
            return
        with st.spinner("Searching your documents and thinking…"):
            answer = pipeline.ask(user_email, question)
        st.markdown(answer.text)
        if answer.sources:
            _render_sources(answer.sources)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer.text,
                "sources": [
                    {"source": s.source, "page": s.page, "text": s.text}
                    for s in answer.sources
                ],
            }
        )


def _render_sources(sources: list) -> None:
    """Show the retrieved chunks so the user can verify where the answer came from."""
    with st.expander(f"📑 Sources used ({len(sources)})"):
        for i, s in enumerate(sources, start=1):
            src = s["source"] if isinstance(s, dict) else s.source
            page = s["page"] if isinstance(s, dict) else s.page
            text = s["text"] if isinstance(s, dict) else s.text
            st.markdown(f"**[Source {i}] {src} — page {page}**")
            st.caption(text[:600] + ("…" if len(text) > 600 else ""))


def main() -> None:
    _init_state()
    _start_backend_warmup()
    inject_styles()
    inject_header_toggle()
    if not st.session_state.user_email:
        login_view()
        return
    user_email = st.session_state.user_email
    sidebar(user_email)
    scan_root = st.session_state.pop("drive_scan_root", None)
    if scan_root:
        _navbar(user_email)
        _drive_scan_view(user_email, scan_root)
        return
    chat_view(user_email)


if __name__ == "__main__":
    main()
