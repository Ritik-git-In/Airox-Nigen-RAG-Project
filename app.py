"""Streamlit UI for the PDF RAG assistant.

Flow: log in with an email -> upload PDFs (indexed into your private vector
store) -> ask questions and get answers with citations.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os

# chromadb pulls in opentelemetry, whose protobuf-generated code clashes with a
# newer protobuf runtime on some hosts (e.g. Streamlit Cloud), raising a
# TypeError in _CheckCalledFromGeneratedFile at import time. Force the pure-Python
# protobuf parser so the import succeeds regardless of the installed version.
# Must be set BEFORE chromadb is imported (i.e. before the `from rag` import).
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import time

import streamlit as st
import streamlit.components.v1 as components

from rag import auth, config, pipeline, security, vectorstore

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
        /* Keep the header (it holds the sidebar toggle) but make it blend in. */
        [data-testid="stHeader"] { background: transparent; }

        /* Soft gradient backdrop */
        .stApp {
            background:
                radial-gradient(1100px 520px at 12% -8%, #1d2748 0%, rgba(13,17,28,0) 55%),
                radial-gradient(900px 500px at 100% 0%, #21183f 0%, rgba(13,17,28,0) 50%),
                #0B0E16;
        }

        /* Pull everything up: Streamlit's default top padding is very large */
        [data-testid="stMainBlockContainer"], .block-container {
            padding-top: 0.8rem !important;
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
            background: rgba(20,26,40,0.9);
            box-shadow: 0 8px 30px rgba(0,0,0,0.35);
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: rgba(12,15,24,0.92);
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


def auto_scroll(nonce: int) -> None:
    """Scroll the chat area to the bottom (like ChatGPT) after each render.

    ``nonce`` (the message count) is embedded so the component re-runs its
    script whenever a new message is added.
    """
    components.html(
        f"""
        <script>
        // nonce={nonce}
        const targets = [
            'section.stMain', '[data-testid="stMain"]', '.main',
            '[data-testid="stAppViewContainer"]'
        ];
        function toBottom() {{
            const doc = window.parent.document;
            targets.forEach(sel => {{
                doc.querySelectorAll(sel).forEach(el => {{
                    el.scrollTo({{ top: el.scrollHeight, behavior: 'smooth' }});
                }});
            }});
        }}
        [0, 90, 220, 450].forEach(t => setTimeout(toBottom, t));
        </script>
        """,
        height=0,
    )


# --------------------------------------------------------------------------- #
# State                                                                        #
# --------------------------------------------------------------------------- #
def _init_state() -> None:
    st.session_state.setdefault("user_email", None)
    st.session_state.setdefault("messages", [])  # list[dict(role, content, sources)]
    st.session_state.setdefault("ask_times", [])  # timestamps for rate limiting


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
                email = st.text_input("Email", placeholder="you@example.com")
                pw = st.text_input("Password", type="password")
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
                email2 = st.text_input("Email", placeholder="you@example.com", key="su_email")
                pw1 = st.text_input(
                    f"Password (min {auth.MIN_PASSWORD_LEN} characters)",
                    type="password",
                    key="su_pw1",
                )
                pw2 = st.text_input("Confirm password", type="password", key="su_pw2")
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
    """Upload area + the user's indexed documents."""
    with st.sidebar:
        st.subheader("Upload PDFs")
        uploads = st.file_uploader(
            "Drop your PDFs here", type=["pdf"], accept_multiple_files=True
        )
        if uploads and st.button(
            "Index uploaded PDFs", type="primary", use_container_width=True
        ):
            _index_uploads(user_email, uploads)

        st.divider()
        st.subheader("Your documents")
        sources = vectorstore.list_sources(user_email)
        if not sources:
            st.caption("No documents indexed yet.")
        for src in sources:
            cols = st.columns([0.82, 0.18])
            cols[0].write(f"• {src}")
            if cols[1].button("🗑", key=f"del_{src}", help=f"Remove {src}"):
                vectorstore.delete_source(user_email, src)
                st.rerun()


def _index_uploads(user_email: str, uploads) -> None:
    """Validate, save, and index each uploaded PDF.

    Security: caps the batch size, rejects oversized files and anything that
    isn't really a PDF, and writes to a sanitized, hash-namespaced path so a
    crafted file name can't escape the upload directory.
    """
    config.ensure_dirs()

    if len(uploads) > security.MAX_FILES_PER_UPLOAD:
        st.warning(
            f"Only the first {security.MAX_FILES_PER_UPLOAD} files will be "
            f"indexed (you selected {len(uploads)})."
        )
        uploads = uploads[: security.MAX_FILES_PER_UPLOAD]

    progress = st.progress(0.0, text="Starting…")
    total = len(uploads)
    for i, uploaded in enumerate(uploads, start=1):
        display_name = security.sanitize_filename(uploaded.name)
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
                st.toast(
                    f"Indexed {result.source}: {result.pages} pages, "
                    f"{result.chunks} chunks",
                    icon="✅",
                )
        except Exception as exc:  # surface, don't crash the app
            st.error(f"Failed to index {display_name}: {exc}")
        progress.progress(i / total, text=f"Done {i}/{total}")
    time.sleep(0.4)
    progress.empty()
    st.rerun()


def _navbar(user_email: str) -> None:
    """Compact top navigation bar: brand on the left, user menu on the right."""
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
                st.rerun()
    st.markdown('<hr class="nav-divider">', unsafe_allow_html=True)


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

    question = st.chat_input("Ask a question about your PDFs…")
    if question:
        question = question.strip()[: security.MAX_QUESTION_CHARS]
        if security.rate_limited(st.session_state.ask_times):
            st.warning(
                f"You're sending questions too fast. Please wait a moment "
                f"(limit: {security.RATE_LIMIT_MAX} per "
                f"{security.RATE_LIMIT_WINDOW_S}s)."
            )
        elif question:
            _handle_question(user_email, question)

    # Keep the latest message in view, ChatGPT-style.
    auto_scroll(len(st.session_state.messages))


def _handle_question(user_email: str, question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not vectorstore.list_sources(user_email):
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
    inject_styles()
    if not st.session_state.user_email:
        login_view()
        return
    user_email = st.session_state.user_email
    sidebar(user_email)
    chat_view(user_email)


if __name__ == "__main__":
    main()
