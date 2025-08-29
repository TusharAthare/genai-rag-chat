import os
import streamlit as st


def get_api_key() -> str:
    """Return an API key stored only in the Streamlit session.

    Behavior:
    - If a key exists in `st.session_state["api_key"]`, use it.
    - Otherwise, render a password input in the sidebar to let the user
      enter the key for this session (not persisted to disk).
    - Optionally allow using `GEMINI_API_KEY` from the environment as a convenience.
    - Provide a clear button to remove it from the current session.
    """
    ss = st.session_state

    # If key already set in session, offer a clear button and return it
    if ss.get("api_key"):
        with st.sidebar:
            st.caption("API key is set for this session.")
            if st.button("Clear API Key"):
                ss.pop("api_key", None)
                st.rerun()
        return ss["api_key"]

    env_key = os.getenv("GEMINI_API_KEY", "")

    # Ask for key in sidebar
    with st.sidebar:
        st.subheader("API Key")
        st.caption("Stored only in this browser session. It clears when the tab closes.")
        key_input = st.text_input("Gemini API Key", type="password", key="api_key_input")

        cols = st.columns(2 if env_key else 1)
        use_clicked = cols[0].button("Use Key", type="primary")
        used_env = False
        if env_key:
            used_env = cols[-1].button("Use Environment Key")

        if use_clicked:
            if key_input.strip():
                ss["api_key"] = key_input.strip()
                st.rerun()
            else:
                st.warning("Please enter a valid API key.")
        elif used_env:
            ss["api_key"] = env_key
            st.rerun()

    st.info("Enter your API key in the sidebar to continue.")
    st.stop()
