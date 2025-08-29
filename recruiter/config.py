import os
import streamlit as st


def get_api_key() -> str:
    """Fetch required GEMINI_API_KEY from environment or stop the app.

    Returns:
        The API key string.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Please set GEMINI_API_KEY as an environment variable before running.")
        st.stop()
    return api_key

