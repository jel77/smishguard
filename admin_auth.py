import hmac
import os

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


def configured_credentials():
    try:
        username = st.secrets.get("SMISHGUARD_ADMIN_USERNAME", "")
        password = st.secrets.get("SMISHGUARD_ADMIN_PASSWORD", "")
    except StreamlitSecretNotFoundError:
        username, password = "", ""
    return (
        username or os.getenv("SMISHGUARD_ADMIN_USERNAME", ""),
        password or os.getenv("SMISHGUARD_ADMIN_PASSWORD", ""),
    )


def render_sidebar_login():
    username, password = configured_credentials()
    with st.sidebar:
        st.divider()
        st.markdown("### Admin")
        if not username or not password:
            st.caption("Admin login is not configured.")
            return

        if st.session_state.get("admin_authenticated"):
            st.caption(f"Signed in as {username}.")
            if not st.session_state.get("admin_view") and st.button(
                "Open admin inbox", key="admin_sidebar_open", icon=":material/admin_panel_settings:"
            ):
                st.session_state["admin_view"] = True
                st.rerun()
            if st.button("Log out", key="admin_sidebar_logout", icon=":material/logout:"):
                st.session_state.pop("admin_authenticated", None)
                st.session_state["admin_view"] = False
                st.rerun()
            return

        with st.form("admin_sidebar_login"):
            submitted_username = st.text_input("Username")
            submitted_password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Admin sign in", type="primary")
        if submitted:
            valid_username = hmac.compare_digest(submitted_username, str(username))
            valid_password = hmac.compare_digest(submitted_password, str(password))
            if valid_username and valid_password:
                st.session_state["admin_authenticated"] = True
                st.rerun()
            st.error("Incorrect username or password.")
