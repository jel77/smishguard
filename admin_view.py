import os

import pandas as pd
import streamlit as st

import admin_store


def render():
    st.title("Admin inbox", anchor=False, icon=":material/admin_panel_settings:")
    st.caption("Review, organize, export, or delete messages submitted for analysis.")

    admin_store.initialize()
    messages = admin_store.list_messages()
    st.metric("Saved analyses", f"{len(messages):,}")

    with st.container(border=True):
        search_col, verdict_col, category_col = st.columns([3, 1, 1])
        search = search_col.text_input("Search message or sender", placeholder="Type to search")
        verdict = verdict_col.selectbox("Model verdict", ["All", "Smishing", "Legitimate", "Uncertain"])
        category = category_col.selectbox("Admin category", ["All", *admin_store.CATEGORIES])
    filtered = admin_store.list_messages(search=search, verdict=verdict, category=category)

    if not filtered:
        st.info("No messages match these filters.")
    else:
        df = pd.DataFrame(filtered)
        df["probabilities"] = df["probabilities"].map(
            lambda values: ", ".join(f"{name}: {probability:.1%}" for name, probability in values.items())
        )
        columns = ["id", "created_at", "sender", "text", "verdict", "probabilities", "category", "notes"]
        edited = st.data_editor(
            df[columns],
            width="stretch",
            hide_index=True,
            disabled=["id", "created_at", "sender", "text", "verdict", "probabilities"],
            column_config={
                "id": st.column_config.NumberColumn("ID"),
                "created_at": st.column_config.TextColumn("Submitted (UTC)"),
                "sender": st.column_config.TextColumn("Sender"),
                "text": st.column_config.TextColumn("Message", width="large"),
                "verdict": st.column_config.TextColumn("Model verdict"),
                "probabilities": st.column_config.TextColumn("Model probabilities", width="large"),
                "category": st.column_config.SelectboxColumn(
                    "Admin category", options=admin_store.CATEGORIES, required=True
                ),
                "notes": st.column_config.TextColumn("Admin notes", max_chars=500, width="large"),
            },
            key="admin_message_editor",
        )

        export_df = pd.DataFrame(filtered)[columns]
        with st.container(horizontal=True, gap="small", vertical_alignment="center"):
            if st.button("Save categories and notes", icon=":material/save:"):
                admin_store.update_reviews(edited[["id", "category", "notes"]].to_dict("records"))
                st.toast("Admin updates saved.")
                st.rerun()

            st.download_button(
                "Export filtered messages as CSV",
                export_df.to_csv(index=False),
                file_name="smishguard-admin-messages.csv",
                mime="text/csv",
                icon=":material/download:",
            )

        message_ids = [int(message["id"]) for message in filtered]
        selected_ids = st.multiselect(
            "Select message IDs to delete",
            options=message_ids,
            format_func=lambda message_id: f"#{message_id}",
        )
        if st.button(
            "Delete selected",
            disabled=not selected_ids,
            icon=":material/delete:",
        ):
            admin_store.delete_messages(selected_ids)
            st.rerun()

    st.divider()
    confirm_delete_all = st.checkbox("I understand this permanently deletes every saved analysis.")
    if st.button(
        "Delete all saved messages",
        type="secondary",
        disabled=not confirm_delete_all or not messages,
        icon=":material/delete_forever:",
    ):
        admin_store.delete_all_messages()
        st.rerun()

    st.caption(
        f"Database: `{os.getenv('SMISHGUARD_DB_PATH', str(admin_store.DEFAULT_DB_PATH))}`. "
        "Use the same database path for every app process that should share this inbox."
    )
