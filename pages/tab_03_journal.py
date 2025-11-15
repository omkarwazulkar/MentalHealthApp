import streamlit as st # type: ignore

def tab_journal():
    st.subheader("Your Journal")
    journal_text = st.text_area("Write your thoughts for today:")
    if st.button("Save Journal Entry"):
        st.success("Journal saved!")