import streamlit as st
import requests
import pandas as pd

# =========================
# API & Sheet URLs
# =========================
# TODO: Replace with your FastAPI endpoint (use ngrok if deployed)
API_URL = "https://monthly-earthworm-recharger.ngrok-free.dev/ask"

# TODO: Replace with your Google Sheet CSV export link
SHEET_URL = "https://docs.google.com/spreadsheets/d/1wT_dGrCYxDbmx2gX9TqvVxhGt3t9ax9NsAz-8ZySJy0/export?format=csv"

st.set_page_config(page_title="NileTel Assistant", layout="wide")

st.title("📡 NileTel Support Assistant")

# =========================
# 💬 ASK QUESTION
# =========================
st.subheader("Ask a question")

# TODO: Create a text input for the user to type a question

query = st.text_input("Type your question")


if st.button("Send"):
    if query:
        try:
            # TODO: Send the query to the API using requests.post
            response = requests.post(API_URL, json={"query": query})

            if response.status_code == 200:
                # TODO: Parse the JSON response
                data = response.json()

                st.markdown("### 💬 Answer")
                # TODO: Display the answer from the API
                st.success(data["answer"])

                st.markdown("### ⚙️ Needs Action")
                if data["needs_action"] == "YES":
                    st.warning("Ticket created / action triggered")
                else:
                    st.info("No action needed")

                st.markdown("### 📄 Sources")
                # TODO: Show the sources returned by the API
                st.write(data["sources"])

            else:
                st.error("API error")

        except Exception as e:
            st.error(f"Connection error: {e}")

# =========================
# 🎫 SHOW TICKETS
# =========================
st.divider()
st.subheader("📋 Tickets")

if st.button("Load Tickets"):
    try:
        # TODO: Read the Google Sheet CSV into a DataFrame
        df = pd.read_csv(SHEET_URL)
        

        st.success("Tickets loaded successfully")

        # TODO: Display the DataFrame in Streamlit
        st.dataframe(df)
        

    except Exception as e:
        st.error(f"Failed to load tickets: {e}")
