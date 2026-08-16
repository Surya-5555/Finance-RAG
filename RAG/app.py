import os
import requests
# pyrefly: ignore [missing-import]
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Finance RAG", layout="wide")

st.title("Finance RAG System")
st.markdown("Upload quarterly financial reports and ask questions.")

# Sidebar for indexing
st.sidebar.header("Document Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload Quarterly PDFs", 
    type=["pdf"], 
    accept_multiple_files=True
)

if st.sidebar.button("Index Documents"):
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF.")
    else:
        with st.spinner("Processing and indexing documents..."):
            files_to_upload = []
            for f in uploaded_files:
                # read the file content
                files_to_upload.append(("files", (f.name, f.getvalue(), "application/pdf")))
            
            try:
                response = requests.post(f"{API_URL}/ingest", files=files_to_upload)
                if response.status_code == 200:
                    data = response.json()
                    st.sidebar.success(f"{data['files']} files processed, {data['chunks']} chunks stored.")
                else:
                    st.sidebar.error(f"Error: {response.text}")
            except Exception as e:
                st.sidebar.error(f"Failed to connect to API: {str(e)}")

st.sidebar.divider()
st.sidebar.header("System Stats")
if st.sidebar.button("Refresh Stats"):
    try:
        response = requests.get(f"{API_URL}/stats")
        if response.status_code == 200:
            stats = response.json()
            for k, v in stats.items():
                st.sidebar.text(f"{k}: {v}")
    except Exception as e:
        st.sidebar.error("Could not fetch stats.")


# Main UI for Asking
question = st.text_input("Ask a question about the uploaded documents:")
if st.button("Submit"):
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving and Generating Answer..."):
            try:
                response = requests.post(f"{API_URL}/ask", json={"question": question, "top_k": 4})
                if response.status_code == 200:
                    data = response.json()
                    st.markdown("### Answer")
                    st.write(data["answer"])
                    
                    if data.get("sources"):
                        st.markdown("### Sources")
                        for idx, source in enumerate(data["sources"]):
                            st.caption(f"[{idx+1}] File: {source['filename']}, Page: {source['page']}")
                            
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {str(e)}")
