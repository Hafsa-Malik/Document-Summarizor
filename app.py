import streamlit as st
from langchain.document_loaders import PyPDFLoader
from transformers import BartForConditionalGeneration, BartTokenizer
import base64

def parse_file(file):
    loader =  PyPDFLoader(file)
    pages = loader.load_and_split()
    final_texts = ""
    for page in pages:
        final_texts = final_texts + page.page_content
    return final_texts

def llm_pipeline(filepath, max_length, min_length):
    input_text = parse_file(filepath)
    base_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = base_model.generate(inputs, max_length=max_length, min_length=min_length)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Document Summarization Using `facebook/bart-large-cnn`")
    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    values = st.slider('Customize min_length and max_length', 20, 1000, (50, 500))
    st.write('Current min_length:', values[0])
    st.write('Current max_length:', values[1])

    if uploaded_file is not None:
        if st.button("Summarize"):
            col1, col2 = st.columns(2)
            filepath = uploaded_file.name
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                st.info("Uploaded File")
                pdf_view = displayPDF(filepath)
            with col2:
                with st.spinner("Summarizing text, this might take a few seconds depending on document size..."):
                    summary = llm_pipeline(filepath, values[1], values[0])
                    st.info("Summarization Complete")
                    st.success(summary)