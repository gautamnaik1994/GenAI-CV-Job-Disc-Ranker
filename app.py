import os
import re
import tempfile
from typing import Dict, List
import streamlit as st
import json
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_together import ChatTogether
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
# MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"


def init_llm():
    """Initialize the language model."""
    return ChatTogether(
        api_key=os.getenv("TOGETHER_AI_API_KEY"),
        temperature=0.0,
        model=MODEL_NAME
    )


@st.cache_resource
def get_embedding_model():
    """Initialize and cache the embedding model."""
    return HuggingFaceEmbeddings()


def get_text_splitter():
    """Initialize the text splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )


def process_pdf(file):
    """Process the uploaded PDF file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        return documents
    finally:
        os.remove(temp_file_path)


def create_qa_chain(documents, embedding_model):
    """Create QA chain from documents."""
    text_splitter = get_text_splitter()
    splits = text_splitter.split_documents(documents)

    if not splits:
        return None

    vectorstore = InMemoryVectorStore.from_documents(
        documents=splits,
        embedding=embedding_model
    )

    return RetrievalQA.from_chain_type(
        init_llm(),
        retriever=vectorstore.as_retriever()
    )


def process_multiple_pdfs(files) -> List[Dict]:
    """Process multiple PDF files and return a list of documents with metadata."""
    documents_with_metadata = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

            try:
                loader = PyPDFLoader(temp_file_path)
                documents = loader.load()
                candidate_name = file.name.replace('.pdf', '')
                for doc in documents:
                    doc.metadata['candidate_name'] = candidate_name
                documents_with_metadata.extend(documents)
            finally:
                os.remove(temp_file_path)

    return documents_with_metadata


def create_qa_chain_with_scoring(documents, embedding_model, candidate_name=None):
    """Create QA chain that returns matching scores for a specific candidate."""
    if candidate_name:
        documents = [doc for doc in documents if doc.metadata.get(
            'candidate_name') == candidate_name]

    text_splitter = get_text_splitter()
    splits = text_splitter.split_documents(documents)

    if not splits:
        return None

    vectorstore = InMemoryVectorStore.from_documents(
        documents=splits,
        embedding=embedding_model
    )

    return RetrievalQA.from_chain_type(
        init_llm(),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )


def analyze_candidate(qa_chain, query: str, candidate_name: str) -> Dict[str, float]:
    """Analyze a single candidate and return a matching score."""

    qa_template = f"""
    Based on the job description, evaluate how well the candidate {candidate_name} matches the requirements.
    Provide a matching score between 0 and 100, where 100 means perfect match.
    Job description: {query}
    
    Return ONLY a valid JSON object with this exact structure, without any explanation, markdown formatting, or code block indicators:
    {{
      "rating": <score between 0-100>,
      "reason": "<brief explanation for the score>"
    }}
    """

    default_score = {
        "rating": 0,
        "reason": "No score calculated"
    }
    response = qa_chain.invoke({"query": qa_template})
    try:
        result = response['result']
        result = result.replace('```json', '').replace('```', '')
        result = result.strip()
        result_json = json.loads(result)
        score = float(result_json.get("rating", 0))
        default_score["rating"] = min(100, max(0, score))
        default_score["reason"] = result_json.get(
            "reason", "No reason provided")
        return default_score

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        try:
            score = float(re.findall(r'\d+', response['result'])[0])
            score = min(100, max(0, score))
            default_score["rating"] = score
            default_score["reason"] = "No reason provided"
            return default_score
        except:
            return default_score
    except Exception as e:
        print(f"Unexpected error: {e}")
        return default_score


input_query = """We are looking for a machine learning engineer with experience in building and deploying machine learning models.
The candidate should have a background in Python, Generative AI, time series forecasting and cloud platforms like AWS or Azure."""


def main():
    """Main application function."""

    # add page title

    st.set_page_config(page_title="GenAI CV Job Description Ranker",
                       page_icon="🔍")

    st.title("GenAI CV Job Description Ranker")
    st.write(
        "Upload multiple CVs and get candidate rankings based on job description.")

    with st.expander("How it works"):
        st.markdown('''This app uses Generative AI to evaluate how well a candidate matches a job description. It converts the CV content into vectors using Hugging Face embeddings, and then uses a language model to evaluate the match between the job description and the candidate's CV. The app ranks the candidates based on the matching score and provides a brief explanation for the score.''')
        st.code('''
            Based on the job description, evaluate how well the candidate {candidate_name} matches the requirements.
            Provide a matching score between 0 and 100, where 100 means perfect match.
            Job description: {query}
            
            Return ONLY a valid JSON object with this exact structure, without any explanation, markdown formatting, or code block indicators:
            {{
            "rating": <score between 0-100>,
            "reason": "<brief explanation for the score>"
            }} 
            ''', language='markdown', wrap_lines=True)

    embedding_model = get_embedding_model()

    query = st.text_area("Enter job description",
                         height=200, value=input_query)
    uploaded_files = st.file_uploader(
        "Choose CV files (Max 5)", type="pdf", accept_multiple_files=True)

    if len(uploaded_files) > 5:
        st.warning("Please upload a maximum of 5 files at a time.")
        uploaded_files = uploaded_files[:5]

    if uploaded_files and query.strip():
        try:
            with st.spinner('Processing CVs...'):
                documents = process_multiple_pdfs(uploaded_files)

                if not documents:
                    st.warning("No text found in the uploaded files.")
                    return

                candidate_names = list(set(doc.metadata['candidate_name']
                                           for doc in documents))

                results = []
                progress_bar = st.progress(0)

                # Process each candidate separately
                for i, name in enumerate(candidate_names):
                    # Create a QA chain specific to this candidate
                    qa_chain = create_qa_chain_with_scoring(
                        documents, embedding_model, name)

                    if qa_chain:
                        score = analyze_candidate(qa_chain, query, name)
                        results.append(
                            {"name": name, "score": score["rating"], "reason": score["reason"]})

                    # Update progress
                    progress_bar.progress((i + 1) / len(candidate_names))

                ranked_candidates = sorted(results,
                                           key=lambda x: x['score'],
                                           reverse=True)

                st.subheader("Ranked Candidates:")
                for rank, candidate in enumerate(ranked_candidates, 1):
                    st.write(
                        f"{rank}. {candidate['name']} - Match Score: {candidate['score']:.1f}% - Reason: {candidate['reason']}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload CVs and enter a job description to begin matching.")


if __name__ == "__main__":
    main()
