import os
import fitz  # PyMuPDF
import streamlit as st
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.schema import Document
from langchain.chains.retrieval_qa.base import RetrievalQA
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
import torch
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

# 상수 정의
VECTORSTORE_PATH = "faiss_index"
PDF_DIRECTORY = "../data"

@st.cache_resource
def load_models():
    # 임베딩 모델 설정
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    
    # LLM 모델 설정
    model_name = "maum-ai/Llama-3-MAAL-8B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map="auto",
                                                 torch_dtype=torch.float32,
                                                 low_cpu_mem_usage=True)
    
    pipe = pipeline(
        "text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
        do_sample=True
    )
    
    local_llm = HuggingFacePipeline(pipeline=pipe)
    
    return embeddings, local_llm

def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def split_text(text):
    text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

@st.cache_resource
def create_or_load_vectorstore(_embeddings):
    if os.path.exists(VECTORSTORE_PATH):
        st.info("기존 벡터 저장소를 불러옵니다.")
        return FAISS.load_local(VECTORSTORE_PATH, _embeddings, allow_dangerous_deserialization=True)
    else:
        st.info("새로운 벡터 저장소를 생성합니다...")
        return create_new_vectorstore(_embeddings)

def create_new_vectorstore(_embeddings):
    documents = []
    for filename in os.listdir(PDF_DIRECTORY):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIRECTORY, filename)
            st.text(f"처리 중: {filename}")
            text = process_pdf(pdf_path)
            if text:
                chunks = split_text(text)
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={"source": filename, "chunk": i}
                    )
                    documents.append(doc)
    
    vectorstore = FAISS.from_documents(documents, _embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore

def main():
    st.title("서울대학교병원 연구비 행정 챗봇")
    
    embeddings, local_llm = load_models()
    
    vectorstore = create_or_load_vectorstore(embeddings)
    
    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=vectorstore.as_retriever()
    )
    
    prompt_template = """당신은 서울대학교병원 연구비 행정 챗봇입니다. 
    주어진 정보를 바탕으로 사용자의 질문에 친절하게 답변하세요.

    금액 관련 답변 시 다음 지침을 따르세요:
    1. 금액의 크기에 따라 적절한 단위(원, 만원, 억원)를 사용하세요.
    2. 1만원 미만: 원 단위로 표현 (예: 5,000원)
    3. 1만원 이상 1억원 미만: 만원 단위로 표현 (예: 50만원, 1,200만원)
    4. 1억원 이상: 억원 단위로 표현하되, 천만원 단위까지 표시 (예: 1억 2,000만원, 25억 3,000만원)
    5. 금액 뒤에 단위(원, 만원, 억원)를 명시적으로 붙이세요.
    6. 1,000 단위마다 쉼표(,)를 사용하여 구분하세요.
    7. 정확한 금액을 모를 경우, 대략적인 범위나 추정치를 제공하고 그렇게 한 이유를 설명하세요.

    예시:
    - 5,000원
    - 50만원
    - 1,200만원
    - 1억 2,000만원
    - 25억 3,000만원
    - 약 1,000만원에서 1,500만원 사이 (정확한 금액은 개별 상황에 따라 다를 수 있습니다)

    주어진 정보: {context}

    질문: {question}
    답변: """

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"],
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=local_llm,
        chain_type="stuff",
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
            "document_variable_name": "context"
        }
    )
    
    st.success("챗봇이 준비되었습니다. 질문해 주세요.")
    
    query = st.text_input("질문을 입력하세요:")
    if query:
        with st.spinner('답변을 생성 중입니다...'):
            result = qa_chain.invoke(query)
            answer = result['result']
            
            # 응답에서 "답변:" 이후의 텍스트만 추출
            clean_answer = answer.split("답변:")[-1].strip()
            
            st.write("답변:", clean_answer)
            
            st.subheader("참고 문서:")
            for doc in result['source_documents']:
                st.write(f"- {doc.metadata['source']}, Chunk: {doc.metadata['chunk']}")

if __name__ == "__main__":
    main()