import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/chat"

st.title("서울대학교병원 연구비 행정 챗봇")

# 사용자 입력
user_input = st.text_input("질문을 입력하세요:")

if user_input:
    with st.spinner('답변을 생성 중입니다...'):
        try:
            response = requests.post(API_URL, json={"question": user_input})
            result = response.json()
            
            st.write("답변:", result['answer'])
            
            st.subheader("참고 문서:")
            for doc in result['source_documents']:
                st.write(f"- {doc['source']}, Chunk: {doc['chunk']}")
        except Exception as e:
            st.error(f"오류 발생: {e}")
            st.error("다시 질문해 주세요.")