# 대화형 검색 기반 Q&A 시스템
이 프로젝트는 Flask 기반 웹 애플리케이션에서 SentenceTransformer와 FAISS를 활용해 문서를 임베딩 및 유사도 검색하고, ReAct 및 RAG 기법을 통해 4o mini LLM을 사용하여 자연어 답변을 생성하는 시스템이다.

## 주요 기술
- ### Flask
  웹 애플리케이션 프레임워크

- ### SentenceTransformer & FAISS
  문서 임베딩 생성 및 유사도 기반 검색

- ### ReAct & RAG
  검색된 문서를 활용해 LLM(4o mini)으로 자연어 응답 생성

## 기능
- ### 문서 임베딩 및 검색
  문서 데이터를 임베딩한 후 FAISS 인덱스를 구축하여, 사용자 질문과 유사한 문서를 효율적으로 검색한다.

- ### 자연어 응답 생성
  검색된 문서를 기반으로 LLM(4o mini)을 호출하여, ReAct 및 RAG 기법으로 답변을 생성한다.

- ### 심층 연구 (Deep Research)
  동일 질문에 대해 여러 번 병렬 LLM 호출을 통해 다양한 응답을 취합하고, 최종 답변을 정제하는 다단계 대화형 응답 생성 과정을 구현한다.
