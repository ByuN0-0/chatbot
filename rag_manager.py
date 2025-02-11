import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openaicall import LlmClient
from config import DATA_PATH, MODEL_NAME, TOP_K
from logger import logger

llmClient = LlmClient()


class RAGManager:
    def __init__(self, data_path: str = DATA_PATH,
                 model_name: str = MODEL_NAME):
        try:
            self.documents = self.load_dataset(data_path)
            logger.info("Dataset loaded successfully from %s", data_path)
        except Exception as e:
            logger.exception("Error loading dataset: %s", e)
            raise

        try:
            self.embedding_model = self.init_embedding_model(model_name)
            logger.info("Embedding model %s initialized.", model_name)
        except Exception as e:
            logger.exception("Error initializing embedding model: %s", e)
            raise

        try:
            self.index = self.create_faiss_index(self.documents, self.embedding_model)
            logger.info("FAISS index created successfully.")
        except Exception as e:
            logger.exception("Error creating FAISS index: %s", e)
            raise

    def load_dataset(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        return dataset["documents"]

    def init_embedding_model(self, model_name):
        return SentenceTransformer(model_name)

    def create_faiss_index(self, documents, embedding_model):
        try:
            doc_texts = [doc["content"] for doc in documents]
            doc_embeddings = embedding_model.encode(doc_texts)
            doc_embeddings = np.array(doc_embeddings).astype('float32')
            dimension = doc_embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(doc_embeddings)
            return index
        except Exception as e:
            logger.exception("Error during FAISS index creation: %s", e)
            raise

    def generate_response(self, user_message, top_k=TOP_K):
        try:
            # 1. 사용자 메시지 임베딩 생성
            query_embedding = self.embedding_model.encode([user_message])[0]
            query_embedding = np.array([query_embedding]).astype('float32')
        except Exception as e:
            logger.exception("Error encoding user message: %s", e)
            return "임베딩 생성 중 오류가 발생했습니다."

        try:
            # 2. FAISS를 이용해 관련 문서 검색
            distances, indices = self.index.search(query_embedding, k=top_k)
            # 예시: 각 문서의 title을 가져옴
            retrieved_docs = [self.documents[i]["title"] for i in indices[0]]
        except Exception as e:
            logger.exception("Error during FAISS search: %s", e)
            return "검색 중 오류가 발생했습니다."

        # 프롬프트 템플릿 구성
        system_prompt = (
            " - 사용자의 질문에 대해 답변을 해주세요.\n"
            " - 예시 문서를 기반으로 답변을 해주세요.\n"
            " - 예시 문서가 질문에 대해 연관이 없다면 사용하지 마세요.\n"
            " - 예시 문서를 기반으로 답변하되, 추가 부연 설명이나 코드 제공을 해도 좋습니다."
        )

        question_prompt = (
            f"- 사용자 질문:\n{user_message}\n\n"
            f"- 예시 문서:\n{', '.join(retrieved_docs)}"
        )

        try:
            response = llmClient.call_llm(system_prompt, question_prompt)
            logger.info("LLM response received.")
            # 최종 답변 추출 (객체의 속성으로 접근)
            return response.choices[0].message.content
        except Exception as e:
            logger.exception("Error during LLM call: %s", e)
            return "LLM 호출 중 오류가 발생했습니다."

    def deep_research(self, user_message: str, max_iterations: int = 3) -> str:
        conversation_history = f"User: {user_message}\n"
        system_prompt = (
            "당신은 전문 리서처입니다. 사용자 질문에 대해 깊이 있는 답변을 제공해야 합니다. "
            "만약 자신의 내부 지식만으로 답변하기 어렵거나 추가 정보가 필요하다고 판단되면, "
            "반드시 'ACTION: SEARCH <검색어>' 명령을 사용하여 필요한 추가 정보를 요청하세요. "
            "받은 검색 결과(Observation)를 반영하여 답변을 보완한 후, 최종 답변을 작성하십시오. "
            "최종 답변은 사용자 질문에 대해 명확하고 신뢰할 수 있는 정보를 제공해야 합니다."
        )

        final_response = ""
        for iteration in range(max_iterations):
            try:
                current_prompt = f"{system_prompt}\n{conversation_history}"
                llm_resp = llmClient.call_llm(system_prompt, current_prompt)
                response_text = llm_resp.choices[0].message.content.strip()
                logger.info("Iteration %d: LLM response: %s", iteration + 1, response_text)
            except Exception as e:
                logger.exception("LLM call error during deep research: %s", e)
                return "LLM 호출 중 오류가 발생했습니다."

            conversation_history += f"Assistant: {response_text}\n"

            if "ACTION: SEARCH" in response_text:
                try:
                    import re
                    match = re.search(r"ACTION:\s*SEARCH\s*(.*)", response_text)
                    if match:
                        search_query = match.group(1).strip()
                    else:
                        search_query = ""
                    logger.info("Parsed search query: '%s'", search_query)
                except Exception as e:
                    logger.exception("Error parsing ACTION: SEARCH command: %s", e)
                    search_query = ""

                if search_query:
                    try:
                        query_embedding = self.embedding_model.encode([search_query])[0]
                        query_embedding = np.array([query_embedding]).astype('float32')
                        distances, indices = self.index.search(query_embedding, k=TOP_K)
                        retrieved_docs = [self.documents[i]["content"] for i in indices[0]]
                        observation = f"Observation: {', '.join(retrieved_docs)}"
                        logger.info("Search observation: %s", observation)
                        conversation_history += observation + "\n"
                    except Exception as e:
                        logger.exception("Error during FAISS search in deep research: %s", e)
                        conversation_history += "Observation: 검색 중 오류 발생\n"
                else:
                    conversation_history += "Observation: 유효한 검색어가 추출되지 않음\n"
            else:
                final_response = response_text
                break

        logger.info("Final deep research response: %s", final_response)
        return final_response
