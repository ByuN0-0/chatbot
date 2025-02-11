from openai import OpenAI

from config import API_KEY, LLM_MODEL
from logger import logger


class LlmClient:

    def __init__(self):
        try:
            self.client = OpenAI(api_key=API_KEY)
            self.model = LLM_MODEL
            logger.info("LLM client initialized successfully.")
        except Exception as e:
            logger.exception("Error initializing LLM client: %s", e)
            raise

    def call_llm(self, system_prompt: str, user_prompt: str):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                seed=42
            )
            logger.info("LLM API call successful.")
            return response
        except Exception as e:
            logger.exception("Error during LLM API call: %s", e)
            raise
