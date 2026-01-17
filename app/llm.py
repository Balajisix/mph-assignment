from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os

load_dotenv()

def get_llm():
    model_id = os.getenv("HF_MODEL")
    access_token = os.getenv("HF_TOKEN")

    llm = HuggingFaceEndpoint(
        repo_id=model_id,
        huggingfacehub_api_token=access_token,
        # task="text-generation",
        temperature=0.1,
        max_new_tokens=512,
    )

    model = ChatHuggingFace(llm=llm)

    return model