from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings


def get_embedding_function():
    GOOGLE_API_KEY ="AIzaSyBix_tv-jF8TGPdE7heMusPblk8hzqTs48"
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=GOOGLE_API_KEY,
        model="models/embedding-001"
    )
    return embeddings
