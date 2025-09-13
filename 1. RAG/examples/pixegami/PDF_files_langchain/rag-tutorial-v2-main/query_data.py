import argparse
import re
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

CHROMA_PATH = "/home/administrator/Romaco/langchain/lessons/1. RAG/examples/pixegami/PDF_files_langchain/rag-tutorial-v2-main/chromadb"
GOOGLE_API_KEY ="AIzaSyDBttu_LeddM_M_sXiOTMEFMl-zaQUsC4Y"   

PROMPT_TEMPLATE = """
You are a technical assistant. You are answering a question based only on the following context.

Instructions:
- Do NOT ask for more context. You already have all the context you need below.
- Do NOT respond with anything other than the answer.
- Do NOT include phrases like "Based on the context", "According to the document", etc.
- Be specific, concise, and professional.
- If tools, buttons, or modes are mentioned, explain their purpose and when to use them.
- If a process is involved, present it in clear **chronological** numbered steps.
- Make sure to include all relevant details from the context.

Here is your task:

---
**Question:** {question}

**Context:**
{context}

---

Now answer the question.
---
**Answer:**
"""


def main():
    # Parse CLI argument
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Load embedding function
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    # Load Chroma DB
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search for relevant documents
    results = db.similarity_search_with_relevance_scores(query_text, k=10)
    if len(results) == 0:
        print(" No relevant chunks retrieved from ChromaDB.")
        return

    with open("retrieved_chunks.txt", "w", encoding="utf-8") as f:
        for i, (doc, score) in enumerate(results):
            f.write(f"[Chunk {i+1}] (Score: {score:.2f})\n")
            f.write(doc.page_content + "\n" )
            f.write("-" * 80 + "\n")

    def clean_context(text):
        text = re.sub(r"System Name:.*?User Manual.*?\n+", "", text)
        text = re.sub(r"SPAN Inspection Systems Pvt\. Ltd\..*?spansystems\.in\s*", "", text)
        return text.strip()

    context_text = "\n\n---\n\n".join([clean_context(doc.page_content) for doc, _ in results])

    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)

    # Use Gemini for LLM response
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key="AIzaSyDBttu_LeddM_M_sXiOTMEFMl-zaQUsC4Y",
        temperature=0.1,
    )
    response_text = model.invoke(prompt)

    # Clean display: only the final formatted output
    print(f"\nQuestion: {query_text}")
    # print(f"\n Response:\n\n{response_text.content.strip()}\n")
    print(f"\n The Response is:\n\n{response_text.content}")

if __name__ == "__main__":
    main()
