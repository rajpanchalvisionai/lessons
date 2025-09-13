import argparse
import re
import os
import getpass
from typing import TypedDict, Annotated, Literal

# Corrected imports for Pydantic v2 and the @tool decorator
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from langchain_core.messages import ToolMessage
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

GOOGLE_API_KEY="AIzaSyDBttu_LeddM_M_sXiOTMEFMl-zaQUsC4Y"
# --- Configuration and Setup ---
# if "GOOGLE_API_KEY" not in os.environ:
#     os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API key: ")

CHROMA_PATH = "/home/administrator/Romaco/langchain/lessons/1. RAG/examples/pixegami/PDF_files_langchain/rag-tutorial-v2-main/chroma"

# --- 1. Define the Agent's State ---
class AgentState(TypedDict):
    question: str
    messages: Annotated[list, add_messages]
    initial_docs: str

# --- 2. Define Tools using the @tool Decorator ---

# Initialize dependencies globally to be used inside tools
embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
generation_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0, google_api_key=GOOGLE_API_KEY)

def clean_context(text: str) -> str:
    """Helper function to clean text, not a tool."""
    text = re.sub(r"System Name:.*?User Manual.*?\n+", "", text)
    text = re.sub(r"SPAN Inspection Systems Pvt\. Ltd\..*?spansystems\.in\s*", "", text)
    return text.strip()

class RetrieveInfoArgs(BaseModel):
    """Input schema for the retrieve_info tool."""
    query: str = Field(description="A specific, targeted search query to find more relevant information.")

@tool(args_schema=RetrieveInfoArgs)
def retrieve_info(query: str) -> str:
    """Use this tool to search the knowledge base for specific information if the initial documents are not sufficient."""
    print(f"---EXECUTING TOOL: Searching for '{query}'---")
    results = db.similarity_search_with_relevance_scores(query, k=5)
    if not results: return "No new information found."
    return "\n\n---\n\n".join([clean_context(doc.page_content) for doc, _ in results])

@tool
def generate_final_answer() -> str:
    """Use this tool when you have enough information from the retrieved documents to provide a complete, actionable solution to the user's question."""
    pass

@tool
def ask_user(question: str) -> str:
    """Use this tool ONLY if you have retrieved relevant documents but need a single, critical piece of information from the user to proceed."""
    print(f"---EXECUTING TOOL: ask_user---")
    return f"QUERY FOR USER: {question}"

@tool
def escalate() -> str:
    """Use this tool when you cannot find relevant information after searching and are unable to help the user."""
    print(f"---EXECUTING TOOL: Escalating---")
    return "Escalation: The agent was unable to resolve the issue with the available information."

# --- 3. Define the Graph Nodes ---

agent_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0, google_api_key=GOOGLE_API_KEY)

def initial_retrieval_node(state: AgentState) -> dict:
    print("---EXECUTING MANDATORY INITIAL RETRIEVAL---")
    query = state['question']
    results = db.similarity_search_with_relevance_scores(query, k=10)
    docs = "\n\n---\n\n".join([clean_context(doc.page_content) for doc, _ in results]) if results else "No information found."
    return {"initial_docs": docs}

def agent_node(state: AgentState) -> dict:
    print("---AGENT THINKING---")
    llm_with_tools = agent_llm.bind_tools([generate_final_answer, retrieve_info, ask_user, escalate])
    prompt = f"""You are a troubleshooting assistant. Your goal is to resolve the user's question using the provided documents.
Decide which tool to use next. Your primary goal is to use `generate_final_answer`. Only use other tools if absolutely necessary.

User's Question: {state['question']}
Initial Documents:
{state['initial_docs']}
Conversation History (previous tool calls):
{state['messages']}
"""
    response = llm_with_tools.invoke(prompt)
    return {"messages": [response]}

def generate_final_answer_node(state: AgentState) -> dict:
    print("---GENERATING FINAL ANSWER---")
    prompt = f"""You are a technical assistant. Based ONLY on the following context, provide a complete, chronological, numbered list of troubleshooting steps to answer the user's question.

**Question:** {state['question']}

**Context:**
{state['initial_docs']}

**Answer:**"""
    response = generation_llm.invoke(prompt)
    # The 'tool_call_id' is important for LangGraph to correctly attribute the result.
    return {"messages": [ToolMessage(content=response.content, tool_call_id="final_answer_tool_call")]}

# --- 4. Assemble the Graph ---

def create_agent_graph():
    tools = [retrieve_info, ask_user, escalate]
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("initial_retrieval", initial_retrieval_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("generate_final_answer", generate_final_answer_node)

    graph.set_entry_point("initial_retrieval")
    graph.add_edge("initial_retrieval", "agent")

    def router(state: AgentState) -> Literal["tools", "generate_final_answer", "__end__"]:
        if state["messages"][-1].tool_calls:
            if state["messages"][-1].tool_calls[0]['name'] == 'generate_final_answer':
                return "generate_final_answer"
            else:
                return "tools"
        return "__end__"

    graph.add_conditional_edges("agent", router)
    graph.add_edge("tools", "agent")
    graph.add_edge("generate_final_answer", END)

    return graph.compile()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The user's question.")
    args = parser.parse_args()

    agent_app = create_agent_graph()
    initial_state = {"question": args.query_text, "messages": []}

    print("---STARTING AGENT RUN---")

    # *** THIS IS THE FIX ***
    # Use .invoke() to run the graph and get the complete final state dictionary.
    final_state = agent_app.invoke(initial_state, {"recursion_limit": 15})

    # The final answer is now reliably in the 'messages' list of the final state.
    final_answer = final_state['messages'][-1].content

    print("\n---AGENT RUN COMPLETE---")
    print(f"\nFinal Answer:\n\n{final_answer.strip()}")


if __name__ == "__main__":
    main()