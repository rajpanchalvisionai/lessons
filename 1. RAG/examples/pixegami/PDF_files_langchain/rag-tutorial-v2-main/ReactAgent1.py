import argparse
import re
import os
import getpass
from typing import TypedDict, Annotated, Literal, Optional

# Corrected imports for Pydantic v2 and the @tool decorator
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from langchain_core.messages import ToolMessage, AIMessage
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
# User-specified threshold for triggering summarization
MESSAGE_THRESHOLD = 6

# --- 1. Define the Agent's State ---
class AgentState(TypedDict):
    question: str
    messages: Annotated[list, add_messages]
    initial_docs: str
    summary: Optional[str] # Added to store the final summary

# --- 2. Define Tools using the @tool Decorator ---

# Initialize dependencies globally to be used inside tools
embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
generation_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0, google_api_key=GOOGLE_API_KEY)

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
    """Use this tool when you cannot find relevant information after searching and must escalate to a human expert."""
    # This tool's implementation is now handled by a dedicated graph node
    # to allow for summarization before ending the process.
    pass

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
    # Filter out any placeholder messages without tool calls before sending to the LLM
    valid_messages = [m for m in state['messages'] if isinstance(m, AIMessage) and m.tool_calls]
    
    llm_with_tools = agent_llm.bind_tools([generate_final_answer, retrieve_info, ask_user, escalate])
    prompt = f"""You are a troubleshooting assistant. Your goal is to resolve the user's question using the provided documents.
Decide which tool to use next. Your primary goal is to use `generate_final_answer`. Only use other tools if absolutely necessary.
If you have tried to retrieve information and are still unable to answer, use the `escalate` tool.

User's Question: {state['question']}
Initial Documents:
{state['initial_docs']}
Conversation History (previous tool calls):
{valid_messages}
"""
    response = llm_with_tools.invoke(prompt)
    return {"messages": [response]}

def generate_final_answer_node(state: AgentState) -> dict:
    print("---GENERATING FINAL ANSWER---")
    # Find the tool call ID for the final answer to maintain graph continuity
    tool_call_id = next((tc['id'] for tc in state['messages'][-1].tool_calls if tc['name'] == 'generate_final_answer'), None)

    prompt = f"""You are a technical assistant. Based ONLY on the following context, provide a complete, chronological, numbered list of troubleshooting steps to answer the user's question.

**Question:** {state['question']}

**Context:**
{state['initial_docs']}

**Answer:**"""
    response = generation_llm.invoke(prompt)
    return {"messages": [ToolMessage(content=response.content, tool_call_id=tool_call_id)]}

def summarize_node(state: AgentState) -> dict:
    """NEW: Node to generate a summary of the interaction."""
    print("---GENERATING SUMMARY---")
    prompt = f"""Concisely summarize the following troubleshooting interaction. Capture the initial user question, the steps taken by the agent, and the final resolution.

Initial Question: {state['question']}
Conversation History:
{state['messages']}

Summary:"""
    response = generation_llm.invoke(prompt)
    return {"summary": response.content}

def escalate_and_summarize_node(state: AgentState) -> dict:
    """NEW: Node to handle escalation with a summary."""
    print("---ESCALATING AND GENERATING SUMMARY---")
    tool_call_id = next((tc['id'] for tc in state['messages'][-1].tool_calls if tc['name'] == 'escalate'), None)

    summary_prompt = f"""The troubleshooting agent could not resolve the issue and is escalating. Please summarize the user's initial question and the steps the agent took.

Initial Question: {state['question']}
Conversation History:
{state['messages']}

Summary for Escalation Ticket:"""
    summary_response = generation_llm.invoke(summary_prompt)
    
    escalation_message = (
        "**Escalation Required**\n\n"
        "The agent was unable to resolve the issue with the available information and tools.\n\n"
        "**Troubleshooting Summary:**\n"
        f"{summary_response.content}"
    )
    return {"messages": [ToolMessage(content=escalation_message, tool_call_id=tool_call_id)]}


# --- 4. Assemble the Graph ---

def create_agent_graph():
    tools = [retrieve_info, ask_user] # Excludes escalate and generate_final_answer as they are handled by dedicated nodes
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("initial_retrieval", initial_retrieval_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("generate_final_answer", generate_final_answer_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("escalate_and_summarize", escalate_and_summarize_node)


    graph.set_entry_point("initial_retrieval")
    graph.add_edge("initial_retrieval", "agent")

    def router(state: AgentState) -> Literal["tools", "generate_final_answer", "escalate_and_summarize", "__end__"]:
        """This router directs the workflow based on the agent's most recent decision."""
        if state["messages"][-1].tool_calls:
            tool_name = state["messages"][-1].tool_calls[0]['name']
            if tool_name == 'generate_final_answer':
                return "generate_final_answer"
            elif tool_name == 'escalate':
                return "escalate_and_summarize"
            else:
                return "tools"
        return "__end__"

    def check_for_summary(state: AgentState) -> Literal["summarize", "__end__"]:
        """NEW: After providing an answer, check if a summary is needed."""
        if len(state['messages']) > MESSAGE_THRESHOLD:
            return "summarize"
        return "__end__"

    graph.add_conditional_edges("agent", router)
    graph.add_edge("tools", "agent")
    
    # NEW: After generating an answer, decide whether to summarize or end
    graph.add_conditional_edges("generate_final_answer", check_for_summary)
    
    graph.add_edge("summarize", END)
    graph.add_edge("escalate_and_summarize", END)

    return graph.compile()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The user's question.")
    args = parser.parse_args()

    agent_app = create_agent_graph()
    # Add an empty summary field to the initial state
    initial_state = {"question": args.query_text, "messages": [], "summary": None}

    print("---STARTING AGENT RUN---")

    final_state = agent_app.invoke(initial_state, {"recursion_limit": 15})

    # The final answer is in the last ToolMessage in the 'messages' list.
    final_answer_message = next((m for m in reversed(final_state['messages']) if isinstance(m, ToolMessage)), None)
    final_answer = final_answer_message.content if final_answer_message else "No final answer was generated."

    print("\n---AGENT RUN COMPLETE---")
    print(f"\nFinal Answer:\n\n{final_answer.strip()}")
    
    # NEW: Check if a summary was created and print it
    if final_state.get('summary'):
        print("\n---Conversation Summary---")
        print(final_state['summary'].strip())


if __name__ == "__main__":
    main()