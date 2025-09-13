import argparse
import re
import os
import getpass
import redis
from typing import TypedDict, Annotated, Literal, Optional

# Corrected imports for Pydantic v2 and the @tool decorator
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from langchain_core.messages import ToolMessage, AIMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.checkpoint.redis import RedisSaver

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
    summary: Optional[str]

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
    results = db.similarity_search_with_relevance_scores(query, k=10)
    file_path = "/home/administrator/Romaco/langchain/lessons/1. RAG/examples/pixegami/PDF_files_langchain/rag-tutorial-v2-main/retrieved.txt"
    print(f"---Writing results of retrieve_info tool to {file_path} (this will overwrite previous content)---")
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    if not results:
        with open(file_path, "w") as f:
            f.write("No new information found.\n")  # Always create/overwrite the file
        return "No new information found."
    retrieved_text = "\n\n---\n\n".join([clean_context(doc.page_content) for doc, _ in results])
    with open(file_path, "w") as f:
        f.write(retrieved_text)
    return retrieved_text

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
    pass

# --- 3. Define the Graph Nodes ---

agent_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0, google_api_key=GOOGLE_API_KEY)

def initial_retrieval_node(state: AgentState) -> dict:
    print("---EXECUTING MANDATORY INITIAL RETRIEVAL---")
    print(f"Initial retrieval query: '{state['question']}'") # Print query explicitly
    results = db.similarity_search_with_relevance_scores(state['question'], k=20)
    print(f"Initial retrieval results count: {len(results)}")
    if not results:
        print("No results found during initial retrieval.")
        docs = "No information found."
    else:
        for i, (doc, score) in enumerate(results):
            print(f"  Chunk {i+1}: Score={score:.2f}, Content (first 100 chars): {doc.page_content[:100]}")
        docs = "\n\n---\n\n".join([clean_context(doc.page_content) for doc, _ in results])
    
    file_path = "/home/administrator/Romaco/langchain/lessons/1. RAG/examples/pixegami/PDF_files_langchain/rag-tutorial-v2-main/retrieved.txt"
    print(f"---Writing initial retrieval results to {file_path} (this will be overwritten by retrieve_info if called)---")
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, "w") as f:
        f.write(docs)
    
    # *** IMPORTANT DEBUGGING STEP ***
    # Print the full initial_docs content here to verify what's being passed to the agent
    print("\n---Full Initial Documents (passed to agent)---")
    print(docs)
    print("---------------------------------------------\n")

    return {"initial_docs": docs}

def agent_node(state: AgentState) -> dict:
    print("---AGENT THINKING---")
    # Filter out any SystemMessages which are used for internal state management (like summary)
    # The LLM doesn't need to see the SystemMessage that replaces history.
    valid_messages = [m for m in state['messages'] if not isinstance(m, SystemMessage)]
    
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
    tool_call_id = next((tc['id'] for tc in state['messages'][-1].tool_calls if tc['name'] == 'generate_final_answer'), None)

    prompt = f"""You are a technical assistant. Based ONLY on the following context, provide a complete, chronological, numbered list of troubleshooting steps to answer the user's question.

**Question:** {state['question']}
**Context:** {state['initial_docs']}
**Answer:**"""
    response = generation_llm.invoke(prompt)
    return {"messages": [ToolMessage(content=response.content, tool_call_id=tool_call_id)]}

def summarize_node(state: AgentState) -> dict:
    """NEW: Node to generate a summary and condense the history."""
    print("---GENERATING SUMMARY AND CONDENSING HISTORY---")
    # Exclude SystemMessages (like prior summaries) from the history sent for summarization
    history_for_summary = [m for m in state['messages'] if not isinstance(m, SystemMessage)]

    prompt = f"""Concisely summarize the following troubleshooting interaction. Capture the initial user question, the steps taken by the agent, and the final resolution.

Initial Question: {state['question']}
Conversation History:
{history_for_summary}
Summary:"""
    summary_response = generation_llm.invoke(prompt)
    
    # Create a condensed history message
    condensed_history_message = SystemMessage(
        content="The preceding conversation has been summarized to conserve context space. "
                "The full history is preserved in the system logs.\n\n"
                f"**Summary of Past Interaction:**\n{summary_response.content}"
    )
    
    # The new message list contains only the summary message
    # The full history is saved in Redis by the checkpointer
    return {
        "summary": summary_response.content,
        "messages": [condensed_history_message]
    }

def escalate_and_summarize_node(state: AgentState) -> dict:
    """Node to handle escalation with a summary."""
    print("---ESCALATING AND GENERATING SUMMARY---")
    tool_call_id = next((tc['id'] for tc in state['messages'][-1].tool_calls if tc['name'] == 'escalate'), None)
    
    # Exclude SystemMessages from the history sent for summarization
    history_for_summary = [m for m in state['messages'] if not isinstance(m, SystemMessage)]

    summary_prompt = f"""The troubleshooting agent could not resolve the issue and is escalating. Please summarize the user's initial question and the steps the agent took.

Initial Question: {state['question']}
Conversation History:
{history_for_summary}
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

def create_agent_graph(checkpointer):
    """Factory function to create the agent graph with a checkpointer."""
    tools = [retrieve_info, ask_user]
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
        if state["messages"] and state["messages"][-1].tool_calls:
            tool_name = state["messages"][-1].tool_calls[0]['name']
            if tool_name == 'generate_final_answer': return "generate_final_answer"
            if tool_name == 'escalate': return "escalate_and_summarize"
            return "tools"
        # If no tool call, it means the agent likely generated a direct response, so we end.
        return "__end__"

    def check_for_summary(state: AgentState) -> Literal["summarize", "__end__"]:
        # Count only actual conversation turns (AIMessage, HumanMessage, ToolMessage)
        # Exclude SystemMessages which are internal state.
        conversation_messages = [m for m in state['messages'] if not isinstance(m, SystemMessage)]
        if len(conversation_messages) > MESSAGE_THRESHOLD:
            return "summarize"
        return "__end__"

    graph.add_conditional_edges("agent", router)
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("generate_final_answer", check_for_summary)
    graph.add_edge("summarize", END)
    graph.add_edge("escalate_and_summarize", END)

    return graph.compile(checkpointer=checkpointer)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The user's question.")
    parser.add_argument("--thread-id", type=str, help="The unique ID for the conversation thread.", required=True)
    args = parser.parse_args()

    # Set up Redis connection and checkpointer
    try:
        redis_client = redis.from_url("redis://localhost:6379", db=0)
        redis_client.ping() # Check connection
        print("---Successfully connected to Redis---")
    except redis.exceptions.ConnectionError as e:
        print(f"---Redis connection error: {e}---")
        print("---Please ensure Redis is running on redis://localhost:6379---")
        return

    # Use RedisSaver as a context manager for proper cleanup
    with RedisSaver.from_conn_string("redis://localhost:6379") as redis_saver:
        # No need for redis_saver.setup() with from_conn_string context manager
        agent_app = create_agent_graph(checkpointer=redis_saver)

        # Configuration for the specific conversation thread
        config = {"configurable": {"thread_id": args.thread_id}}
        
        # Attempt to get the existing state for the thread, if it exists
        try:
            current_state = agent_app.get_state(config)
            print(f"---Resuming conversation for thread ID: {args.thread_id}---")
            # When resuming, the new question should be added to messages as a HumanMessage
            initial_messages = current_state.values['messages'] + [("user", args.query_text)]
            # LangGraph uses 'question' for the initial entry point, but then relies on 'messages' for history.
            # So, we populate 'messages' with the new user query.
            # The 'initial_docs' might need re-retrieval if the question has significantly changed.
            # For simplicity, we'll let initial_retrieval_node run again.
            initial_state = {"question": args.query_text, "messages": initial_messages}
        except Exception as e:
            print(f"---Starting new conversation for thread ID: {args.thread_id} (No existing state found or error: {e})---")
            initial_state = {"question": args.query_text, "messages": []}


        print(f"---STARTING AGENT RUN FOR THREAD: {args.thread_id}---")

        # Use .stream() to execute the graph with the checkpointer
        for event in agent_app.stream(initial_state, config, stream_mode="values"):
            # The stream method yields the current state after each step.
            # We can simply consume the stream to drive the graph to completion.
            pass

        # After the run, get the final state from the checkpointer
        final_state = agent_app.get_state(config)
        
        final_answer_message = next((m for m in reversed(final_state.values['messages']) if isinstance(m, ToolMessage)), None)
        final_answer = final_answer_message.content if final_answer_message else "No final answer was generated."

        print("\n---AGENT RUN COMPLETE---")
        print(f"\nFinal Answer:\n\n{final_answer.strip()}")
        
        if final_state.values.get('summary'):
            print("\n---Conversation Summary---")
            print(final_state.values['summary'].strip())

if __name__ == "__main__":
    main()