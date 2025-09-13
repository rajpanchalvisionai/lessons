import argparse
import re
import os
import getpass
import redis
import json # Added for JSON parsing
import uuid # Added for generating tool call IDs
from typing import TypedDict, Annotated, Literal, Optional

# Corrected imports for Pydantic v2 and the @tool decorator
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from langchain_core.messages import ToolMessage, AIMessage, SystemMessage, HumanMessage # Added HumanMessage for history rendering
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.checkpoint.redis import RedisSaver
from langchain_community.llms import VLLM




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
generation_llm = VLLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    vllm_kwargs={
        "gpu_memory_utilization": 0.85,
        "max_model_len": 100_000,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 1024,
        "cpu_offload_gb": 4,
        "enable_chunked_prefill": True,
    },
)

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

# All tools that the VLLM is allowed to "call" via JSON output
ALL_LLM_CALLABLE_TOOLS = [retrieve_info, ask_user, generate_final_answer, escalate]

def format_tools_for_vllm(tools):
    """Formats the tools into a string representation for the VLLM prompt."""
    tool_descriptions = []
    for tool_func in tools:
        schema = tool_func.args_schema.schema() if tool_func.args_schema else {}
        name = tool_func.name
        description = tool_func.__doc__.strip() if tool_func.__doc__ else "No description available."
        
        # Format parameters for clarity
        params_list = []
        for param_name, param_details in schema.get('properties', {}).items():
            param_type = param_details.get('type', 'string')
            param_desc = param_details.get('description', '')
            params_list.append(f"    - {param_name} ({param_type}): {param_desc}")
        
        params_str = "\n".join(params_list) if params_list else "    - None"
        
        tool_descriptions.append(f"Tool Name: {name}\nDescription: {description}\nParameters:\n{params_str}\n")
    
    return "\n".join(tool_descriptions)

def parse_vllm_tool_output(text: str):
    """
    Parses the VLLM's text output to extract tool calls.
    Assumes the VLLM will output a JSON object like:
    {"name": "tool_name", "arguments": {"arg1": "value1"}}
    """
    try:
        # Try to parse as JSON. If it's a tool call, it should be JSON.
        data = json.loads(text.strip())
        if isinstance(data, dict) and "name" in data and "arguments" in data:
            tool_name = data["name"]
            tool_args = data["arguments"]
            return {"name": tool_name, "arguments": tool_args}
    except json.JSONDecodeError:
        # Not a JSON, so treat it as direct text or invalid output
        pass
    return None

# --- 3. Define the Graph Nodes ---

def initial_retrieval_node(state: AgentState) -> dict:
    print("---EXECUTING MANDATORY INITIAL RETRIEVAL---")
    print(f"Initial retrieval query: '{state['question']}'")
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
    
    print("\n---Full Initial Documents (passed to agent)---")
    print(docs)
    print("---------------------------------------------\n")

    return {"initial_docs": docs}

def agent_node(state: AgentState) -> dict:
    print("---AGENT THINKING (VLLM)---")
    
    valid_messages_for_llm = [m for m in state['messages'] if not isinstance(m, SystemMessage)]
    
    history_str = ""
    for msg in valid_messages_for_llm:
        if isinstance(msg, HumanMessage):
            history_str += f"\nHuman: {msg.content}"
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    # FIX: Access ToolCall attributes directly
                    history_str += f"\nAI requested tool '{tc.name}' with args {json.dumps(tc.args)}"
            else:
                history_str += f"\nAI: {msg.content}"
        elif isinstance(msg, ToolMessage):
            history_str += f"\nTool Output for '{msg.name}': {msg.content}"

    formatted_tools = format_tools_for_vllm(ALL_LLM_CALLABLE_TOOLS)

    prompt_template = f"""You are a troubleshooting assistant. Your goal is to resolve the user's question using the provided documents and tools.
You MUST respond with a JSON object representing a tool call. Do not include any other text or explanation outside the JSON.

Available Tools:
{formatted_tools}

Example JSON output for retrieve_info:
{{ "name": "retrieve_info", "arguments": {{ "query": "specific information needed" }} }}

Example JSON output for generate_final_answer:
{{ "name": "generate_final_answer", "arguments": {{}} }}

Example JSON output for ask_user:
{{ "name": "ask_user", "arguments": {{ "question": "What is the model number?" }} }}

Example JSON output for escalate:
{{ "name": "escalate", "arguments": {{}} }}

User's Question: {state['question']}

Initial Documents:
{state['initial_docs']}

Conversation History:
{history_str}

Your Response (MUST be a JSON tool call):
"""

    raw_response = generation_llm.invoke(prompt_template)
    print(f"---VLLM Raw Response: {raw_response}---") 

    parsed_tool_call = parse_vllm_tool_output(raw_response)

    if parsed_tool_call:
        tool_name = parsed_tool_call['name']
        tool_arguments = parsed_tool_call['arguments']

        ai_message = AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [{
                    "id": f"call_{uuid.uuid4().hex}",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_arguments)
                    },
                    "type": "function"
                }]
            }
        )
        return {"messages": [ai_message]}
    else:
        print("---VLLM failed to output a valid tool call JSON. Forcing escalation.---")
        ai_message = AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [{
                    "id": f"call_{uuid.uuid4().hex}",
                    "function": {
                        "name": "escalate",
                        "arguments": json.dumps({})
                    },
                    "type": "function"
                }]
            }
        )
        return {"messages": [ai_message]}

def generate_final_answer_node(state: AgentState) -> dict:
    print("---GENERATING FINAL ANSWER---")
    tool_call_id = None
    if state['messages'] and state['messages'][-1].tool_calls:
        for tc in state['messages'][-1].tool_calls:
            if tc.name == 'generate_final_answer': # FIX: Access ToolCall attribute
                tool_call_id = tc.id # FIX: Access ToolCall attribute
                break

    prompt = f"""You are a technical assistant. Based ONLY on the following context, provide a complete, chronological, numbered list of troubleshooting steps to answer the user's question.

**Question:** {state['question']}
**Context:** {state['initial_docs']}
**Answer:**"""
    response = generation_llm.invoke(prompt)
    return {"messages": [ToolMessage(content=response, tool_call_id=tool_call_id)]}

def summarize_node(state: AgentState) -> dict:
    """NEW: Node to generate a summary and condense the history."""
    print("---GENERATING SUMMARY AND CONDENSING HISTORY---")
    history_for_summary = [m for m in state['messages'] if not isinstance(m, SystemMessage)]

    prompt = f"""Concisely summarize the following troubleshooting interaction. Capture the initial user question, the steps taken by the agent, and the final resolution.

Initial Question: {state['question']}
Conversation History:
{history_for_summary}
Summary:"""
    summary_response = generation_llm.invoke(prompt)
    
    condensed_history_message = SystemMessage(
        content="The preceding conversation has been summarized to conserve context space. "
                "The full history is preserved in the system logs.\n\n"
                f"**Summary of Past Interaction:**\n{summary_response}"
    )
    
    return {
        "summary": summary_response,
        "messages": [condensed_history_message]
    }

def escalate_and_summarize_node(state: AgentState) -> dict:
    """Node to handle escalation with a summary."""
    print("---ESCALATING AND GENERATING SUMMARY---")
    tool_call_id = None
    if state['messages'] and state['messages'][-1].tool_calls:
        for tc in state['messages'][-1].tool_calls:
            if tc.name == 'escalate': # FIX: Access ToolCall attribute
                tool_call_id = tc.id # FIX: Access ToolCall attribute
                break
    
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
        f"{summary_response}"
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
        last_message = state["messages"][-1]
        
        if last_message.tool_calls:
            tool_name = last_message.tool_calls[0].name # FIX: Access ToolCall attribute
            if tool_name == 'generate_final_answer': return "generate_final_answer"
            if tool_name == 'escalate': return "escalate_and_summarize"
            return "tools"
        else:
            print("---Router: Last message has no tool calls. This shouldn't happen with the revised agent_node. Ending graph.---")
            return "__end__"

    def check_for_summary(state: AgentState) -> Literal["summarize", "__end__"]:
        conversation_messages = [m for m in state['messages'] if not isinstance(m, SystemMessage)]
        if len(conversation_messages) > MESSAGE_THRESHOLD:
            return "summarize"
        return END

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
        redis_client.ping()
        print("---Successfully connected to Redis---")
    except redis.exceptions.ConnectionError as e:
        print(f"---Redis connection error: {e}---")
        print("---Please ensure Redis is running on redis://localhost:6379---")
        return

    with RedisSaver.from_conn_string("redis://localhost:6379") as redis_saver:
        agent_app = create_agent_graph(checkpointer=redis_saver)

        config = {"configurable": {"thread_id": args.thread_id}}
        
        try:
            current_state = agent_app.get_state(config)
            print(f"---Resuming conversation for thread ID: {args.thread_id}---")
            initial_messages = current_state.values['messages'] + [HumanMessage(content=args.query_text)]
            initial_state = {"question": args.query_text, "messages": initial_messages}
        except Exception as e:
            print(f"---Starting new conversation for thread ID: {args.thread_id} (No existing state found or error: {e})---")
            initial_state = {"question": args.query_text, "messages": [HumanMessage(content=args.query_text)]}


        print(f"---STARTING AGENT RUN FOR THREAD: {args.thread_id}---")

        for event in agent_app.stream(initial_state, config, stream_mode="values"):
            pass

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