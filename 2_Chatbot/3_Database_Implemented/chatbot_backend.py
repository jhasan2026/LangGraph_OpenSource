from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from llm_manager import get_llm_instance
import sqlite3

# 1. LLM
llm = get_llm_instance()

# 2. state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 3. Node Define
def chat_node(state: ChatState):
    message = state['messages']
    response = llm.invoke(message)
    return {'messages': [response]}

# 4. Graph Design
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

checkpointer_list = SqliteSaver(conn=conn)

chatbot = graph.compile(checkpointer= checkpointer_list)

def retrieveAllThreads():
    all_threads = set()
    for checkpointer in checkpointer_list.list(None):
        all_threads.add(checkpointer.config['configurable']['thread_id'])

    return list(all_threads)