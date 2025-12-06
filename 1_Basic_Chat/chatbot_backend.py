from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from llm_manager import get_llm_instance

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

checkpointer = InMemorySaver()

chatbot = graph.compile(checkpointer= checkpointer)

