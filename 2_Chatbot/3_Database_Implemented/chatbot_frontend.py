import streamlit as st
from chatbot_backend import chatbot, retrieveAllThreads
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# ----------------------------------1. Configure and utility Set------------------------
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def add_new_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_new_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages',[])

def conversation_title(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    messages = state.values.get("messages", [])

    first_human = None
    for msg in messages:
        if msg.type == "human":
            first_human = msg
            break

    if first_human:
        first_60_chars = first_human.content[:60] + "...."
    else:
        first_60_chars = "Current Conversation"
    return first_60_chars

#---------------------------------- 2. Set session-------------------------------------------------
# list of all messages
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# list of all threads
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieveAllThreads()

# a thread id
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

add_new_thread(st.session_state['thread_id'])


# ************************************** Sidebar UI**************************************

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversation")

for thread_id in st.session_state['chat_threads'][::-1]:
    conversation_title_str = conversation_title(thread_id)

    # Unique key for each button
    if st.sidebar.button(conversation_title_str, key=f"btn_{thread_id}"):
        st.session_state['thread_id'] = thread_id

        messages = load_conversation(thread_id)
        temp_messages = []

        for msg in messages:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages


#------------------------------- 3. Fetch the conversation history-----------------------------------
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


#---------------------------------- 4. Inference----------------------------------------------
user_input = st.chat_input("Ask the question to AI")

if user_input:

    # save the user_input
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})

    # print the user_input
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {
        'configurable': {'thread_id': st.session_state['thread_id']},
        "metadata" : {
            'thread_id': st.session_state['thread_id']
        },
        "run_name" : "chat_turn"
    }

    # generate ai response
    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    # yield only assistant tokens
                    yield message_chunk.content


        ai_message = st.write_stream(ai_only_stream())
    # save the ai response
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

