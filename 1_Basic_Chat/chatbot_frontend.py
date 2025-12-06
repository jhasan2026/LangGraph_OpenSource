import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import HumanMessage

# 1. Configure Set
CONFIG = {"configurable": {
    "thread_id": 'thread-1'
}}

# 2. Set session
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []


# st.session_state['message_history'].append({'role': 'assistant', 'content': "This is the ai message"})
# st.session_state['message_history'].append({'role': 'user', 'content': "This is the user message"})

# 3. Fetch the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input("Ask the question to AI")

if user_input:
    # save the user_input
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})

    # print the user_input
    with st.chat_message('user'):
        st.text(user_input)

    # generate ai response
    response = chatbot.invoke(
        {'messages': [HumanMessage(content=user_input)]},
        config=CONFIG
    )
    ai_message = response['messages'][-1].content


    # save the ai response
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

    # print the ai response
    with st.chat_message('assistant'):
        st.text(ai_message)
