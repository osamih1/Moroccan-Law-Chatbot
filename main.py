from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *


st.subheader("Chatbot with Langchain, Chatgpt, Pinecone, and Streamlit")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]
if 'requests' not in st.session_state:
    st.session_state['requests'] = []
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=2, return_messages=True)

# you must fill the openai api_key
llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                 openai_api_key="",
                 temperature=0.0)

system_message_template = SystemMessagePromptTemplate.from_template(
    template="""Answer the question as truthfully as possible using only the provided context, and if the answer is not contained within the context below, say "I don't know" and don't try to make up an answer."""
)

human_message_template = HumanMessagePromptTemplate.from_template(
    template="{input}"
)

prompt_template = ChatPromptTemplate.from_messages([
    system_message_template,
    MessagesPlaceholder(variable_name="history"),
    human_message_template
])

conversation = ConversationChain(
    memory=st.session_state.buffer_memory,
    prompt=get_prompt_template(),
    llm=llm,
    verbose=True
)


# container for chat history
response_container = st.container()

# container for text box
textcontainer = st.container()


with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("Typing..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            refined_query = query_refiner(conversation_string, query)
            st.subheader("Refined query:")
            st.write(refined_query)
            context = find_match(query)
            # print(context)
            response = conversation.predict(input=f"\nContext:\n\"{context}\"\n\nQuestion:\n\"{query}\"")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state['requests'][i], is_user=True, key=str(i)+'_user')