from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

# you must fill the openai api_key
openai.api_key = ""
model = SentenceTransformer('all-MiniLM-L6-v2')

# you must fill the pinecone api_key
pinecone.init(
    api_key="",
    environment="us-west4-gcp-free"
)
index = pinecone.Index('langchain-chatbot')


def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text']


def get_conversation_string():
    conversation_string = "Bot: " + st.session_state['responses'][0] + "\n"
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string


def get_prompt_template():
    response_valid_schema = ResponseSchema(name='is_valid',
                                           description='Is the user question/situation related to the context.',
                                           type='boolean')
    question_schema = ResponseSchema(name='question',
                                     description='Rewrite the given question/situation',
                                     type='string')
    details_schema = ResponseSchema(name='details',
                                    description='Answer about given question/situation',
                                    type='string')

    response_schemas = [response_valid_schema, question_schema, details_schema]

    main_template = '''\
       Answer the question as truthfully as possible using only the provided context, and if the answer is not contained within the context below, say "I don't know" and don't try to make up an answer.\n

       {format_instructions}

     '''

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()

    main_prompt = ChatPromptTemplate.from_template(template=main_template)

    system_template = SystemMessagePromptTemplate.from_template(
        template=main_prompt.format_messages(format_instructions=format_instructions)[0].content.replace('{',
                                                                                                         '</OPEN_CURLY_BRACKETS>').replace(
            '}', '</CLOSE_CURLY_BRACKETS>'))
    human_template = HumanMessagePromptTemplate.from_template(template="{input}")
    prompt_template = ChatPromptTemplate.from_messages(
        [system_template, MessagesPlaceholder(variable_name='history'), human_template])
    return prompt_template