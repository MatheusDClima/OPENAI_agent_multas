import os
from dotenv import load_dotenv
import streamlit as st

from decouple import config

from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit  import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
# from langchain.chat_models import ChatOpenAI


load_dotenv()  # carrega as variáveis do .env automaticamente

# Agora a chave já está no ambiente, basta pegar
openai_api_key = os.getenv("OPENAI_API_KEY")


st.set_page_config(
    page_title='Autuações GPT',
    page_icon='img/logo.png',
)
st.header('Assistente Autuações Pioneira 🤖')

model_options = [
    'gpt-3.5-turbo',
    'gpt-4',
    'gpt-4-turbo',
    'gpt-4o-mini',
    'gpt-4o'
]

st.sidebar.image("img/logo.png", width=100)

selected_model = st.sidebar.selectbox(
    label = 'Selecione o modelo LLM',
    options = model_options
)

st.sidebar.markdown('### Sobre')
st.sidebar.markdown('Este agente consulta o banco de dados das Autuações utilizando um modelo GPT.')

st.write('Peça um relatório completo ou faça perguntas sobre as multas, como valor mensal, anual ou em relação ao ano anterior.')
user_question = st.text_input('O que deseja saber?')

# CRIAÇÃO DO AGENTE
model = ChatOpenAI(
    model = selected_model,
    openai_api_key = openai_api_key,
    max_retries=5,   # tenta novamente se der RateLimit ou erro de rede
    temperature=0    # opcional: mais previsível p/ queries SQL
)

# CONEXÃO COM O BANCO
db = SQLDatabase.from_uri('sqlite:///autuacoes.db')
toolkit = SQLDatabaseToolkit(
    db = db,
    llm = model
)
system_message = hub.pull('hwchase17/react')

# CRIANDO AGENTE
agent = create_react_agent(
    llm = model,
    tools = toolkit.get_tools(),
    prompt = system_message
)

# CRIANDO AGENTE EXECUTOR
agent_executor = AgentExecutor(
    agent = agent,
    tools = toolkit.get_tools(),
    verbose = True
)

# PROMPT
prompt = '''
    Você é um assistente especializado em autuações da Viação Pioneira.
    Use as ferramentas necessárias para responder perguntas sobre multas.
    Forneça insights sobre quantidade, valores, comparação com ano/mês anterior
    e relatórios conforme solicitado. Sempre responda em português brasileiro.
    resposta: {q}
'''
prompt_template = PromptTemplate.from_template(prompt)

# CONDIÇÃO DO BOTÃO
if st.button('Consultar'):
    if user_question:
        with st.spinner('Consultando o Banco de Dados...'):
            formatted_prompt = prompt_template.format(q=user_question)
            output = agent_executor.invoke({'input': formatted_prompt})
            resposta = output.get("output", "Não foi possível gerar uma resposta.")
            st.markdown(resposta)
    else:
        st.warning('Por favor, insira uma pergunta.')


memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,  # mantém como lista de mensagens
    k=5  # mantém apenas as últimas 5 mensagens
)
