import streamlit as st
import openai
from langchain_community.document_loaders import WikipediaLoader
import os
from dotenv import load_dotenv
import nltk
from langdetect import detect
from query_data import query_data

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# agent = ConversationalAgent2()

# Function to interact with the GPT-3.5-turbo model with tunable parameters

# Streamlit app header and title
# tattooed geek logo
logo1 = 'https://demo.curriculum.gov.bd/img/logo.png'
wiki_logo = "data:image/jpeg;base64,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"

# Streamlit app header and title
st.set_page_config(page_title="Research Lab | Curriculam Portal", page_icon=logo1 , layout="wide")

st.write("# Research Lab :abacus: ")

# Show all available models
st.sidebar.markdown("# Models")

model_info_dict = {
    "gpt-3.5-turbo-0125": {"tokens": 16385, "date": "Sep 2021"},
    "gpt-3.5-turbo-instruct": {"tokens": 4096 , "date": "Sep 2021"}
}

model_options = list(model_info_dict.keys())  # Extract option names

# Move everything inside the sidebar context
with st.sidebar:
    selected_option = st.selectbox("Select a model:", model_options)
    model_selected_option = selected_option

    if model_selected_option:
        information = model_info_dict[model_selected_option]

        # Create columns for left-right layout
        col1, col2 = st.columns([1, 1])

        with col1:
            # Display "Tokens" label
            st.write(f"Tokens: <span style='color: green;'> {information['tokens']} </span>", unsafe_allow_html=True)

        with col2:
            # Display date value right-aligned
            st.write(f"<div style='text-align: right;'>Date: <span style='color: red;'> {information['date']} </span> </div>", unsafe_allow_html=True)
            
# HTML sidebar to fine-tune model's parameters to customize the bot's responses.
st.sidebar.markdown("# Model Parameters")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
print(temperature)
max_tokens = st.sidebar.number_input("Max Tokens", 50, 500, 256, step=50)
top_p = st.sidebar.slider("Top P", 0.1, 1.0, 0.9, 0.1)
n = st.sidebar.number_input("N", 1, 5, 2, step=1)
stop = st.sidebar.text_input("Stop", "")
frequency_penalty = st.sidebar.slider("Frequency Penalty", 0.0, 1.0, 0.9, 0.1)
presence_penalty = st.sidebar.slider("Presence Penalty", 0.0, 1.0, 0.9, 0.1)

# Show all available vector databases
st.sidebar.markdown("# Vector Databases")
vectordb_options = ["ChromaDB", "Pinecone"]


with st.sidebar:
    # Creating the select box
    vectordb_selected_option = st.selectbox("Choose a database option:", vectordb_options)
    # Display the selected option
    st.write(f"You selected: {vectordb_selected_option}")

    # Conditional message based on the selected option
    if vectordb_selected_option == "Pinecone":
        st.markdown(
        """
        <div style="color: blue; background-color: #E9F7EF; padding: 10px; border-radius: 5px;">
            Please ensure you have configured your Pinecone credentials properly.
        </div>
        """,
        unsafe_allow_html=True
        )
        
        # Input fields for Pinecone credentials
        pinecone_api_key = st.text_input("Pinecone API Key", type="password")
        environment = st.text_input("Pinecone Environment")

        # Display the entered credentials (for demonstration purposes)
        # In a real application, you might want to use these credentials to connect to Pinecone
        if pinecone_api_key and environment:
            st.write(f"API Key: {pinecone_api_key}")
            st.write(f"Environment: {environment}")


# WikipediaLoader
def generate_wikipedia_response(input_text):
    # Get Wikipedia summary for the input text
    docs = WikipediaLoader(query=input_text, load_max_docs=1).load()
    if docs:
        wiki_metadata = docs[0].metadata
        wiki_summary = wiki_metadata.get('summary', 'No summary found')
        wiki_source = wiki_metadata.get('source', 'No source found')
        
        if wiki_summary and wiki_source:
            return f"{wiki_summary}\n\n**Source:** {wiki_source}"
        else:
            return "No complete Wikipedia metadata found for the input text."
    else:
        return "No Wikipedia documents found for the input text."

# Main app where user enters prompt and gets the response
chat_cnt = 0

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        print(message)
        # st.markdown(message["question"])
        st.markdown(message["content"])


# nltk.download('punkt')
# nltk.download('stopwords')

# tfidf_agent = TfidfGeneratorAgent()
# image_producer = ImageProducerAgent(tfidf_agent)

if user_input := st.chat_input("Enter a prompt here:", key=chat_cnt):
    # st.button("Generate Response", key="button")

    # Button to generate response
    if user_input:

        # query_language = detect(user_input)
        # print(query_language)
        # prompt_template = PromptTemplate.from_template(
        # "Generate the reply of {context} into {language} Language."
        # )
        # modified_query = prompt_template.format(context=user_input, language = query_language)
        # agent.query = modified_query
        # query3 = "অভিজ্ঞতাভিত্তিক শিখন পদ্ধতিটি কি?"
        # query2_with_knowledge = qa.invoke(query3)

        chat_history = []
        print("Result generation starts...")
        result = query_data(user_input)
        print("Wkipedia response starts..")
        wikiContent = generate_wikipedia_response(user_input)
        print("Wkipedia response GOT.")

        with st.chat_message("assistant"):
            st.markdown(result["query_text"])
            st.markdown(result["response"])
        # print(image_producer.data)
        # print(images)

        st.logo(wiki_logo)

        with st.expander("See explanation:"):
        # print(wikiContent)
            st.write(wikiContent)
            st.session_state.messages.append({"role": "user", "question": user_input, "content": result["response"]})

    else:
        st.write("Please enter a topic before generating a response.")
    chat_cnt += 1
