import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langdetect import detect

from dotenv import load_dotenv

import openai 
import os

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE_WITH_CONTEXT = """
Here is teh Context:
{context}

---

Try to answer the question based on the above context: {question} in {language} language. If you don't know, then think rationally and respond from your knowledge.
"""

PROMPT_TEMPLATE_WITHOUT_CONTEXT = """
---

Try to think rationally and answer the question based on your knowledge {question} in {language} language.
"""

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

response_dict = dict()


def query_data(query_text):
    # Create CLI.
    # parser = argparse.ArgumentParser()
    # parser.add_argument("query_text", type=str, help="The query text.")
    # args = parser.parse_args()
    # query_text = args.query_text

    # Prepare the DB.
    language = detect(query_text)

    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        # print(f"Unable to find matching results.")
        # return
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_WITHOUT_CONTEXT)
        prompt = prompt_template.format(question=query_text, language = language)
        print(prompt)

    else:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_WITH_CONTEXT)
        prompt = prompt_template.format(context=context_text, question=query_text, language = language)
        print(prompt)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    response_dict["query_text"] = query_text
    response_dict["response"] = response_text
    response_dict["source"] = sources

    print(response_dict)
    return response_dict


if __name__ == "__main__":
    while True:
        query_text = input("Please enter your question: ")
        query_data(query_text)
    
