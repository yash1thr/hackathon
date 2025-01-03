
import streamlit as st
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared, operations
import os
from unstructured_client.models.errors import SDKError
from unstructured.staging.base import dict_to_elements
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.chains.summarize import load_summarize_chain
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import uuid
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from base64 import b64decode
import chromadb

chromadb.api.client.SharedSystemClient.clear_system_cache()

os.environ['OPENAI_API_KEY']=''
working_dir = os.getcwd()

# Prompt
prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.
    Table or text chunk: {element}
    """
image_prompt_template = """Describe the image in detail. For context,
                  the image is part of a research paper explaining the transformers
                  architecture. Be specific about graphs, such as bar plots."""

def load_documents(file_path):
    ##Code to load tables, images and text from unstructured.io
    s = UnstructuredClient(
        api_key_auth="",
        server_url="https://api.unstructured.io/general/v0/general",
    )
    with open(file_path, "rb") as f:
      files=shared.Files(
        content=f.read(),
        file_name=file_path,
      )
    req = operations.PartitionRequest(
        partition_parameters=shared.PartitionParameters(
        files=files,
        strategy="hi_res",
        hi_res_model_name="yolox",
        skip_infer_table_types=[],
        pdf_infer_table_structure=True,
        extract_image_block_types=["Image"],
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
      ))
    try:
        resp = s.general.partition(request=req)
        elements = dict_to_elements(resp.elements)
        return elements
    except SDKError as e:
        print(e)

        
def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64


def parse_docs(docs):
    """Split base64-encoded images and texts"""
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(doc)
    return {"images": b64, "texts": text}


def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.text

    # construct prompt with context (including images)
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and the below image.
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )


def setup_vectorstore(elements):
    tables = []
    texts = []
    for chunk in elements:
        if "Table" in str(type(chunk)):
            tables.append(chunk)
        if "CompositeElement" in str(type((chunk))):
            texts.append(chunk)
            
    images = get_images_base64(elements)
    prompt = ChatPromptTemplate.from_template(prompt_text)
    # Summary chain
    model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    # Summarize text
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})

    # Summarize tables
    tables_html = [table.metadata.text_as_html for table in tables]
    table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})
    messages = [
        (
            "user",
            [
                {"type": "text", "text": image_prompt_template},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                },
            ],
        )
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
    image_summaries = chain.batch(images)
    st.session_state.vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=OpenAIEmbeddings())

    # The storage layer
    store = InMemoryStore()
    id_key = "doc_id"
    st.session_state.retriever = MultiVectorRetriever(
        vectorstore=st.session_state.vectorstore,
        docstore=store,
        id_key=id_key,
    )
    # Add texts
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
    ]
    st.session_state.retriever.vectorstore.add_documents(summary_texts)
    st.session_state.retriever.docstore.mset(list(zip(doc_ids, texts)))

    # Add tables
    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
    ]
    st.session_state.retriever.vectorstore.add_documents(summary_tables)
    st.session_state.retriever.docstore.mset(list(zip(table_ids, tables)))

    # Add image summaries
    img_ids = [str(uuid.uuid4()) for _ in images]
    summary_img = [
        Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)
    ]
    st.session_state.retriever.vectorstore.add_documents(summary_img)
    st.session_state.retriever.docstore.mset(list(zip(img_ids, images)))

    chain = (
      {
          "context": st.session_state.retriever | RunnableLambda(parse_docs),
          "question": RunnablePassthrough(),
      }
      | RunnableLambda(build_prompt)
      | ChatOpenAI(model="gpt-4o-mini")
      | StrOutputParser()
    )
    return chain


st.set_page_config(
    page_title="Chat with your documents",
    page_icon="📑",
    layout="centered"
)

st.title("📝Chat With your docs 😎")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader(label="Upload your PDF")

if uploaded_file:
    if "conversation_chain" not in st.session_state:
        progress = st.progress(0,text='loading file')
        file_path = f"{working_dir}/{uploaded_file.name}"
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        progress.progress(1/ 3, text='Extracting data from Unstructred.io')
        elements= load_documents(file_path)
        progress.progress(2/ 3, text='Adding to VectorDB')
        conversation_chain= setup_vectorstore(elements)
        progress.progress(1.0, text='Successfully loaded to VectorDB')
        st.session_state.conversation_chain = conversation_chain
    
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask any questions relevant to uploaded pdf")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if "conversation_chain" not in st.session_state:
            response='Please upload pdf'
        else:
            response =st.session_state.conversation_chain.invoke(user_input)
        assistant_response = response
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

