import streamlit as st
from huggingface_hub import login
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.huggingface import HuggingFaceLLM
import torch

st.set_page_config(page_title="Chat with the Wikipedia docs", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)

st.title("Chat with the Wikipedia docs, powered by LlamaIndex 💬🦙")
st.info("Check out the tutorial to build RAG with LlamaIndex [blog post](https://huggingface.co/learn/cookbook/en/rag_llamaindex_librarian)", icon="📃")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about the subjects in Wikipedia docs in Arabic.",
        }
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    #chunk_dir = "/Rag Dataset/txt_files"  # Directory where your chunks are stored
    #documents_txt = SimpleDirectoryReader(chunk_dir).load_data()

    system_prompt = """
    You are a Q&A assistant. Your goal is to answer questions as
    accurately as possible based on the instructions and context provided.
    """
    query_wrapper_prompt = SimpleInputPrompt("{query_str}")

    # Log in to Hugging Face using the token from secrets
    huggingface_token = st.secrets.huggingface_token
    login(huggingface_token)

    # Check if GPU is available
    if torch.cuda.is_available():
        device_map = "auto"
        model_kwargs = {"torch_dtype": torch.float16, "load_in_8bit": True}
    else:
        device_map = {"": "cpu"}
        model_kwargs = {}

    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.1, "do_sample": True},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="CohereForAI/aya-23-8B",
        model_name="CohereForAI/aya-23-8B",
        device_map=device_map,
        model_kwargs=model_kwargs
    )

    embed_model = HuggingFaceEmbedding(model_name="google-bert/bert-base-multilingual-uncased")
    service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        llm=llm,
        embed_model=embed_model
    )

    persist_dir = "/db_indexing"
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context, embed_model=embed_model)

    return index

index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )

if prompt := st.chat_input("Ask a question"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        # Add response to message history
        st.session_state.messages.append(message)
