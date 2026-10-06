from xml.etree.ElementTree import ParseError

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAIError
from youtube_transcript_api import (
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeRequestFailed,
)

from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()
st.set_page_config(page_title="YouTube Video Chat", page_icon="▶")
st.title("Chat with a YouTube video")
st.write("Paste a YouTube link to load its English captions, then ask questions.")


def format_docs(retrieved_docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


def create_chain(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("The transcript did not contain any text.")

    vector_store = FAISS.from_documents(
        chunks,
        OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )
    prompt = PromptTemplate(
        template=(
            "You are a helpful assistant. Answer ONLY from the provided "
            "transcript context. If the context is insufficient, say you "
            "don't know.\n\n{context}\n\nQuestion: {question}"
        ),
        input_variables=["context", "question"],
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    return parallel_chain | prompt | llm | StrOutputParser()


if "chain" not in st.session_state:
    st.session_state.chain = None
    st.session_state.video_id = None
    st.session_state.messages = []

with st.form("youtube_video_form"):
    video_url = st.text_input(
        "YouTube video link",
        placeholder="https://www.youtube.com/watch?v=...",
    )
    load_video = st.form_submit_button("Load video")

if load_video:
    try:
        video_id = YoutubeLoader.extract_video_id(video_url.strip())
    except ValueError as error:
        st.error(f"Enter a valid YouTube video URL. {error}")
    else:
        st.session_state.chain = None
        st.session_state.video_id = None
        st.session_state.messages = []
        try:
            with st.spinner("Fetching captions and preparing the video..."):
                documents = YoutubeLoader.from_youtube_url(
                    video_url.strip(),
                    language=["en"],
                ).load()
                if not documents or not any(
                    document.page_content.strip() for document in documents
                ):
                    raise TranscriptsDisabled(video_id)
                chain = create_chain(documents)
        except TranscriptsDisabled:
            st.error("Captions are disabled for this video.")
        except NoTranscriptFound:
            st.error("No English captions were found for this video.")
        except VideoUnavailable:
            st.error("This video is unavailable. Check the link and try again.")
        except RequestBlocked:
            st.error(
                "YouTube blocked transcript requests from this network. "
                "Wait and try again later; using LangChain's loader cannot "
                "bypass YouTube's rate limit."
            )
        except YouTubeRequestFailed as error:
            if "429" in str(error) or "too many requests" in str(error).lower():
                st.error(
                    "YouTube is temporarily rate-limiting transcript requests "
                    "from this network (HTTP 429). Wait and try again later; "
                    "the app cannot bypass YouTube's rate limit."
                )
            else:
                st.error(f"Could not retrieve captions from YouTube: {error}")
        except ParseError:
            st.error(
                "YouTube returned an empty or invalid captions response. "
                "Try again later or check the video link."
            )
        except ValueError as error:
            st.error(str(error))
        except OpenAIError as error:
            st.error(f"OpenAI setup or request failed: {error}")
        else:
            st.session_state.chain = chain
            st.session_state.video_id = video_id
            st.success("Video loaded. Ask a question below.")

if st.session_state.chain is not None:
    st.caption(f"Loaded video: https://www.youtube.com/watch?v={st.session_state.video_id}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.button("Summarize the video"):
        st.session_state.messages.append(
            {"role": "user", "content": "Can you summarize the video"}
        )
        try:
            with st.spinner("Summarizing..."):
                answer = st.session_state.chain.invoke(
                    "Can you summarize the video"
                )
        except OpenAIError as error:
            st.error(f"OpenAI request failed: {error}")
        else:
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
            st.rerun()

    question = st.chat_input("Ask a question about the video")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        try:
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke(question)
        except OpenAIError as error:
            st.error(f"OpenAI request failed: {error}")
        else:
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
            st.rerun()
