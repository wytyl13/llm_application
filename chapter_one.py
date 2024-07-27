from transformers import AutoTokenizer, AutoModelForCausalLM 
from transformers import pipeline



from utils.utils import _chat_robot_api
import uvicorn
from fastapi import FastAPI
from typing import Dict
from dataclasses import dataclass
from utils.R import R
from utils.utils import reset_param
from utils.utils import _rag_chat_robot_api

from langchain.vectorstores import FAISS

from utils.utils import embedding_file

@dataclass
class request_data:
    session: str
    instruct_prompt: str
    content_prompt: str
    temperature: float
    end_flag: int
    rag_flag: int
    
app = FastAPI()

@app.post('/chat')
async def chat_robot_api(request: request_data):
    session = request.session
    instruct_prompt = request.instruct_prompt
    content_prompt = request.content_prompt
    temperature = request.temperature
    end_flag = request.end_flag
    
    status, session = reset_param(session, "")
    if not status:
        return R.fail(f"参数初始化失败\n{session}") 
    
    status, instruct_prompt = reset_param(instruct_prompt, "")
    if not status:
        return R.fail(f"参数初始化失败\n{instruct_prompt}") 
    
    status, content_prompt = reset_param(content_prompt, "")
    if not status:
        return R.fail(f"参数初始化失败\n{content_prompt}") 
    
    status, temperature = reset_param(temperature, 0.0)
    if not status:
        return R.fail(f"参数初始化失败\n{temperature}") 
    
    status, end_flag = reset_param(end_flag, 0)
    if not status:
        return R.fail(f"参数初始化失败\n{end_flag}") 
    
    # session cntent_prompt must not be empty!
    # temperature default value is 0.0, end_flag default value is 0.
    # instruct prompt can be empty!
    if session == "" or content_prompt == "":
        return R.fail("session或者content_prompt参数不能为空！")
    
    status, result = _chat_robot_api(
        session, 
        instruct_prompt, 
        content_prompt, 
        temperature, 
        end_flag)
    if not status:
        return R.fail(result)
    return R.success(result)

@app.post('/rag_chat')
async def rag_chat_robot_api(request: request_data):
    session = request.session
    instruct_prompt = request.instruct_prompt
    content_prompt = request.content_prompt
    temperature = request.temperature
    end_flag = request.end_flag
    rag_flag = request.rag_flag
    
    status, session = reset_param(session, "")
    if not status:
        return R.fail(f"参数初始化失败\n{session}") 
    
    status, instruct_prompt = reset_param(instruct_prompt, "")
    if not status:
        return R.fail(f"参数初始化失败\n{instruct_prompt}") 
    
    status, content_prompt = reset_param(content_prompt, "")
    if not status:
        return R.fail(f"参数初始化失败\n{content_prompt}") 
    
    status, temperature = reset_param(temperature, 0.0)
    if not status:
        return R.fail(f"参数初始化失败\n{temperature}") 
    
    status, end_flag = reset_param(end_flag, 0)
    if not status:
        return R.fail(f"参数初始化失败\n{end_flag}") 
    
    status, rag_flag = reset_param(rag_flag, 0)
    if not status:
        return R.fail(f"参数初始化失败\n{rag_flag}") 
    
    # session cntent_prompt must not be empty!
    # temperature default value is 0.0, end_flag default value is 0.
    # instruct prompt can be empty!
    if session == "" or content_prompt == "":
        return R.fail("session或者content_prompt参数不能为空！")
    status, result = _rag_chat_robot_api(
        session, 
        instruct_prompt, 
        content_prompt, 
        temperature, 
        end_flag, 
        rag_flag)
    if not status:
        return R.fail(result)
    return R.success(result)

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=6006)
    
    """
    from utils.utils import embedding_file
    status, reuslt = embedding_file("/root/autodl-tmp/llm_dev/data/yuntaisu")
    print(status)
    """
    
    """
    from langchain_community.llms import OpenAI
    LLM = OpenAI(
        model_name='Qwen2-7B-Instruct',
        base_url = "http://localhost:8000/v1",
        api_key = 'get_openai_key()',
        temperature = 0.5,
    )
    
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    persist_directory_chinese = "/root/autodl-tmp/llm_dev/data/index"
    model_kwargs = {'device': 'cuda'}
    model_name = '/root/autodl-tmp/bge-small-zh-v1.5'
    embeddings = HuggingFaceBgeEmbeddings(
        model_name = model_name, 
        model_kwargs = model_kwargs    
    )
    from langchain.vectorstores import FAISS
    query = "什么是基本语义相似度?"
    db = FAISS.load_local(
        persist_directory_chinese, 
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    # docs = db.similarity_search(query, k=3)
    # print(docs)
    # print(len(docs))
    
    from langchain.chains import RetrievalQA
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k":3})
    qa = RetrievalQA.from_chain_type(
        llm=LLM, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True
    )
    query = "什么是基本语义相似度?"
    result = qa({"query": query})
    print(result['result'])
    print(result['source_documents'])
    
    from langchain.memory import ConversationSummaryMemory
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain
    memory = ConversationBufferMemory(llm=LLM, memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(LLM,retriever=retriever,memory=memory)
    question ="什么是LLM?"
    result = qa(question)
    question ="什么是基本语义相似度?"
    result = qa(question)
    print(result['answer'])
    print(qa.memory.buffer)
    
    """
    
    # from langchain.document_loaders import PyPDFLoader
    # from langchain.document_loaders import CSVLoader
    # from langchain.document_loaders import UnstructuredWordDocumentLoader
    # from langchain.document_loaders import TextLoader
    # from langchain.document_loaders import UnstructuredMarkdownLoader
    # from unstructured.file_utils.filetype import FileType, detect_filetype
    # file_loaders ={
    #     FileType.CSV: CSVLoader,
    #     FileType.TXT: TextLoader,
    #     FileType.DOCX: UnstructuredWordDocumentLoader,
    #     FileType.PDF: PyPDFLoader,
    #     FileType.MD: UnstructuredMarkdownLoader
    # }
    # file_path = "/root/autodl-tmp/llm_dev/data/LLM-v1.0.0(1).pdf"
    # file_type = detect_filetype(file_path)
    # file_loader = file_loaders[file_type]
    # loader = file_loader(file_path)
    # documents = loader.load()
    
    # from langchain.text_splitter import RecursiveCharacterTextSplitter
    # text_split = RecursiveCharacterTextSplitter(
    #     chunk_size = 1000, 
    #     chunk_overlap = 200
    # )
    # texts = text_split.split_documents(documents)

    # from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    # model_kwargs = {'device': 'cuda'}
    # model_name = '/root/autodl-tmp/bge-small-zh-v1.5'
    # embeddings = HuggingFaceBgeEmbeddings(
    #     model_name = model_name, 
    #     model_kwargs = model_kwargs    
    # )
    
    # from langchain.vectorstores import FAISS
    # persist_directory_chinese = '/root/autodl-tmp/llm_dev/data/index'
    # db = FAISS.from_documents(texts, embeddings)
    # db.save_local(persist_directory_chinese)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    