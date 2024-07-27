import traceback
from utils.log import logger
import re

from langchain_community.llms import OpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain.schema import BaseOutputParser

from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
import os

# extract vector from file
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import CSVLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from unstructured.file_utils.filetype import FileType, detect_filetype
from langchain.text_splitter import RecursiveCharacterTextSplitter

FILE_LOADERS ={
    FileType.CSV: CSVLoader,
    FileType.TXT: TextLoader,
    FileType.DOCX: UnstructuredWordDocumentLoader,
    FileType.PDF: PyPDFLoader,
    FileType.MD: UnstructuredMarkdownLoader
}

SESSION_DICT = {}
LLM = OpenAI(
    model_name='Qwen2-7B-Instruct',
    base_url = "http://localhost:8000/v1",
    api_key = 'get_openai_key()',
    temperature = 0.5,
    max_tokens = 7500
)

# 根据你生成的向量数据库做修改
persist_directory_chinese = '/root/autodl-tmp/llm_dev/data/yuntaisu'
model_kwargs = {'device': 'cuda'}
model_name = '/root/autodl-tmp/bge-small-zh-v1.5'
EMBEDDINGS = HuggingFaceBgeEmbeddings(
    model_name = model_name, 
    model_kwargs = model_kwargs    
)
DB = FAISS.load_local(
        persist_directory_chinese, 
        embeddings=EMBEDDINGS,
        allow_dangerous_deserialization=True
    )
RETRIEVER = DB.as_retriever(search_type="mmr", search_kwargs={"k":3})

class CleanupOutputParser(BaseOutputParser):
    """
    def parse(self, text: str) -> str:
        user_pattern = r"\nUser"
        text = re.sub(user_pattern, "", text)
        human_pattern = r"\nHuman:"
        text = re.sub(human_pattern, "", text)
        ai_pattern = r"\nAI:"   
        return re.sub(ai_pattern, "", text).strip()
    """
    def parse(self, text: str) -> str:
        """生成结果后处理，去除掉多余的问答！和\n 空格
        Args:
            text (str): _description_
        Returns:
            str: _description_
        """
        human_pattern = r"\nHuman.*"
        text = text.strip()
        text = re.sub(human_pattern, "", text, flags=re.DOTALL)
        return re.sub("\n", "", text)
        
    @property
    def _type(self) -> str:
        return "output_parser"
  
CONVERSATION_CHAIN = ConversationChain(
    llm = LLM, 
    output_parser=CleanupOutputParser(),
)

CONVERSATION_RAG_CHAIN = ConversationalRetrievalChain.from_llm(
    llm = LLM,
    retriever = RETRIEVER,
)

def reset_param(param, reset_value):
    try:
        param = reset_value if param is None or str(param).strip() == "" or str(param) == "null" else param
        return True, param
    except Exception as e:
        traceback_ = traceback.print_exc()
        return False, f"初始化参数错误\n{str(e)}\n{traceback_}"
  
def _chat_robot_api(
    session: str, 
    instruct_prompt: str, 
    content_prompt: str, 
    temperature: float = 0.0, 
    end_flag: int = 0,
    ):
    """问答api
    Args:
        session (str): 区分不同登录用户
        instruct_prompt (str): 
            传给大模型的指令
            针对不同的场景不断测试生成效果最好的指令，作为场景模板的组成部分
        content_prompt (str): 
            传给大模型的内容，可以是网络或本地知识库检索到的
            也可以是用户输入的
        end_flag (int): 
            结束会话的标志
    """
    LLM.temperature = temperature
    memory = SESSION_DICT[session] if session in SESSION_DICT else None
    if memory is None:
        # 构建Conversationmemory
        memory = ConversationBufferMemory(llm=LLM)
        memory.clear()
        memory.save_context(
            {"input": "开启一问一答模式"},
            {"output": "好的，收到！"}
        )
        SESSION_DICT[session] = memory

    # 添加狭义的指令
    if instruct_prompt != "":
        memory.save_context(
            {"input": instruct_prompt},
            {"output": "好的，收到！"}
        )
    
    CONVERSATION_CHAIN.memory = memory 
    CONVERSATION_CHAIN.llm = LLM  
    
    # 推理
    response = CONVERSATION_CHAIN(content_prompt)
    # logger.info(memory.load_memory_variables({}))
    logger.info('=================================')
    logger.info(f"\n{memory.buffer}")
    logger.info('=================================')
    if end_flag:
        try:
            SESSION_DICT.pop(session)
            memory.clear()
            del memory
        except Exception as e:
            traceback_ = traceback.print_exc()
            return False, f"{str(e)}\n{traceback_}"
    return True, response['response']

def _rag_chat_robot_api(
    session: str, 
    instruct_prompt: str, 
    content_prompt: str, 
    temperature: float = 0.0, 
    end_flag: int = 0,
    rag_flag: int = 0
    ):
    """问答api
    Args:
        session (str): 区分不同登录用户
        instruct_prompt (str): 
            传给大模型的指令
            针对不同的场景不断测试生成效果最好的指令，作为场景模板的组成部分
        content_prompt (str): 
            传给大模型的内容，可以是网络或本地知识库检索到的
            也可以是用户输入的
        end_flag (int): 
            结束会话的标志
    """
    LLM.temperature = temperature
    memory = SESSION_DICT[session] if session in SESSION_DICT else None
    if memory is None:
        # 构建 Conversationmemory
        memory = ConversationBufferMemory(llm=LLM, memory_key="history", return_messages=True)
        # memory = ConversationSummaryMemory(llm=LLM, memory_key="history", return_messages=True)
        memory.clear()
        memory.save_context(
            {"input": "开启一问一答模式"},
            {"output": "好的，收到！"}
        )
        SESSION_DICT[session] = memory

    # 添加狭义的指令
    if instruct_prompt != "":
        memory.save_context(
            {"input": instruct_prompt},
            {"output": "好的，收到！"}
        )
    if rag_flag:
        try:
            # change memory_key
            memory.memory_key = "chat_history"
            CONVERSATION_RAG_CHAIN.memory = memory
            response = CONVERSATION_RAG_CHAIN(content_prompt)
            logger.info('=================================')
            logger.info(f"\n{memory.buffer}")
            logger.info('=================================')
            return True, response['answer']
        except Exception as e:
            traceback_ = traceback.print_exc()
            return False, f"{str(e)}\n{traceback_}"
    
    # change memory_key
    memory.memory_key = "history"
    CONVERSATION_CHAIN.memory = memory 
    CONVERSATION_CHAIN.llm = LLM  

    # 推理
    response = CONVERSATION_CHAIN(content_prompt)
    logger.info('=================================')
    logger.info(f"\n{memory.buffer}")
    logger.info('=================================')
    if end_flag:
        try:
            SESSION_DICT.pop(session)
            memory.clear()
            del memory
        except Exception as e:
            traceback_ = traceback.print_exc()
            return False, f"{str(e)}\n{traceback_}"
    return True, response['response']

def embedding_file(directory_path: str):
    return_str = ""
    documents_list = []
    try:
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        for file in files:
            file_path = os.path.join(directory_path, file)
            file_type = detect_filetype(file_path)
            file_loader = FILE_LOADERS[file_type] if file_type in FILE_LOADERS else None
            if file_loader is None:
                return_str += f"{file_path}\n"
                continue
            loader = file_loader(file_path)
            documents = loader.load() 
            documents_list.extend(documents)
            # [1, 2, 3, 4, 5] [6, 7, 8, 9, 10]
            # [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
            # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        text_split = RecursiveCharacterTextSplitter(
            chunk_size = 1000, 
            chunk_overlap = 200
        )
        texts = text_split.split_documents(documents_list)
        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, .........]
        model_kwargs = {'device': 'cuda'}
        model_name = '/root/autodl-tmp/bge-small-zh-v1.5'
        embeddings = HuggingFaceBgeEmbeddings(
            model_name = model_name, 
            model_kwargs = model_kwargs    
        )
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(directory_path)
        return True, return_str
    except Exception as e:
        traceback_ = traceback.print_exc()
        return False, f"{str(e)}\n{traceback_}"
    
