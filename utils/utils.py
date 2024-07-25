import traceback
from utils.log import logger

from langchain_community.llms import OpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

SESSION_DICT = {}
LLM = OpenAI(
    model_name='Qwen2-7B-Instruct',
    base_url = "http://localhost:8000/v1",
    api_key = 'get_openai_key()',
    temperature = 0.5,
)

CONVERSATION_CHAIN = ConversationChain(
    llm = LLM, 
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

