from transformers import AutoTokenizer, AutoModelForCausalLM 
from transformers import pipeline



from utils.utils import _chat_robot_api
import uvicorn
from fastapi import FastAPI
from typing import Dict
from dataclasses import dataclass
from utils.R import R
from utils.utils import reset_param

@dataclass
class request_data:
    session: str
    instruct_prompt: str
    content_prompt: str
    temperature: float
    end_flag: int
    
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

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=6006)
    
# if __name__ == "__main__":
    """
    model_name = "/root/autodl-tmp/qwen/Qwen2-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto')
    pipe = pipeline('text-generation', 
                    model = model, 
                    tokenizer = tokenizer, 
                    max_new_tokens = 2048)
    print(pipe('我是谁？'))
    """
    
    """
    from openai import OpenAI
    client = OpenAI(
        base_url = "http://localhost:8000/v1",
        api_key = 'openai_key',
    )
    messages = [
        {"role": "system", "content": "你是一名心里咨询师，你要帮助用户脱离困境！"}, 
        {"role": "user", "content": "以下是我的人生目标：我是一名ai从业者，我要用最简单的代码解决世界难题！"},
        {"role": "assistant", "content": "好的收到！"},
        {"role": "user", "content": "我在30岁的时候迷茫了，我不知道该怎么实现我的目标！请首先告诉我的目标，然后请告诉我具体实现路径！"},
    ]
    response = client.chat.completions.create(
        model = 'Qwen2-7B-Instruct', 
        messages = messages, 
        temperature = 0.0, 
        max_tokens = 1024
    )
    print(response.choices[0].message.content)
    
    """
    
    """
    from langchain_community.llms import OpenAI
    llm = OpenAI(
        model_name='Qwen2-7B-Instruct',
        base_url = "http://localhost:8000/v1",
        api_key = 'openai_key',
        temperature = 0.0,
    )
    
    messages = [
        {"role": "system", "content": "你是一名心里咨询师，你要帮助用户脱离困境！"}, 
        {"role": "user", "content": "以下是我的人生目标：我是一名ai从业者，我要用最简单的代码解决世界难题！"},
        {"role": "assistant", "content": "好的收到！"},
        {"role": "user", "content": "我在30岁的时候迷茫了，我不知道该怎么实现我的目标！请首先告诉我的目标，然后请告诉我具体实现路径！"},
    ]
    
    from langchain.schema import (
        AIMessage,
        HumanMessage,
        SystemMessage
    )
    messages= [
        SystemMessage(content="你是一名心里咨询师，你要帮助用户脱离困境！"),
        HumanMessage(content="以下是我的人生目标：我是一名ai从业者，我要用最简单的代码解决世界难题！"),
        AIMessage(content="好的收到！"),
        HumanMessage(content="我在30岁的时候迷茫了，我不知道该怎么实现我的目标！请首先告诉我的目标，然后请告诉我具体实现路径！"),
    ]
    
    print(llm.invoke(messages))
    """
    
    """
    from langchain_community.llms import Ollama
    from langchain_community.chat_models.ollama import ChatOllama
    llm = ChatOllama(
        model='qwen', 
        base_url='http://localhost:8000', 
        temperature=0.0
    )
    llm = Ollama(
        model="qwen",
        base_url='http://localhost:8000',
        temperature=0.0
    )
    from langchain.schema import (
        AIMessage,
        HumanMessage,
        SystemMessage
    )
    messages= [
        SystemMessage(content="你是一名心里咨询师，你要帮助用户脱离困境！"),
        HumanMessage(content="以下是我的人生目标：我是一名ai从业者，我要用最简单的代码解决世界难题！"),
        AIMessage(content="好的收到！"),
        HumanMessage(content="我在30岁的时候迷茫了，我不知道该怎么实现我的目标！请首先告诉我的目标，然后请告诉我具体实现路径！"),
    ]
    
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationChain
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm = llm,
        memory = memory,
    )
    content_prompt = '我是谁'
    response = conversation(content_prompt)
    print(conversation.memory.buffer)
    print("======================================")
    print(response['response'])
    """
    
    
    
    
    
    
    
    
    
    
    
    