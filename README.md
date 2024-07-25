#### 大语言模型开发
##### 启动大语言模型服务
```
# 1 模型下载
    https://huggingface.co/Qwen/Qwen2-7B-Instruct
    https://hf-mirror.com/Qwen/Qwen2-7B-Instruct/
    https://modelscope.cn/models

    # int ,float, float32, float16
    # int4 ,int8
    # 7B 70亿
    # float32, 32 bits = 4 Bytes (1 Bytes = 8 bits)
    # 1 KB = 1024 Bytes
    # 1 MB = 1024 KB
    # 1 GB = 1024 MB
    # 70 0000 0000 * 32 / 8 / 1024 / 1024 / 1024 =  26 GB
    # 推理： 前向传播 ... 
    # 训练:  

    # download model from modelscope used code or git clone
        from modelscope import snapshot_download
        model_dir = snapshot_download('qwen/Qwen2-7B-Instruct', cache_dir='/root/autodl-tmp', revision='master')

# 2 三种办法启动大语言模型
源码，transformers, vllm, ollama

conda create --name llm python==3.10

# 2.1 transformers
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install numpy==1.24.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
    from transformers import AutoTokenizer, AutoModelForCausalLM 
    from transformers import pipeline
    model_name = "/root/autodl-tmp/qwen/Qwen2-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline('text-generation', model = model, tokenizer = tokenizer, device='cuda')
    print(pipe('i am very good!'))

# 2.2 vllm
# started the llm server used vllm
# you can check the vllm document from: https://docs.vllm.ai/en/stable/models/supported_models.html
    pip install vllm==0.4.0.post1 -i https://pypi.tuna.tsinghua.edu.cn/simple/
    CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --model /root/autodl-tmp/qwen/Qwen2-7B-Instruct --served-model-name Qwen2-7B-Instruct --max-model-len=2048 -tp 2 --dtype=half --port 6006

    参数解析
    CUDA_VISIBLE_DEVICES=0,1 设置可使用的gpu编号
    --model 模型在磁盘的绝对路径
    --served-model-name 模型开启服务对应的别名
    --max-model-len 设置最大token大小，更小的token占用更低的显存。小模型一般不建议设置该参数
    -tp 2 即tensor_parallel_size=2  意味着分割模型到两个GPU上，也即多卡推理
    --enforce-eager 强制torch开启eager'模式
    --gpu-memory-utilization 该参数越大，占用的显存越低，一般为1或者0.9 如果发现开启模型后报错OOM，那么增大该参数可以适当降低显存消耗
    --dtype=half 半精度，显存占用量小于单精度
    还有一种多卡方式是定义环境变量 export NCCL_IGNORE_DISABLED_P2P=1

    # test the model server started by vllm
    curl http://localhost:8000/v1/models
    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "Qwen2-7B-Instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "你好"}
            ]
    }'

# 2.3 started the llm server used ollama
# change the model path and server host
# download the ollama sh and install it
# check the available models used the link: https://ollama.com/library
curl -fsSL https://ollama.com/install.sh | sh
    export OLLAMA_MODELS=/root/autodl-tmp/ollama_models
    export OLLAMA_HOST=0.0.0.0:8000
    ollama serve
    ollama list
    注意如果使用不同的终端调用
    ollma服务的时候一定要保证不同终端上的 OLLAMA_HOST 和 OLLAMA_MODELS 这两个环境变量一致

    # down load the qwen model
    ollama pull qwen 
    curl http://localhost:6006/api/chat -d '{
    "model": "gemma2",
    "messages": [
        { "role": "user", "content": "why is the sky blue?" }
    ]
    }'

# llm development used langchain.
pip install langchain langchain_community openai -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

##### CHAT system
```
如果你在本地部署了大模型
如果你在本地局域网内开发可以直接使用以上url或者更改该大模型所在机器的局域网ip即可
如果你在非局域网机器开发，则需要你的本地机器开通外网端口
然后你使用nginx或者ssh将大模型所在的机器映射到本地机器

如果你在autodl中启动了大模型，而希望在本地开发，则直接使用官方的外网端口6006做映射
你只需要使用6006端口启动大模型
然后在你的本地机器上做如下操作
ssh -CNg -L 6006:127.0.0.1:6006 root@connect.cqa1.seetacloud.com -p 40278
输入密码后没有任何回显说明映射成功
此时你可以在本地机器使用localhost:6006成功访问autodl上在6006端口部署的模型

在本地使用langchain和url初始化大模型
# 初始化llm used the model server url started by vllm or other openai api
# 使直接使用openai工具，base_url是启动服务的ip+端口，可以实现对源码项目启动，vllm启动和ollama启动的调用
# openai_key 可以忽略
# openai 风格聊天
    pip install openai -i https://pypi.tuna.tsinghua.edu.cn/simple/
    from openai import OpenAI
    client = OpenAI(
        base_url = "http://localhost:6006/v1",
        api_key = 'openai_key',
    )
    messages = [
        {"role": "system", "content": "你是一名心里咨询师，你要帮助用户脱离困境！"}, 
        {"role": "user", "content": "以下是我的人生目标：我是一名ai从业者，我要用最简单的代码解决世界难题！"},
        {"role": "assistant", "content": "好的收到！"},
        {"role": "user", "content": "我在30岁的时候迷茫了，我不知道该怎么实现我的目标！请首先告诉我的目标，然后请告诉我具体实现路径！"},
    ]
    response = client.chat.completions.create(
        model = 'qwen', 
        messages = messages, 
        temperature = 0.0, 
        max_tokens = 1024
    )
    注意max_tokens 不能超过模型启动时候定义的--max-model-len，一般情况下只需要在模型启动的时候指定--max-model-len即可
    print(response.choices[0].message.content)

# 使用langchain中的openai工具 (推荐*******更加通用)
    from langchain_community.llms import OpenAI

    # 這塊發 发现问题
    # 使用 OpenAI 可以正常初始化 vllm启动的 llm 而无法正常初始化ollama启动的 llm
    # 也就是说ollama使用OpenAI 无法正常初始化llm
    # 但是使用openai的接口却可以正常初始化client
    # 现在的问题是ollama开启的服务只能使用Ollama 和 ChatOllama 去初始化llm
    from langchain_community.llms import OpenAI
    llm = OpenAI(
        model_name='Qwen2-7B-Instruct',
        base_url = "http://localhost:6006/v1",
        api_key = 'openai_key',
        temperature = 0.0,
        max_tokens = 4096
    )
    print(llm.invoke('我是谁？'))
    messages = [
        {"role": "system", "content": "你是一名心里咨询师，你要帮助用户脱离困境！"}, 
        {"role": "user", "content": "以下是我的人生目标：我是一名ai从业者，我要用最简单的代码解决世界难题！"},
        {"role": "assistant", "content": "好的收到！"},
        {"role": "user", "content": "我在30岁的时候迷茫了，我不知道该怎么实现我的目标！请首先告诉我的目标，然后请告诉我具体实现路径！"},
    ]
    print(llm.invoke(messages))

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

# 使用langchain中的ollama工具初始化ollama启动的服务
    from langchain_community.llms import Ollama
    from langchain_community.chat_models.ollama import ChatOllama
    <!-- recall  http://localhost:6006/api/chat-->
    llm = ChatOllama(
        model='qwen', 
        base_url='http://localhost:6006', 
        temperature=0.0
    )
    print(llm.invoke('who am i?').content)

    <!-- recall  http://localhost:6006/api/generate-->
    llm = Ollama(
        model="qwen",
        base_url='http://localhost:6006',
        temperature=0.0
    )
    print(llm.invoke('who am i?'))

以上，注意
model_name和开启大模型对应的服务名称要一致
base_url一般是http的
本地部署的url, api_key可以不用输入，访问第三方接口则需要输入

# 使用langchain构建缓存历史的聊天接口
# used Conversationmemory and ConversationChain
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationChain
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm = llm,
        memory = memory,
    )

    # 推理
    content_prompt = 'who am i'
    response = conversation(content_prompt)

    # 获取当前conversation中对应的memory中存储的会话信息
    logger.info(conversation.memory.buffer)

    # 打印历史会话信息
    # 返回当前memory记录的历史信息
    logger.info(memory.load_memory_variables({}))

    # 获取当前对话的回答
    logger.info(response['response'])
```