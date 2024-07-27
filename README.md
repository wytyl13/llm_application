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

##### Naive RAG system
```
retrieval augmented generation.
索引
检索

结构化数据库 0, ... 
全文检索 
0 我

我是谁 file_1
我来自哪里？file_2

我 file_1, file_2
是 file_1
谁 file_1
来自 file_2
哪里 file_2
elasticsearch

来自
来往

embedding
我是谁
我来自哪里？
我 file_1 [1, 2, 3, 4, 5]  512
是 file_1
谁 file_1
我 file_2
来自 file_2
哪里 file_2

我是谁？--> embedding [1, 2, 3, 4, 5, 6], dimension is 512

<!-- query_embedding @ database_embedding -->
database_embedding(6, 512) @ query_embedding(512, 1) = score(6, 1)

余弦相似度
[1, 1, 1, 1, 1]
[1, 1, 1, 1, 1]
cos_theta = 点积 / （根号下（向量A每个元素的平方和）* 根号下（向量B每个元素的平方和））
cos_theta = 5 / 5 = 1


检索方法：文本检索（倒排索引）和语义检索（embedding 向量相似度）
前者的优势是效率高，后者的优势是可以检索到多重词汇

倒排索引一般应用于数据库较完善的全文检索

实现功能
在chat system接口的基础上 添加知识库
详细规则如下：
在缓存历史聊天记录的基础上，添加知识库检索的功能
如：当给出指令去知识库检索后回答的时候，此时将会是一个参考知识库的独立回答
然后将本次回答的内容存储到聊天缓存中，作为后续回答的上下文参考依据

1 将文件数据制作成向量数据库存储到磁盘中
    1 多模态文件读取成文本
        pip install unstructured -i https://pypi.tuna.tsinghua.edu.cn/simple/
        from langchain.document_loaders import PyPDFLoader
        from langchain.document_loaders import CSVLoader
        from langchain.document_loaders import UnstructuredWordDocumentLoader
        from langchain.document_loaders import TextLoader
        from langchain.document_loaders import UnstructuredMarkdownLoader
        from unstructured.file_utils.filetype import FileType, detect_filetype
        file_loaders ={
            FileType.CSV: CSVLoader,
            FileType.TXT: TextLoader,
            FileType.DOCX: UnstructuredWordDocumentLoader,
            FileType.PDF: PyPDFLoader,
            FileType.MD: UnstructuredMarkdownLoader
        }
        file_path = "/root/autodl-tmp/llm_dev/data/LLM-v1.0.0(1).pdf"
        file_type = detect_filetype(file_path)
        file_loader = file_loaders[file_type]
        loader = file_loader(file_path)
        documents = loader.load()
    2 文本分割
        字符分割、递归分割、句子分割、命题分割（可以将句子独立）
        https://www.rungalileo.io/blog/mastering-rag-advanced-chunking-techniques-for-llm-applications#propositions
        langchain中提供了各种分割器
        https://chunkviz.up.railway.app/
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_split = RecursiveCharacterTextSplitter(
            chunk_size = 1000, 
            chunk_overlap = 200
        )
        texts = text_split.split_documents(documents)

    3 embedding模型
        pip install tiktoken pypdf sentence_transformers -i https://mirrors.cloud.tencent.com/pypi/simple/
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        model_kwargs = {'device': 'cuda'}
        model_name = '/root/autodl-tmp/bge-small-zh-v1.5'
        embeddings = HuggingFaceBgeEmbeddings(
            model_name = model_name, 
            model_kwargs = model_kwargs    
        )
        print(embeddings.embed_query("我是谁"))

    4 使用向量数据库存储到本地磁盘
        pip install faiss-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple/
        from langchain.vectorstores import FAISS
        persist_directory_chinese = '/root/autodl-tmp/llm_dev/data/index'
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(persist_directory_chinese)

2 检索增强生成
    1 检索
        余弦距离：夹角 不看向量长度
        点积
        欧氏距离
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
        docs = db.similarity_search(query, k=3)
        print(docs)
    2 增强回答
        # RetrievalQA
        from langchain.chains import RetrievalQA
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k":3})
        qa = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever, 
            return_source_documents=True
        )
        query = "什么是基本语义相似度?"
        result = qa({"query": query})
        print(result['result'])
        print(result['source_documents'])
        
    3 上下文记忆增强回答
        from langchain.memory import ConversationSummaryMemory
        from langchain.chains import ConversationalRetrievalChain
        db = FAISS.load_local(
            '/root/autodl-tmp/llm_dev/data/index', 
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k":3})
        memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
        qa = ConversationalRetrievalChain.from_llm(llm,retriever=retriever,memory=memory)
        question ="什么是基本语义相似度?"
        result = qa(question)
        print(result['answer'])
```