
import time
import threading
import importlib
from toolbox import update_ui, get_conf, update_ui_lastest_msg
from multiprocessing import Process, Pipe
import yshs
import os, sys

model_name = 'openai/gpt-4'
timeout_bot_msg = '[Local Message] Request timeout. Network error. Please check proxy settings in config.py.' + \
                  '网络错误，检查代理服务器是否可用，以及代理设置的格式是否正确，格式须是[协议]://[地址]:[端口]，缺一不可。'

def request_model(model, messages, engine=None, **kwargs):
    """统一的请求模型函数"""
    result = yshs.LLM.chat(
            model=model,  # 模型名称，例如 "openai/gpt-4", 可通过 list_models.py 查看可用模型
            engine=engine,  # 引擎名称, 可选参数，例如 "gpt-4-1106-preview"
            messages=messages,  # 一个会话的消息列表
            stream=True,  # 是否以流式的方式返回结果
            **kwargs  # 其他参数
        )

    full_result = ""
    for i in result:  # result 是一个生成器，每次迭代返回一个消息
        full_result += i
        sys.stdout.write(i)
        sys.stdout.flush()
    # print()
    return full_result

def request_gpt35(messages):
    model = "openai/gpt-3.5-turbo"
    return request_model(model, messages=messages)

def request_gpt4(messages):
    model = "openai/gpt-4"
    engine = 'gpt-4-1106-preview'
    return request_model(model, messages=messages, engine=engine)

def validate_key():
    XFYUN_APPID = get_conf('YSHS_API_KEY')
    if XFYUN_APPID == '00000000' or XFYUN_APPID == '': 
        return False
    return True

def predict_no_ui_long_connection(inputs, llm_kwargs, history=[], sys_prompt="", observe_window=[], console_slience=False):
    """
        ⭐多线程方法
        函数的说明请见 request_llms/bridge_all.py
    """
    watch_dog_patience = 5
    response = ""

    if validate_key() is False:
        raise RuntimeError('请配置YSHS模型的YSHS_APPID')
    messages = generate_message(inputs, history, sys_prompt)
    model = "openai/gpt-4"
    engine = 'gpt-4-1106-preview'
    result = yshs.LLM.chat(
            model=model,  # 模型名称，例如 "openai/gpt-4", 可通过 list_models.py 查看可用模型
            engine=engine,  # 引擎名称, 可选参数，例如 "gpt-4-1106-preview"
            messages=messages,  # 一个会话的消息列表
            stream=True,  # 是否以流式的方式返回结果
        )
    # 开始接收回复  
    response = ""
    for i in result: 
        if len(observe_window) >= 1:
            observe_window[0] = i
        if len(observe_window) >= 2:
            if (time.time()-observe_window[1]) > watch_dog_patience: raise RuntimeError("程序终止。")
        response += i
        sys.stdout.write(i)
        sys.stdout.flush()
        
    return response

def predict(inputs, llm_kwargs, plugin_kwargs, chatbot, history=[], system_prompt='', stream = True, additional_fn=None):
    """
        ⭐单线程方法
        函数的说明请见 request_llms/bridge_all.py
    """
    chatbot.append((inputs, ""))
    yield from update_ui(chatbot=chatbot, history=history)

    if validate_key() is False:
        yield from update_ui_lastest_msg(lastmsg="[Local Message] 请配置YSHS API", chatbot=chatbot, history=history, delay=0)
        return

    if additional_fn is not None:
        from core_functional import handle_core_functionality
        inputs, history = handle_core_functionality(additional_fn, inputs, history, chatbot)

    messages = generate_message(inputs, history, system_prompt)
    model = "openai/gpt-4"
    engine = 'gpt-4-1106-preview'
    result = yshs.LLM.chat(
            model=model,  # 模型名称，例如 "openai/gpt-4", 可通过 list_models.py 查看可用模型
            engine=engine,  # 引擎名称, 可选参数，例如 "gpt-4-1106-preview"
            messages=messages,  # 一个会话的消息列表
            stream=True,  # 是否以流式的方式返回结果
        )
    # 开始接收回复  
    response = ""
    for i in result: 
        response += i
        sys.stdout.write(i)
        sys.stdout.flush()
        chatbot[-1] = (inputs, response)
        yield from update_ui(chatbot=chatbot, history=history)

    # 总结输出
    if response == f"[Local Message] 等待{model_name}响应中 ...":
        response = f"[Local Message] {model_name}响应异常 ..."
    history.extend([inputs, response])
    yield from update_ui(chatbot=chatbot, history=history)

def generate_message(inputs, history, system_prompt):
    """
    整合所有信息，选择LLM模型，生成http请求，为发送请求做准备
    """

    conversation_cnt = len(history) // 2

    messages = [{"role": "system", "content": system_prompt}]
    if conversation_cnt:
        for index in range(0, 2*conversation_cnt, 2):
            what_i_have_asked = {}
            what_i_have_asked["role"] = "user"
            what_i_have_asked["content"] = history[index]
            what_gpt_answer = {}
            what_gpt_answer["role"] = "assistant"
            what_gpt_answer["content"] = history[index+1]
            if what_i_have_asked["content"] != "":
                if what_gpt_answer["content"] == "": continue
                if what_gpt_answer["content"] == timeout_bot_msg: continue
                messages.append(what_i_have_asked)
                messages.append(what_gpt_answer)
            else:
                messages[-1]['content'] = what_gpt_answer['content']

    what_i_ask_now = {}
    what_i_ask_now["role"] = "user"
    what_i_ask_now["content"] = inputs
    messages.append(what_i_ask_now)
    return messages