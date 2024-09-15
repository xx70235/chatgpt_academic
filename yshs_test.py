import os, sys
import yshs
yshs.api_key = os.getenv('YSHS_API_KEY')

def request_model():
    responese = yshs.LLM.chat(
        model="openai/gpt-3.5-turbo",  # 选择模型
        messages=[
          {"role": "system", "content": "You are a helpful assistant."},  # 系统提示
          {"role": "user", "content": "Who won the world series in 2020?"},  # 第一个问题
          # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},  # 第一个的答案
          # {"role": "user", "content": "Where was it played?"}  # 第二个问题
      ]
    )

    full_response = ""
    for x in responese:
        sys.stdout.write(x)  # 逐token输出
        sys.stdout.flush()
        full_response += x
    print()
    return full_response

answer = request_model()
print(answer)