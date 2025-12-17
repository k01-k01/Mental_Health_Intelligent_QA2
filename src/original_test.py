# original_test.py
# 测试原生模型的Gradio脚本

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
from threading import Thread
import torch

def chat_response(message, history):
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **model_inputs,
        streamer=streamer,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.7,
    )
    # 在单独的线程中运行生成
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    # 流式输出
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield generated_text
    # 等待生成完成
    thread.join()

if __name__ == '__main__':
    # 加载模型和分词器
    model_name = "D:/project/qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 创建 Gradio 界面
    demo = gr.ChatInterface(
        fn=chat_response,
        chatbot=gr.Chatbot(height="75vh"),
        textbox=gr.Textbox(placeholder="输入你的问题...", container=False, scale=7),
        title="Qwen2.5-0.5B-Instruct原生模型效果测试",
        description="小毕超：模型蒸馏实验"
    )
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)