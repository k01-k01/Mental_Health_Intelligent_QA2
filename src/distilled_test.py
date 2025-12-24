# distilled_test.py
# 测试蒸馏后模型的Gradio脚本
 
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
from threading import Thread
import torch
import time   
 
model_name = "./output/best"
token_name = "D:/project/qwen2.5-0.5B-Instruct"
# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(token_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
 
def chat_response(message, history):
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **model_inputs,
        streamer=streamer,
        max_new_tokens=768,
        do_sample=True,
        temperature=0.7,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
 
    generated_text = ""
    last_yield = time.time()          # ← 新增
 
    for new_text in streamer:
        generated_text += new_text
 
        if "<|" in generated_text:
            generated_text = generated_text.replace("<|", "<|")
        if "|>" in generated_text:
            generated_text = generated_text.replace("|>", "|>")
 
        # ← 下面这几行是唯一改动的地方（控制刷新频率）
        now = time.time()
        if now - last_yield > 0.05:       # 调整为 20 次/秒，更稳健处理超长输出
            yield generated_text
            last_yield = now
            time.sleep(0.002)             # 稍长 sleep，防止浏览器节流
 
    # 最后强制再吐一次，防止最后几百字被吞
    yield generated_text                  
 
    thread.join()
    print("\n【回答完毕】\n")
    time.sleep(0.15)
    yield generated_text
    time.sleep(0.1)
    yield generated_text
 
 
custom_chatbot = gr.Chatbot(height="75vh")
demo = gr.ChatInterface(
    fn=chat_response,
    chatbot=custom_chatbot,
    textbox=gr.Textbox(placeholder="输入你的问题...", container=False, scale=7),
    title="心理医疗健康智能助手",
    description="模型蒸馏实验"
)
demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
