import os
import random
import argparse

import torch
import gradio as gr
import numpy as np

import ChatTTS


def generate_seed():
    new_seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": new_seed
        }

def generate_audio_old(text, temperature, top_P, top_K, audio_seed, text_seed_input, refine_text_flag = True):
    # create a random speaker embedding
    rand_spk = chat.sample_random_speaker(audio_seed)
    
    # inital text generation parameters
    params_infer_code = {
        'spk_emb': rand_spk, 
        'temperature': temperature,
        'top_P': top_P,
        'top_K': top_K,
        }
    params_refine_text = {'prompt': '[oral_2][laugh_0][break_6]'}
    
    # refine text
    if refine_text_flag:
        torch.manual_seed(text_seed_input)
        text = chat.infer(text, 
                          skip_refine_text=False,
                          refine_text_only=True,
                          params_refine_text=params_refine_text,
                          params_infer_code=params_infer_code
                          )
    
    # generate audio
    torch.manual_seed(text_seed_input)
    wav = chat.infer(text, 
                     skip_refine_text=True, 
                     params_refine_text=params_refine_text, 
                     params_infer_code=params_infer_code
                     )
    
    audio_data = np.array(wav[0]).flatten()
    sample_rate = 24000
    text_data = text[0] if isinstance(text, list) else text
    
    temp_dir = os.environ.get('TEMP_DIR', '/tmp')
    # save the model to temporary directory
    torch.save(rand_spk, f'{temp_dir}/sound_model.pth')
    
    return [(sample_rate, audio_data), text_data, f'{temp_dir}/sound_model.pth']


# sliced actions

def create_sound_model(audio_seed):
    # create a random speaker embedding
    model = chat.sample_random_speaker(audio_seed)
    # save the model to temporary directory of gradio
    # get temporary directory
    temp_dir = os.environ.get('TEMP_DIR', '/tmp')
    # save the model to temporary directory
    torch.save(model, f'{temp_dir}/sound_model.pth')
    
    
    params_infer_code = {
        'spk_emb': model,
        'temperature': 0.00003,
        'top_P': 0.7,
        'top_K': 20,
        }
    params_refine_text = {'prompt': '[oral_2][laugh_0][break_6]'}
    # generate audio
    torch.manual_seed(2)
    wav = chat.infer("这是一段测试音频。", 
                     skip_refine_text=True, 
                     params_refine_text=params_refine_text, 
                     params_infer_code=params_infer_code
                     )
    
    audio_data = np.array(wav[0]).flatten()
    sample_rate = 24000
    
    
    return [f'{temp_dir}/sound_model.pth',(sample_rate, audio_data)]

def load_sound_model(model_name):
    return torch.load(f'speaker/{model_name}.pth')

def refine_text(text, text_seed_input, rand_spk, temperature, top_P, top_K):
    params_infer_code = {
        'spk_emb': rand_spk, 
        'temperature': temperature,
        'top_P': top_P,
        'top_K': top_K,
        }
    params_refine_text = {'prompt': '[oral_2][laugh_0][break_6]'}
    torch.manual_seed(text_seed_input)
    text = chat.infer(text, 
                        skip_refine_text=False,
                        refine_text_only=True,
                        params_refine_text=params_refine_text,
                        params_infer_code=params_infer_code
                        )
    return text[0] if isinstance(text, list) else text

def generate_audio(text, text_seed_input, speaker_model_name, temperature, top_P, top_K):
    spk = torch.load(speaker_model_name)
    params_infer_code = {
        'spk_emb': spk,
        'temperature': temperature,
        'top_P': top_P,
        'top_K': top_K,
        }
    params_refine_text = {'prompt': '[oral_2][laugh_0][break_6]'}
    # generate audio
    torch.manual_seed(text_seed_input)
    wav = chat.infer(text, 
                     skip_refine_text=True, 
                     params_refine_text=params_refine_text, 
                     params_infer_code=params_infer_code
                     )
    
    audio_data = np.array(wav[0]).flatten()
    sample_rate = 24000

    return (sample_rate, audio_data)

def main():

    with gr.Blocks() as demo:
        gr.Markdown("# ChatTTS Webui")
        gr.Markdown("ChatTTS Model: [2noise/ChatTTS](https://github.com/2noise/ChatTTS)")

        default_text = "四川美食确实以辣闻名，但也有不辣的选择。比如甜水面、赖汤圆、蛋烘糕、叶儿粑等，这些小吃口味温和，甜而不腻，也很受欢迎。"   
        gr.Markdown("## 文本输入")
        text_input = gr.Textbox(label="Input Text", lines=4, placeholder="Please Input Text...", value=default_text)
        # sound models
        gr.Markdown("## 声音模型选择")
        with gr.Row():
            audio_model_seed = gr.Number(value=300, label="Model Seed")
            generate_audio_model_seed = gr.Button("\U0001F3B2")
            generate_model_button = gr.Button("Generate Model")
        model_file = gr.File(label="Model File",file_types=["pth"])
        model_test_audio = gr.Audio(label="Test Audio")
        # refine_text
        gr.Markdown("## 规范化文本")
        with gr.Row():
            refine_seed = gr.Number(value=42, label="Text Seed")
            generate_refine_seed = gr.Button("\U0001F3B2")
            refine_button = gr.Button("Refine")
        # generate_audio
        gr.Markdown("## 语音生成")
        with gr.Row():
            audio_seed = gr.Number(value=2, label="Audio Seed")
            generate_audio_seed = gr.Button("\U0001F3B2")
            temperature_slider = gr.Slider(minimum=0.00001, maximum=1.0, step=0.00001, value=0.3, label="Audio temperature")
            top_p_slider = gr.Slider(minimum=0.1, maximum=0.9, step=0.05, value=0.7, label="top_P")
            top_k_slider = gr.Slider(minimum=1, maximum=20, step=1, value=20, label="top_K")
        generate_sound = gr.Button("Generate Audio")
        gr.Markdown("## 一键生成")
        generate_button = gr.Button("Generate All")
        audio_output = gr.Audio(label="Output Audio")
        
        generate_audio_model_seed.click(generate_seed, 
                            inputs=[], 
                            outputs=audio_model_seed)
        
        generate_model_button.click(create_sound_model, 
                                     inputs=[audio_model_seed], 
                                     outputs=[model_file, model_test_audio])
        
        generate_refine_seed.click(generate_seed,
                                      inputs=[],
                                      outputs=refine_seed)

        refine_button.click(refine_text, 
                            inputs=[text_input, refine_seed, model_file, temperature_slider, top_p_slider, top_k_slider], 
                            outputs=[text_input])
        
        generate_audio_seed.click(generate_seed, 
                                  inputs=[], 
                                  outputs=audio_seed)
        generate_sound.click(generate_audio, 
                            inputs=[text_input, audio_seed, model_file, temperature_slider, top_p_slider, top_k_slider], 
                            outputs=[audio_output])

        generate_button.click(generate_audio_old, 
                              inputs=[text_input, temperature_slider, top_p_slider, top_k_slider, audio_seed, refine_seed], 
                              outputs=[audio_output, text_input, model_file])

    parser = argparse.ArgumentParser(description='ChatTTS demo Launch')
    parser.add_argument('--server_name', type=str, default='0.0.0.0', help='Server name')
    parser.add_argument('--server_port', type=int, default=5678, help='Server port')
    parser.add_argument('--local_path', type=str, default=None, help='the local_path if need')
    args = parser.parse_args()

    print("loading ChatTTS model...")
    global chat
    chat = ChatTTS.Chat()

    if args.local_path == None:
        chat.load_models()
    else:
        print('local model path:', args.local_path)
        chat.load_models('local', local_path=args.local_path)

    demo.launch(server_name=args.server_name, server_port=args.server_port, inbrowser=True)


if __name__ == '__main__':
    main()