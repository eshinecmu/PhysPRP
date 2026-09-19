import os, base64
from openai import OpenAI

def _client_and_model(default_model):
    key = os.getenv('LAB_LLM_API_KEY') or os.getenv('OPENAI_API_KEY')
    base = os.getenv('LAB_LLM_BASE_URL') or os.getenv('OPENAI_BASE_URL')
    model = os.getenv('LAB_LLM_MODEL') or os.getenv('OPENAI_MODEL') or default_model
    if not key: raise RuntimeError('Set LAB_LLM_API_KEY or OPENAI_API_KEY')
    kwargs = {'api_key': key}
    if base: kwargs['base_url'] = base
    return OpenAI(**kwargs), model

def Qwen(image_path, prompt):
    data = base64.b64encode(open(image_path, 'rb').read()).decode()
    client, model = _client_and_model('qwen-vl-max-latest')
    r = client.chat.completions.create(model=model, messages=[{'role':'user','content':[{'type':'image_url','image_url':{'url':f'data:image/jpeg;base64,{data}'}},{'type':'text','text':prompt}]}])
    return r.choices[0].message.content

def GPT4V(image_path, prompt):
    data = base64.b64encode(open(image_path, 'rb').read()).decode()
    client, model = _client_and_model('gpt-4o-mini')
    r = client.chat.completions.create(model=model, messages=[{'role':'user','content':[{'type':'text','text':prompt},{'type':'image_url','image_url':{'url':f'data:image/jpeg;base64,{data}'}}]}])
    return r.choices[0]


def get_image_files(directory):
    image_files = []
    # Iterate over the directory
    for i in range(1, 2):  # Modify range if more folders need to be processed
        sub_path = str(i).zfill(2)
        now_path = os.path.join(directory, sub_path)
        all_png = os.listdir(now_path)
        for png in all_png:
            img_path = os.path.join(now_path, png)
            image_files.append(img_path)

    image_files = sorted(image_files, key=lambda x: (x.split('/')[3], x.split('/')[4]))

    return image_files




