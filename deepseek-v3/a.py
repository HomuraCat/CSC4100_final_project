import requests

def chat_completion(prompt, model="deepseek-ai/DeepSeek-V3"):
    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer sk-htcwbomvaolnachojkjyfujzpoolaelxptjsdlqzdjfvkayf",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"API调用异常: {str(e)}")
        return None

# 调用示例
result = chat_completion("用Python实现快速排序")
print(result)