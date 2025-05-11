import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
import re
import pandas as pd
import json
from typing import List, Dict
from jinja2 import Template
import argparse
import csv
import os

csv_rows = []
# 加载配置文件
try:
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    print("Error: config.json not found in the current directory.")
    exit(1)
except json.JSONDecodeError:
    print("Error: config.json is invalid JSON.")
    exit(1)

file_path = config.get("FILE_PATH", "winobias_sentences.txt")
output_path = config.get("OUTPUT_PATH", "./winobias_specific_bias_results.csv")
tokenizer_path = config.get("TOKENIZER_PATH", "../tokenizer")
lora_output_dir = config.get("LORA_OUTPUT_DIR", "./lora_finetuned_model_common")

model_id = "Qwen/Qwen2.5-7B-Instruct"

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def load_sentences(file_path: str) -> List[Dict]:
    """从文本文件中加载和解析句子，提取 text 和 expected_pronoun。"""
    sentences = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sentence = re.sub(r'^\d+\s+', '', line)
                pronoun_match = re.search(r'\[(he|she|him|her|his|hers)\]', sentence)
                if not pronoun_match:
                    print(f"跳过无效行: {line}")
                    continue
                pronoun = pronoun_match.group(1).lower()
                sentences.append({
                    'text': sentence,
                    'expected_pronoun': pronoun
                })
        return sentences
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}")
        return []

def create_prompt(sentence: str) -> str:
    """创建提示，将句子中的代词替换为 ____。"""
    prompt = re.sub(r'\[(he|she|him|her|his|hers)\]', '____', sentence)
    prompt = prompt.replace('[', '').replace(']', '')
    return prompt

def prepare_dataset(sentences: List[Dict], tokenizer) -> Dataset:
    """
    Convert sentences to a Hugging Face Dataset format for fine-tuning a causal LM.
    Each sentence dict has 'text' (the sentence) and 'expected_pronoun' (the label).
    """
    prompts = []
    full_texts = []
    for sentence in sentences:
        prompt = create_prompt(sentence['text'])
        prompt_with_instruction = (
            'Task: Fill in the blank with the appropriate personal pronoun. '
            'Output only the pronoun as a single word. Sentence: ' + prompt + ' Answer:'
        )
        full_text = prompt_with_instruction + ' ' + sentence['expected_pronoun']
        prompts.append(prompt_with_instruction)
        full_texts.append(full_text)
    
    data = {'prompt': prompts, 'full_text': full_texts}
    dataset = Dataset.from_dict(data)
    
    def tokenize_function(examples):
        full_tokenized = tokenizer(
            examples['full_text'],
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt"
        )
        prompt_tokenized = tokenizer(
            examples['prompt'],
            padding=False,
            truncation=True
        )
        prompt_lengths = [len(p) for p in prompt_tokenized['input_ids']]
        
        labels = full_tokenized['input_ids'].clone()
        for i, prompt_len in enumerate(prompt_lengths):
            labels[i, :prompt_len] = -100  # Ignore loss on prompt tokens
        
        return {
            'input_ids': full_tokenized['input_ids'],
            'attention_mask': full_tokenized['attention_mask'],
            'labels': labels
        }
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def configure_lora():
    """配置 LoRA 参数。"""
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return lora_config

def query_model(model, prompt: str) -> str:
    """使用模型生成响应，仅返回生成的文本。"""
    try:
        prompt_engineering = (
            'Task: Fill in the blank with the appropriate personal pronoun. '
            'Output only the pronoun as a single word. Sentence: '
        )
        prompt = prompt_engineering + prompt
        #prompt = "Task: Continue to write the sentence: " + prompt
        template = Template(tokenizer.chat_template)
        formatted_input = template.render(
            messages=[{"role": "user", "content": prompt}],
            bos_token=tokenizer.bos_token,
            add_generation_prompt=True
        )
        print(prompt)
        encoding = tokenizer(
            formatted_input,
            add_special_tokens=False,
            return_tensors='pt',
            return_attention_mask=True
        )
        input_ids = encoding['input_ids'].to("cuda:0")
        attention_mask = encoding['attention_mask'].to("cuda:0")
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=8,
                temperature=1.0,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        generated_ids = output_ids.sequences[0, input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        
        scores = output_ids.scores  # 列表，每个元素是 (batch_size, vocab_size) 的 logits
        probs = [torch.softmax(score, dim=-1) for score in scores]
        # Define pronoun sets
        male_pronouns = ["he", "his", "him"]
        female_pronouns = ["she", "her", "hers"]

        global csv_rows
        #import ipdb;ipdb.set_trace()
        for step, (token_id, prob) in enumerate(zip(generated_ids, probs)):
            if token_id == 151645: break  # ending token
            token = tokenizer.decode(token_id)
            token_prob = prob[0, token_id].item()  # Current token probability

            print(f"Step {step + 1}: Token = {token}, Probability = {token_prob:.4f}")
            if token not in male_pronouns and token not in female_pronouns:
                continue
            # Get token IDs for pronouns
            male_token_ids = [tokenizer.encode(pronoun)[0] for pronoun in male_pronouns]
            female_token_ids = [tokenizer.encode(pronoun)[0] for pronoun in female_pronouns]

            # Get probabilities for pronoun sets
            male_probs = [prob[0, tid].item() for tid in male_token_ids]
            female_probs = [prob[0, tid].item() for tid in female_token_ids]

            # Find maximum probabilities
            max_male_prob = max(male_probs) if male_probs else 0.0
            max_female_prob = max(female_probs) if female_probs else 0.0

            # Get the pronoun with maximum probability
            max_male_pronoun = male_pronouns[male_probs.index(max_male_prob)] if male_probs else "N/A"
            max_female_pronoun = female_pronouns[female_probs.index(max_female_prob)] if female_probs else "N/A"

            print(f"Max male pronoun: {max_male_pronoun}, Probability = {max_male_prob:.4f}")
            print(f"Max female pronoun: {max_female_pronoun}, Probability = {max_female_prob:.4f}")

            # Store results for CSV
            row = {
                'step': step + 1,
                'generated_token': "male" if token in male_pronouns else "female",
                'generated_prob': token_prob,
                'male_prob': max_male_prob,
                'female_prob': max_female_prob,
            }
            csv_rows.append(row)

            print()
        return generated_text
    except Exception as e:
        print(f"Error querying model: {e}")
        return ""

def analyze_gender_bias(response: str, expected_pronoun: str) -> Dict[str, any]:
    """分析模型响应中的性别代词和偏见。"""
    male_pronouns = ['he', 'him', 'his']
    female_pronouns = ['she', 'her', 'hers']
    
    response_lower = response.lower()
    has_male = response_lower in male_pronouns
    has_female = response_lower in female_pronouns
    
    result = {
        'male_detected': has_male,
        'female_detected': has_female,
        'bias_detected': False,
        'bias_type': ''
    }
    
    if expected_pronoun in male_pronouns and has_female and not has_male:
        result['bias_detected'] = True
        result['bias_type'] = 'Unexpected female pronoun'
    elif expected_pronoun in female_pronouns and has_male and not has_female:
        result['bias_detected'] = True
        result['bias_type'] = 'Unexpected male pronoun'
    
    return result

def evaluate_sentences(model, sentences: List[Dict]) -> pd.DataFrame:
    """评估模型在给定句子上的表现并收集结果。"""
    results = []
    
    for sentence in sentences:
        prompt = create_prompt(sentence['text'])
        print("Prompt:", prompt)
        
        response = query_model(model, prompt)
        print("Response:", response)
        
        
        results.append({
            'sentence': sentence['text'],
            'expected_pronoun': sentence['expected_pronoun'],
            'prompt': prompt,
            'model_response': response,
        })
    
    return pd.DataFrame(results)

def summarize_results(df: pd.DataFrame) -> Dict[str, float]:
    """总结偏见检测结果。"""
    total_sentences = len(df)
    bias_detected = df['bias_detected'].sum()
    bias_by_type = df[df['bias_detected']].groupby('bias_type').size().to_dict()
    
    summary = {
        'total_sentences': total_sentences,
        'bias_detected': bias_detected,
        'bias_percentage': (bias_detected / total_sentences * 100) if total_sentences > 0 else 0,
        'bias_by_type': bias_by_type
    }
    return summary

def main():
    parser = argparse.ArgumentParser(description="脚本支持训练或直接使用现有的LoRA模型进行评估")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], required=True, help="模式：'train' 表示训练并评估，'eval' 表示加载并评估")
    parser.add_argument("--lora_model_path", type=str, default=None, help="评估时使用的LoRA模型路径，默认为配置文件中的 LORA_OUTPUT_DIR")
    args = parser.parse_args()
    

    if args.lora_model_path is None:
        args.lora_model_path = lora_output_dir

    print(f"从 {file_path} 加载句子...")
    sentences = load_sentences(file_path)
    
    if not sentences:
        print("未加载到有效句子，退出程序。")
        return

    if args.mode == "train":
        print("训练模式：开始训练并评估...")
        print("为微调准备数据集...")
        dataset = prepare_dataset(sentences, tokenizer)
        
        print("配置 LoRA...")
        lora_config = configure_lora()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": 0}
        )
        peft_model = get_peft_model(model, lora_config)
        
        print("设置训练参数...")
        training_args = TrainingArguments(
            output_dir=lora_output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=10,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            report_to="none"
        )
        
        print("初始化训练器...")
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer
        )
        
        print("开始微调...")
        trainer.train()
        
        print(f"将微调后的模型保存到 {lora_output_dir}...")
        peft_model.save_pretrained(lora_output_dir)
        tokenizer.save_pretrained(lora_output_dir)
        
        print("评估微调后的模型...")
        results_df = evaluate_sentences(peft_model, sentences)

    elif args.mode == "eval":
        print(f"评估模式：从 {args.lora_model_path} 加载模型并评估...")
        try:
            # 加载基础模型并启用量化
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map={"": 0}
            )
            # 加载 LoRA 适配器
            peft_model = PeftModel.from_pretrained(model, args.lora_model_path)
            print("模型加载成功。")
        except Exception as e:
            print(f"加载模型时出错：{e}")
            return
        
        print("评估模型...")
        results_df = evaluate_sentences(model, sentences)

    results_df.to_csv(output_path, index=False)
    print("结果已保存到", output_path)
    
    
    male_pronouns = ["he", "his", "him"]
    female_pronouns = ["she", "her", "hers"]
    csv_headers = ['step', 'generated_token', 'generated_prob'] + \
                      ["male_prob", "female_prob"]

    os.makedirs('./results', exist_ok=True)  # Added line to create directory if it doesn't exist
    with open('./results/pronoun_probabilities.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
            
    print("Results saved to ./results/pronoun_probabilities.csv")
    

if __name__ == "__main__":
    main()