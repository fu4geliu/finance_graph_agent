import time
import json
import datetime
import os
import re
import torch
import akshare as ak
from openai import OpenAI
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from database import run_query

load_dotenv()

# ✅ 云端模型配置：使用 DashScope
cloud_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)
CLOUD_MODEL_NAME = os.getenv("LLM_MODEL", "qwen-max")

# ✅ 本地模型配置
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "")
COMPLEXITY_THRESHOLD = float(os.getenv("COMPLEXITY_THRESHOLD", "0.55"))  # 复杂度阈值

# 全局模型变量（懒加载）
_local_tokenizer = None
_local_model = None

def get_realtime_news():
    """获取财联社电报"""
    try:
        print("📡 正在拉取最新电报...")
        df = ak.stock_telegraph_cls(symbol="全部")
        if df.empty: return []
        return df.head(3)['content'].tolist()
    except Exception as e:
        print(f"❌ 获取新闻失败: {e}")
        return []

def score_news_complexity(news_text):
    """根据长度、句子数、实体/数字数打分，用于后续路由决策。"""
    if not news_text:
        return 0.0
    length_score = min(len(news_text) / 600, 1.2)
    sentence_score = min(len(re.split(r"[。！？!?]", news_text)) / 8, 1.0)
    entity_matches = re.findall(r"[A-Za-z0-9\u4e00-\u9fa5]{2,}", news_text)
    unique_entities = len(set(entity_matches))
    entity_score = min(unique_entities / 50, 1.0)
    digit_score = min(len(re.findall(r"\d+", news_text)) / 10, 1.0)
    score = (0.4 * length_score) + (0.25 * sentence_score) + (0.25 * entity_score) + (0.1 * digit_score)
    return round(min(score, 1.0), 3)

def parse_events_from_text(text):
    """尝试从模型输出中解析 JSON 列表。"""
    if not text:
        return []
    json_block = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    candidate = json_block.group(1) if json_block else text
    try:
        events = json.loads(candidate)
        if isinstance(events, list):
            return events
    except json.JSONDecodeError:
        pass
    return []

def events_valid(events):
    """简单校验事件列表是否满足最基本结构。"""
    if not isinstance(events, list) or not events:
        return False
    for event in events:
        if not isinstance(event, dict):
            return False
        if "event_type" not in event or "trigger" not in event or "arguments" not in event:
            return False
    return True

def initialize_local_model():
    """初始化本地微调模型（懒加载，只初始化一次）。"""
    global _local_tokenizer, _local_model
    
    if _local_tokenizer is not None and _local_model is not None:
        return _local_tokenizer, _local_model
    
    if not LOCAL_MODEL_PATH or not os.path.exists(LOCAL_MODEL_PATH):
        print("⚠️ 本地模型路径未配置或不存在，将只使用云端模型")
        return None, None
    
    try:
        print("🔄 正在加载本地微调模型...")
        _local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
        _local_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=False,
            use_safetensors=True
        )
        print("✅ 本地模型加载完成！")
        return _local_tokenizer, _local_model
    except Exception as e:
        print(f"❌ 本地模型加载失败: {e}，将只使用云端模型")
        return None, None

def extract_with_local_model(news_text, tokenizer, model):
    """使用本地微调模型抽取事件。"""
    prompt = f"""
你是一个金融新闻信息抽取助手。请只输出 JSON 数组，每个元素包含：
- "event_type"
- "trigger"
- "arguments": 包含"主体"、"客体"、"时间"、"地点"等键，缺失字段可省略。
新闻内容：
{news_text}

严格输出 JSON（勿输出其它解释）。
""".strip()
    try:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        if isinstance(inputs, torch.Tensor):
            inputs = {"input_ids": inputs}
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.2,
                top_p=0.9
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        events = parse_events_from_text(response.strip())
        if events_valid(events):
            print("   ✅ 使用本地模型抽取成功")
            return events
        print("   ⚠️ 本地模型输出未通过校验，准备升级为云端模型")
        return []
    except Exception as e:
        print(f"   ⚠️ 本地模型抽取异常: {e}，切换到云端模型")
        return []

def extract_with_cloud_model(news_text):
    """使用云端大模型（DashScope）抽取事件。"""
    prompt = f"""
从以下新闻中提取关键金融事件。返回JSON列表，每个元素包含:
- event_type (如: 投资, 涨停, 罚款, 收购)
- trigger (触发词)
- arguments (字典, 包含: 主体, 客体, 金额, 原因等)

新闻: {news_text}

注意：请直接返回纯 JSON 数组，不要包含 Markdown 格式（如 ```json ... ```）。
"""
    try:
        response = cloud_client.chat.completions.create(
            model=CLOUD_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        content = response.choices[0].message.content
        # 清洗可能存在的 Markdown 标记
        content = content.replace("```json", "").replace("```", "").strip()
        events = parse_events_from_text(content)
        if events_valid(events):
            print("   ✅ 使用云端模型抽取成功")
            return events
        print("   ❌ 云端模型响应未解析出有效 JSON")
        return []
    except json.JSONDecodeError:
        print(f"   ⚠️ JSON 解析失败: {content[:50] if 'content' in locals() else 'unknown'}...")
        return []
    except Exception as e:
        print(f"   ❌ 云端模型抽取失败: {e}")
        return []

def extract_events(text):
    """根据复杂度在本地模型与云端模型之间路由。"""
    complexity = score_news_complexity(text)
    print(f"   📊 文本复杂度得分: {complexity:.3f} (阈值: {COMPLEXITY_THRESHOLD})")
    
    # 优先使用本地模型处理简单新闻
    tokenizer, model = initialize_local_model()
    prefer_cloud = complexity >= COMPLEXITY_THRESHOLD
    
    if not prefer_cloud and tokenizer is not None and model is not None:
        events = extract_with_local_model(text, tokenizer, model)
        if events_valid(events):
            return events
        print("   ⏫ 本地模型处理失败，切换到云端模型...")
    
    # 复杂新闻或本地模型失败时使用云端模型
    events = extract_with_cloud_model(text)
    if events_valid(events):
        return events
    
    print("   ❌ 事件提取全部失败")
    return []

def save_to_neo4j(events):
    """存入图数据库 (保持不变)"""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for event in events:
        e_type = event.get("event_type", "未知")
        trigger = event.get("trigger", "未知")
        args = event.get("arguments", {})
        
        cypher_event = """
        MERGE (e:Event {type: $type, trigger: $trigger})
        ON CREATE SET e.timestamp = $time, e.created_at = timestamp()
        ON MATCH SET e.last_seen = $time
        RETURN elementId(e) as id
        """
        run_query(cypher_event, {"type": e_type, "trigger": trigger, "time": current_time})
        
        for role, name in args.items():
            if not name or not isinstance(name, str): continue
            # 简单的实体名清洗
            name = name.replace('"', '').replace("'", "")
            
            cypher_rel = """
            MATCH (e:Event {type: $type, trigger: $trigger})
            MERGE (ent:Entity {name: $name})
            MERGE (ent)-[:PARTICIPATES_IN {role: $role}]->(e)
            """
            run_query(cypher_rel, {"type": e_type, "trigger": trigger, "name": name, "role": role})
            print(f"   ✅ 关系入库: ({name})--[{role}]-->({e_type})")

def main_loop():
    processed_hashes = set()
    model_info = f"云端: {CLOUD_MODEL_NAME}"
    if LOCAL_MODEL_PATH and os.path.exists(LOCAL_MODEL_PATH):
        model_info += f" | 本地: {os.path.basename(LOCAL_MODEL_PATH)}"
    print(f"🚀 后台采集服务已启动 ({model_info})...")
    print(f"📌 复杂度阈值: {COMPLEXITY_THRESHOLD} (>=阈值使用云端模型)")
    
    # 预加载本地模型（如果配置了）
    if LOCAL_MODEL_PATH:
        initialize_local_model()
    
    while True:
        news_list = get_realtime_news()
        for news in news_list:
            h = hash(news)
            if h in processed_hashes: continue
            
            print(f"\n📰 处理新闻: {news[:30]}...")
            events = extract_events(news)
            if events:
                save_to_neo4j(events)
            processed_hashes.add(h)
        
        print("💤 等待 60 秒...")
        time.sleep(60)

if __name__ == "__main__":
    main_loop()