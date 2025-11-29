import os
import json
import colorama
from openai import OpenAI
from dotenv import load_dotenv
from mcp_server import TOOLS_SCHEMA, AVAILABLE_TOOLS

# 初始化环境
load_dotenv()
colorama.init(autoreset=True)

# ✅ 修改点 1：使用 DashScope Client
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)
MODEL_NAME = os.getenv("LLM_MODEL", "qwen-max")

SYSTEM_PROMPT = """
你是一个专业的金融分析助手。你的数据来源是实时的图数据库。
当用户提问时，请优先使用提供的工具查询真实数据。
如果工具返回了数据，请基于数据进行分析和回答。
请直接回答，不要输出思考过程。
"""

def chat_loop():
    print(colorama.Fore.GREEN + f"🤖 金融图谱助手已启动 (Model: {MODEL_NAME}, 输入 'exit' 退出)")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = input(colorama.Fore.BLUE + "\nUser: ")
        if user_input.lower() in ["exit", "quit"]: break
        
        messages.append({"role": "user", "content": user_input})

        try:
            # 1. 第一轮调用：让 Qwen 决定是否使用工具
            # 注意：Qwen 在兼容模式下 Tool Calling 格式与 OpenAI 基本一致
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto"
            )
            
            msg = response.choices[0].message
            messages.append(msg) 

            # 2. 检查模型是否想调用工具
            if msg.tool_calls:
                print(colorama.Fore.YELLOW + f"   ⚙️  模型决定调用工具: {len(msg.tool_calls)} 个")
                
                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    # Qwen 有时候返回的 arguments 可能是字符串形式的 JSON
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        print(colorama.Fore.RED + f"   ❌ 参数解析失败: {tool_call.function.arguments}")
                        continue
                    
                    print(colorama.Fore.YELLOW + f"   🔍 执行: {func_name}({args})")
                    
                    if func_name in AVAILABLE_TOOLS:
                        function_to_call = AVAILABLE_TOOLS[func_name]
                        try:
                            tool_result = function_to_call(**args)
                        except Exception as e:
                            tool_result = f"Error: {str(e)}"
                    else:
                        tool_result = "Error: Unknown tool"

                    # 3. 将工具结果反馈给模型
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name, # Qwen 有时需要明确 name
                        "content": str(tool_result)
                    })

                # 4. 第二轮调用：模型拿到数据后，生成最终回答
                # 这一步我们可以尝试用流式输出，体验更好
                print(colorama.Fore.GREEN + "Agent: ", end="")
                stream = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    stream=True # ✅ 开启流式输出
                )
                
                full_content = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        full_content += content
                print() # 换行
                
                # 记得把完整的回答存入历史，以便多轮对话
                messages.append({"role": "assistant", "content": full_content})

            else:
                # 没调用工具，直接回答
                print(colorama.Fore.GREEN + f"Agent: {msg.content}")

        except Exception as e:
            print(colorama.Fore.RED + f"❌ 发生错误: {e}")

if __name__ == "__main__":
    chat_loop()