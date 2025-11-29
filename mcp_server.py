# mcp_server.py
from database import run_query

# --- 工具函数定义 ---

def get_latest_market_events(limit: int = 5):
    """
    【工具】获取最近发生的财经事件。
    当用户问“最近有什么新闻”、“发生了什么”时调用。
    """
    cypher = """
    MATCH (e:Event)
    WHERE e.timestamp IS NOT NULL
    // 按时间倒序
    WITH e ORDER BY e.timestamp DESC LIMIT $limit
    MATCH (ent:Entity)-[r:PARTICIPATES_IN]->(e)
    RETURN e.timestamp as time, e.type as type, collect(ent.name + '(' + r.role + ')') as entities
    """
    results = run_query(cypher, {"limit": limit})
    if not results: return "暂无最近数据。"
    return str(results)

def search_entity_history(entity_name: str):
    """
    【工具】查询某个公司或人物的历史事件。
    当用户问“马云最近干了啥”、“宁德时代有什么动作”时调用。
    """
    cypher = """
    MATCH (ent:Entity)-[r:PARTICIPATES_IN]->(e:Event)
    WHERE ent.name CONTAINS $name
    RETURN e.timestamp as time, e.type as type, e.trigger as action, r.role as role
    ORDER BY e.timestamp DESC LIMIT 10
    """
    results = run_query(cypher, {"name": entity_name})
    if not results: return f"未找到关于 {entity_name} 的记录。"
    return str(results)

def find_relationships(entity1: str, entity2: str):
    """
    【工具】查找两个实体之间的关联路径。
    当用户问“A和B有什么关系”时调用。
    """
    cypher = """
    MATCH p = shortestPath((e1:Entity {name: $n1})-[*]-(e2:Entity {name: $n2}))
    RETURN [n in nodes(p) | labels(n)[0] + ': ' + coalesce(n.name, n.type)] as path
    """
    results = run_query(cypher, {"n1": entity1, "n2": entity2})
    if not results: return f"未找到 {entity1} 和 {entity2} 之间的直接关联。"
    return str(results)

# 工具注册表：让 LLM 知道有哪些工具可用
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_latest_market_events",
            "description": "获取最新的财经新闻事件列表，了解市场动态。",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "返回的数量，默认5"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_entity_history",
            "description": "查询特定实体（公司、人）的过往事件和动态。",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "实体名称，如'茅台'"}
                },
                "required": ["entity_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_relationships",
            "description": "分析两个实体之间的关系路径。",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity1": {"type": "string"},
                    "entity2": {"type": "string"}
                },
                "required": ["entity1", "entity2"],
            },
        },
    }
]

# 工具映射表：用于实际执行
AVAILABLE_TOOLS = {
    "get_latest_market_events": get_latest_market_events,
    "search_entity_history": search_entity_history,
    "find_relationships": find_relationships
}