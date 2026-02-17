from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START
from langchain_ollama import ChatOllama

class AgentState(TypedDict):
    objective: str
    logs: Annotated[list, add]

# Initialize local Qwen for both to save memory, but with different system prompts
model = ChatOllama(model="qwen2.5-coder:7b")

def architect_node(state: AgentState):
    # Logic: High-level strategy
    prompt = "Challenge: Find a hidden flag in /challenge/task.bin. What is the strategic first step?"
    res = model.invoke(prompt)
    return {"objective": res.content, "logs": [f"Architect: {res.content[:50]}"]}

def executor_node(state: AgentState):
    # Logic: Convert strategy to a command
    prompt = f"Objective: {state['objective']}. Give me a bash command to achieve this."
    res = model.invoke(prompt)
    return {"logs": [f"Executor: Running `{res.content.strip()[:30]}...`"]}

builder = StateGraph(AgentState)
builder.add_node("architect", architect_node)
builder.add_node("executor", executor_node)
builder.add_edge(START, "architect")
builder.add_edge("architect", "executor")

graph = builder.compile()
for output in graph.stream({"objective": "", "logs": []}):
    print(output)