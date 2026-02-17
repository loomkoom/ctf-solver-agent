from pathlib import Path

from langchain_core.messages import HumanMessage
from src.agent import graph
from src.tools.rag import ingest_writeups
from src.config import settings

def main():
    # Index the knowledge base using path from settings
    kb_path = Path(settings.knowledge_base_path).resolve()
    print(f"Indexing from {kb_path}...")
    ingest_writeups(str(kb_path))

    # 2. Define the Challenge
    challenge_text = "I have a file at /challenge/task.bin. Find the flag."
    print(f"\nStarting challenge: {challenge_text}")

    # 3. Initialize Tiered State
    # Note: 'messages' and 'logs' use Annotated(..., add) in our state.py
    # so we start them as empty lists.
    inputs = {
        "messages": [HumanMessage(content=challenge_text)],
        "challenge_context": challenge_text,
        "current_objective": "Initial Reconnaissance",
        "logs": ["System: Challenge initialized."]
    }

    config = {"recursion_limit": settings.max_iterations}

    # 4. Stream the Tiered Execution
    # Using 'updates' mode to see exactly what each node contributes
    for output in graph.stream(inputs,config=config, stream_mode="updates"):
        for node_name, node_update in output.items():
            print(f"\n{'='*20} Node: {node_name} {'='*20}")

            # Show the high-level reasoning from logs
            if "logs" in node_update:
                for log in node_update["logs"]:
                    print(f"LOG: {log}")

            # Show the objective if the Architect updated it
            if "current_objective" in node_update:
                print(f"New Objective: {node_update['current_objective']}")

            # Show the last message (tool output or model thought)
            if "messages" in node_update:
                last_msg = node_update["messages"][-1]
                # If it's a HumanMessage, it's usually a tool result
                prefix = "Tool Result" if isinstance(last_msg, HumanMessage) else "AI Thought"
                print(f"{prefix}: {last_msg.content[:500]}...")

if __name__ == "__main__":
    main()