aicyberchallenge.com
https://aicyberchallenge.com/

 AIxCC Competition Archive | AIxCC Competition Archive
https://archive.aicyberchallenge.com/

 Research Publications and Source Code
https://team-atlanta.github.io/artifacts/

TR-Team-Atlanta.pdf
https://team-atlanta.github.io/papers/TR-Team-Atlanta.pdf

 NYU CTF Bench
https://nyu-llm-ctf.github.io/

GitHub - cooldadhacking/nix-ctf-workspace
https://github.com/cooldadhacking/nix-ctf-workspace

 GitHub - NYU-LLM-CTF/nyuctf_agents: The D-CIPHER and NYU CTF baseline LLM Agents built for NYU CTF Bench
https://github.com/NYU-LLM-CTF/nyuctf_agents

 LLM CTF Attack Challenge | CSAW
https://www.csaw.io/llm-attack-challenge

 GitHub - SWE-agent/SWE-agent: SWE-agent takes a GitHub issue and tries to automatically fix it, using your LM of choice. It can also be employed for offensive cybersecurity or competitive coding challenges. [NeurIPS 2024]
https://github.com/SWE-agent/SWE-agent

 GitHub - aliasrobotics/cai: Cybersecurity AI (CAI), the framework for AI Security
https://github.com/aliasrobotics/cai

 HackSynth: LLM Agent and Evaluation Framework for Autonomous Penetration Testing
https://arxiv.org/html/2412.01778v1

 Aikido Attack: Autonomous AI Pentests | Aikido Security
https://www.aikido.dev/attack/aipentest

[https://medium.com/seercurity-spotlight/building-your-first-cybersecurity-ai-agent-with-langgraph-d27107ac872a](https://medium.com/seercurity-spotlight/building-your-first-cybersecurity-ai-agent-with-langgraph-d27107ac872a "https://medium.com/seercurity-spotlight/building-your-first-cybersecurity-ai-agent-with-langgraph-d27107ac872a")

[https://xoxruns.medium.com/feedback-driven-iteration-and-fully-local-webapp-pentesting-ai-agent-achieving-78-on-xbow-199ef719bf01](https://xoxruns.medium.com/feedback-driven-iteration-and-fully-local-webapp-pentesting-ai-agent-achieving-78-on-xbow-199ef719bf01 "https://xoxruns.medium.com/feedback-driven-iteration-and-fully-local-webapp-pentesting-ai-agent-achieving-78-on-xbow-199ef719bf01")

[https://www.aikido.dev/blog/top-automated-penetration-testing-tools](https://www.aikido.dev/blog/top-automated-penetration-testing-tools "https://www.aikido.dev/blog/top-automated-penetration-testing-tools")

[https://github.com/cooldadhacking/nix-ctf-workspace](https://github.com/cooldadhacking/nix-ctf-workspace "https://github.com/cooldadhacking/nix-ctf-workspace")

[https://www.linkedin.com/posts/robbe-verwilghen_aikido-ai-pentest-whitepaper-2025-activity-7404556275960934401-5Fg-?utm_source=share&utm_medium=member_android&rcm=ACoAADmBbtQBDmm-LrCuSj5ol8vn5EiI5HzwMUo](https://www.linkedin.com/posts/robbe-verwilghen_aikido-ai-pentest-whitepaper-2025-activity-7404556275960934401-5Fg-?utm_source=share&utm_medium=member_android&rcm=ACoAADmBbtQBDmm-LrCuSj5ol8vn5EiI5HzwMUo "https://www.linkedin.com/posts/robbe-verwilghen_aikido-ai-pentest-whitepaper-2025-activity-7404556275960934401-5Fg-?utm_source=share&utm_medium=member_android&rcm=ACoAADmBbtQBDmm-LrCuSj5ol8vn5EiI5HzwMUo")

[https://github.com/xoxruns/deadend-cli](https://github.com/xoxruns/deadend-cli "https://github.com/xoxruns/deadend-cli")

[https://xoxruns.medium.com/feedback-driven-iteration-and-fully-local-webapp-pentesting-ai-agent-achieving-78-on-xbow-199ef719bf01](https://xoxruns.medium.com/feedback-driven-iteration-and-fully-local-webapp-pentesting-ai-agent-achieving-78-on-xbow-199ef719bf01 "https://xoxruns.medium.com/feedback-driven-iteration-and-fully-local-webapp-pentesting-ai-agent-achieving-78-on-xbow-199ef719bf01")

[https://github.com/CyberSecurityUP/NeuroSploit](https://github.com/CyberSecurityUP/NeuroSploit "https://github.com/CyberSecurityUP/NeuroSploit")

[https://github.com/steveschofield/guardian-cli-deluxe](https://github.com/steveschofield/guardian-cli-deluxe "https://github.com/steveschofield/guardian-cli-deluxe")


# tools 
| **Category**  | **Tool**                      | **Why your Agent needs it**                                                                                                    |
| ------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Web**       | `ffuf` / `sqlmap`             | LLMs are bad at brute-forcing directories. Let `ffuf` do the fast work and the agent analyze the 200/301 responses.            |
| **Pwn**       | `pwntools` / `GEF`            | Your agent shouldn't calculate stack offsets by hand. `pwntools` (Python) is the language LLMs speak best for exploit writing. |
| **Reverse**   | `radare2` / `binaryninja-api` | Instead of dumping raw bytes, the agent should use `r2` to get a JSON output of strings or functions.                          |
| **Forensics** | `binwalk` / `exiftool`        | Crucial for "hidden in plain sight" challenges. The agent needs to "see" inside files.                                         |
| **Crypto**    | `ciphey` / `RsaCtfTool`       | Automated identification of weird encodings (Base58, Morse, etc.) saves your token budget.                                     |

## agent
#### example **Dynamic Research and Development** cycle
- **Architect (Strategy):** 
	- Breaks the problem into sub-tasks (e.g., "Enumerate services," "Reverse the authentication binary").
- **Research (Discovery):** 
	- If a write-up isn't found, this node uses the `bash` tool to run `nmap`, `strings`, or `checksec`. It populates the `challenge_context` with **live data** rather than just static descriptions.
- **Executor (Tactical):** 
	- Generates and runs the exact `pwntools` or `curl` commands.
- **Verifier (Critique):** 
	- Checks the `STDOUT`. Did we get a flag? If not, why? It feeds the failure back to the Architect to pivot.