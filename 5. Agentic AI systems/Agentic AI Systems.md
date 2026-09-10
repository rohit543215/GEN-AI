# Agentic AI Systems — Complete Guide

---

# 1. AGENT ARCHITECTURES

## 1.1 What Agents Are and Why They Matter

**Agentic AI systems** are autonomous systems that use LLMs as reasoning engines to pursue goals, make decisions, and take actions in dynamic environments. Unlike standard chatbots that respond once, agents operate in **loops**: perceive → reason → act → observe → repeat .

**The core insight:** An LLM alone is a reasoning engine with no agency. An agent wraps the LLM with:
- **Tools** to interact with the world
- **Memory** to retain context across steps
- **Planning** to break down complex goals
- **Reflection** to learn from mistakes

## 1.2 The ReAct Pattern (Reasoning + Acting)

**The breakthrough that made agents practical:** ReAct alternates between **reasoning** (thinking about what to do next) and **acting** (executing tools and observing results). This interleaving allows the agent to adapt its plan based on new information .

**The loop:**
```
Thought: I need to find the current weather in Paris
Action: search(query="weather in Paris today")
Observation: "Paris: 18°C, sunny"
Thought: I have the weather, now I can answer
Answer: "The weather in Paris is 18°C and sunny."
```

**Why this works:** By interleaving reasoning and action, the agent can adjust its plan mid-stream. If the search returns unexpected results, the agent can change its approach .

## 1.3 Agent Types (2026)

| Architecture | Description | Best For |
|--------------|-------------|----------|
| **ReAct** | Simple thought-action-observation loop | Most tasks, easy to implement |
| **Plan-and-Execute** | Plan entire sequence first, then execute step by step | Complex multi-step tasks |
| **Reflection Agents** | Generate output, then critique and improve it | Writing, code generation |
| **Plan-Solve** | Plan multiple steps before acting, then execute | Complex reasoning |
| **LLM-Compiler** | Generate parallel execution plans | Tasks with independent subtasks |

## 1.4 Code: Basic ReAct Agent

```python
import openai

class ReActAgent:
    def __init__(self, tools, llm):
        self.tools = {t.name: t for t in tools}
        self.llm = llm
        self.max_steps = 10

    def run(self, query: str):
        """Execute the ReAct loop"""
        context = f"Question: {query}\n"
        
        for step in range(self.max_steps):
            # Generate thought and action
            prompt = self._build_prompt(context)
            response = self.llm(prompt)
            
            # Parse response
            thought, action, action_input = self._parse(response)
            context += f"Thought: {thought}\n"
            
            if action == "Final Answer":
                return action_input
            
            # Execute tool
            tool = self.tools.get(action)
            if tool:
                observation = tool.run(action_input)
                context += f"Action: {action}({action_input})\n"
                context += f"Observation: {observation}\n"
            else:
                context += f"Error: Tool {action} not found\n"
        
        return "Max steps reached"
```


---

# 2. TOOL USE / FUNCTION CALLING

## 2.1 What It Is

**Function calling** is the mechanism by which agents interact with external systems. The LLM outputs structured calls to functions (tools) rather than free-form text, making execution deterministic and safe .

## 2.2 How It Works

1. **Define tools** with schema (name, description, parameters)
2. **LLM decides** which tools to call based on the query
3. **Execute** the tool with the LLM-provided arguments
4. **Return results** back to the LLM for further reasoning

**Function schema (OpenAI format):**
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["city"]
            }
        }
    }
]
```

## 2.3 Code: Function Calling

```python
# OpenAI Function Calling
import openai

def get_weather(city: str) -> str:
    # Implementation
    return f"Weather in {city}: 18°C, sunny"

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }
    }
}]

response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto"
)

if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    result = get_weather(**args)
```

## 2.4 Tool Categories

| Category | Examples | When to Use |
|----------|----------|-------------|
| **APIs** | Weather, stocks, calendar | External data |
| **Internal** | Database queries, internal APIs | Company data |
| **Search** | Web search, document search | Knowledge retrieval |
| **Computation** | Python, calculator | Math, logic |
| **Memory** | Save/retrieve info | Across conversations |
| **Action** | Send email, create ticket | Execution |



---

# 3. MULTI-AGENT ORCHESTRATION

## 3.1 Why Multiple Agents

**Single agent limitation:** One agent struggles with complex tasks requiring different expertise, long context windows, or parallel work .

**Multi-agent advantages:**
- **Specialization:** Each agent has a role (e.g., researcher, writer, critic)
- **Parallel execution:** Independent subtasks run concurrently
- **Self-correction:** Agents critique each other's work
- **Context management:** Each agent maintains its own context

## 3.2 Orchestration Patterns

| Pattern | Description | Best For |
|---------|-------------|----------|
| **Hierarchical** | Manager delegates to workers | Large tasks with clear sub-tasks |
| **Peer-to-Peer** | Agents collaborate freely | Creative tasks, brainstorming |
| **Sequential** | Agents process in order | Pipelines (research → write → edit) |
| **Swarm** | Many agents dynamically interact | Complex, open-ended tasks |
| **Debate** | Agents argue opposing positions | Verifying facts, decision-making |


## 3.3 Code: Multi-Agent (LangGraph)

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    messages: list
    research: str
    draft: str
    final: str

# Define agents
def researcher(state: AgentState) -> AgentState:
    # Research topic
    return {"research": "Research findings..."}

def writer(state: AgentState) -> AgentState:
    # Write based on research
    return {"draft": "Draft based on research..."}

def editor(state: AgentState) -> AgentState:
    # Edit the draft
    return {"final": "Edited final version..."}

# Build graph
builder = StateGraph(AgentState)
builder.add_node("researcher", researcher)
builder.add_node("writer", writer)
builder.add_node("editor", editor)

# Define flow
builder.add_edge("researcher", "writer")
builder.add_edge("writer", "editor")
builder.add_edge("editor", END)

builder.set_entry_point("researcher")
graph = builder.compile()
```


---

# 4. FRAMEWORKS

## 4.1 LangChain

**What it is:** The original agent framework — provides abstractions for chains, agents, memory, tools, and document loaders.

**Best for:** Quick prototyping, broad ecosystem, ease of use.

**Core Concepts:**
- **Chains:** Sequences of operations
- **Agents:** LLMs that choose actions
- **Tools:** Functions agents can call
- **Memory:** State management between calls

**Code:**
```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a city"""
    return f"{city}: 18°C, sunny"

tools = [get_weather]
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
result = executor.invoke({"input": "Weather in Paris?"})
```

## 4.2 LangGraph

**What it is:** Built on LangChain but designed for **graph-based stateful agents** — better for complex, multi-step workflows that require loops, branching, and state management.

**Best for:** Production agents with complex state and control flow.

**Key Features:**
- **Graph-based architecture:** Nodes (functions) and edges (transitions)
- **Stateful:** Full control over state across steps
- **Looping:** Built-in support for cycles
- **Human-in-the-loop:** Breakpoints, interrupt, resume

**Code:**
```python
from langgraph.graph import StateGraph, MessageGraph
from langgraph.checkpoint import MemorySaver

# Agent with memory/checkpointing
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# Invoke with thread_id for persistence
config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke({"messages": [HumanMessage("Hello")]}, config)

# Resume from breakpoint
result = graph.invoke(None, config)  # Continues where left off
```

## 4.3 AutoGen

**What it is:** Microsoft's framework for **conversational multi-agent systems** — agents talk to each other in natural language.

**Best for:** Multi-agent collaboration, research, complex reasoning.

**Key Features:**
- **Conversational agents:** Agents communicate via messages
- **Agent roles:** Assistant, UserProxy, custom agents
- **Group chat:** Multiple agents in a round-robin conversation
- **Tool use:** Built-in support for function calling

**Code:**
```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Define agents
assistant = AssistantAgent("assistant", llm_config=llm_config)
user_proxy = UserProxyAgent("user_proxy", human_input_mode="NEVER")

# Group chat
group_chat = GroupChat(agents=[assistant, user_proxy], messages=[])
manager = GroupChatManager(groupchat=group_chat)

# Start conversation
user_proxy.initiate_chat(manager, message="Plan a trip to Paris")
```


## 4.4 Framework Decision Guide

| Framework | Best For | When to Choose |
|-----------|----------|----------------|
| **LangChain** | Quick prototyping, simple agents | You need to build something fast |
| **LangGraph** | Production agents, complex state | You need loops, branching, human-in-loop |
| **AutoGen** | Multi-agent collaboration | Your agents need to talk to each other |


---

# 5. MEMORY SYSTEMS FOR AGENTS

## 5.1 Why Memory Matters

Agents without memory are stateless — they forget everything between turns . Memory enables:
- **Short-term:** Current conversation context
- **Long-term:** Facts about the user across sessions
- **Procedural:** Learned patterns and preferences

## 5.2 Memory Types

| Memory Type | Scope | Storage | Best For |
|-------------|-------|---------|----------|
| **Short-term** | Current conversation | In-context | Immediate reasoning |
| **Long-term** | Across sessions | Vector DB | User preferences, facts |
| **Working** | Current task | Short-term | Step-by-step execution |
| **Episodic** | Past interactions | Vector DB | Learning from experience |

## 5.3 Memory Implementation

**Short-term memory (in-context):**
```python
messages = [SystemMessage(prompt)]
while conversation:
    messages.append(HumanMessage(user_input))
    response = llm(messages)
    messages.append(AIMessage(response))
    messages = messages[-10:]  # Sliding window
```

**Long-term memory (vector DB):**
```python
class MemorySystem:
    def __init__(self, vector_db):
        self.memory = vector_db
        self.embedder = get_embedding

    def remember(self, user_id: str, content: str, type: str):
        """Store a memory"""
        vector = self.embedder(content)
        self.memory.insert({
            "user_id": user_id,
            "text": content,
            "type": type,  # "preference", "fact", "event"
            "vector": vector,
            "timestamp": now()
        })

    def recall(self, user_id: str, query: str, limit: int = 5):
        """Retrieve relevant memories"""
        vector = self.embedder(query)
        return self.memory.search({
            "vector": vector,
            "filter": {"user_id": user_id},
            "limit": limit
        })
```

## 5.4 Memory Strategies

| Strategy | When to Use | Trade-offs |
|----------|-------------|------------|
| **Sliding window** | Small, focused tasks | Low cost, limited context |
| **Summarization** | Long conversations | Good compression, loses detail |
| **RAG retrieval** | Large knowledge bases | Accurate, requires vector DB |
| **Structured memory** | User preferences | Easy to query, limited flexibility |
| **Episodic memory** | Learning user patterns | Rich, complex to implement |



---

# FULL COMPARISON TABLE

| Component | Options | Best For | Trade-off |
|-----------|---------|----------|-----------|
| **Architecture** | ReAct, Plan-and-Execute, Plan-Solve, Reflection | Most tasks, complex planning, multi-step, self-improvement | Simplicity vs capability |
| **Tools** | APIs, Internal, Search, Computation, Memory | External data, company data, knowledge, math, persistence | Range vs complexity |
| **Orchestration** | Hierarchical, Peer-to-Peer, Sequential, Swarm | Sub-tasks, creativity, pipelines, exploration | Control vs flexibility |
| **Framework** | LangChain, LangGraph, AutoGen | Prototyping, production state, multi-agent | Ease vs control |
| **Memory** | Short-term, Long-term, Episodic, Working | Current context, across sessions, learning, task state | Cost vs capability |

---

# QUICK DECISION RULES

1. **Start with:** ReAct + LangChain + basic tools + short-term memory
2. **Scale up to:** LangGraph when you need loops, branching, or human-in-loop
3. **Use AutoGen** when agents need to collaborate via conversation
4. **Add long-term memory** when users return across sessions
5. **Plan-and-Execute** for complex multi-step tasks with clear sub-goals
6. **Always implement tool calling** properly with schema and error handling

---

# 5 MOST-ASKED AGENT INTERVIEW QUESTIONS

**1. What is the ReAct pattern and why is it important?** ReAct interleaves Reasoning and Acting in a loop: Thought → Action → Observation → Repeat. It's important because it allows agents to adapt plans based on new information, unlike pure reasoning or pure acting approaches.

**2. What's the difference between LangChain and LangGraph?** LangChain is for simple linear chains and basic agents. LangGraph is built for complex, stateful agents with control flow — loops, branching, and conditional transitions — making it better for production agent systems.

**3. How do you implement memory in agents?** Use short-term memory via message windows/summarization for immediate context, and long-term memory via vector databases for facts and user preferences across sessions. Episodic memory stores past interactions for learning.

**4. What is function calling and why is it important?** Function calling allows LLMs to output structured calls to external tools rather than free-form text. It's important because it makes agent actions deterministic, safe, and verifiable.

**5. What's the difference between tool calling and function calling?** They're often used interchangeably. Tool calling is the broader concept — any capability an agent can use. Function calling specifically refers to the structured mechanism where the LLM outputs a function name and arguments in a standardized format.