# DeepSearch
## Search-o1
Search-O1 introduces a structured search-enhanced generation process that coordinates when to search and how to incorporate search results — without changing the model parameters.
![](image/1.png)

## Search-R1
### Intro
Search-R1 is an extension of reinforcement learning (RL) for reasoning frameworks where the LLM learns to autonomously generate (multiple) search queries during step-by-step reasoning with real-time retrieval.

In 2025, DeepSeek-R1 demonstrated that pure **reinforcement learning** can cultivate the self-verification ability of LLMs, but without integrating external retrieval. This paper proposes extending the reinforcement learning framework to a retrieval-augmented scenario.

### Key Concept
1. Enhanced Reasoning with Search
- This search is **dynamic**, meaning the model can decide in real-time whether it needs additional data, and then retrieve it.
2. Application Areas
- Search-R1 is particularly useful for tasks in **scientific research, knowledge-intensive applications, programming, and mathematical reasoning**, where knowledge is constantly evolving or too large to be entirely stored in a model's parameters.

### RL
![](image/2.png)
### Training template
```
Think ---> Search ---> Knowledge ---> Answer
 ^                          |
 |                          |
 |__________________________|
```
![](image/3.png)

## MaskSearch
![](image/4.png)

### 1. RAMP Task: Retrieval-Augmented Masked Prediction
**Core idea**: Instead of simple masked language modeling, RAMP requires the model to actively search for the information needed to fill masked parts in sentences.
```
Andrew Barto received his [mask] with distinction in [mask] from the University of Michigan in 1970.
```

### 2. Multi-Agent Collaboration for SFT
To generate high-quality chain-of-thought (CoT) data for supervised fine-tuning (SFT), MaskSearch builds a multi-agent system:

1. Planner: Determines masked items to search.

2. Rewriter: Crafts effective search queries.

3. Observer: Reads and extracts facts.

4. Teacher: Validates correctness.

### 3. Reinforcement Learning with DAPO
After SFT, MaskSearch uses reinforcement learning (RL) to further refine the model using Dynamic-Aware Policy Optimization (DAPO), which optimizes two reward types:

**Format reward**: Ensures the output structure matches expected formats.

**Answer reward**: Evaluates correctness vs. the ground truth (using large-instruction models like Qwen2.5-72B-Instruct)

### 4. Curriculum Learning
MaskSearch introduces difficulty-based curriculum learning:

- Prompts contain varying numbers of masked tokens.

- The model starts with simpler tasks (fewer masks) and gradually tackles more complex ones (multiple masks), improving stability and reasoning capability.
