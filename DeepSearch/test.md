# DeepSearch
## Search-o1

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

### Training template
```
Think ---> Search ---> Knowledge ---> Answer
 ^                          |
 |                          |
 |__________________________|
```