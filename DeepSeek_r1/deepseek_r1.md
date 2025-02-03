![20250202211854.png](image/20250202211854.png)
## 1. Load Balancing Strategy(MoE)
- high-load experts

The high-load experts are detected based on statistics collected during the online deployment and are adjusted periodically (e.g., every 10 minutes)

- redundant experts(dynamic)

Redundant experts are copys of high-load experts. In every GPU(total 32), except for 8 experts, there is a redundant expert to make sure every GPU is processing similar amount of token.

## 2. Multi-head Latent Attention
The core of MLA is the low-rank joint compression for attention keys and values to reduce Key-Value (KV) cache during inference.

## 3. Pipeline Parallelism
### MoE Parallelism
- **computational and communication tasks**

In this parallelism strategy, while one GPU is executing computational tasks (such as the forward pass, backpropagation, or any other computation), it can simultaneously receive data from another GPU.

![20250202205001.png](image/20250202205001.png)

## 4. Multi_token Prediction

When input is t2, t3, t4, t5,  the model will predict t6 and t7 simutaneously.

![20250202205605.png](image/20250202205605.png)