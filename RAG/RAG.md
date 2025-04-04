## RAG
### Light RAG
![image-20250201124929335](image/1.png)

`D(.)` : Deduplication

`P(.)` : Profiling

`R(.)` : Relationship Extraction

![image-20250201124929335](image/2.png)

Entities and relationships are stored in a graph structure for fast and semantically rich retrieval.

Nodes = concepts/entities, edges = relations

![image-20250201124929335](image/3.png)

Given a **user query**, the system retrieves both:

- **Low-Level Keys**: Directly mentioned terms (e.g., Beekeeper, Honey Bee, Hive).

- **High-Level Keys**: LLM-generated abstract concepts (e.g., Agriculture, Production, Environmental Impact).