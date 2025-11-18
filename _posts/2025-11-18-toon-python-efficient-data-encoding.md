---
title: TOON Python - Efficient Data Encoding for Large Language Models
date: 2025-11-18 14:30:00 +0400
categories: [AI, Development]
tags: [python, llm, optimization, data-encoding]
author: aadhil
description: "Real-world benchmark comparison of Python and Go clients for LLM APIs using Groq's ultra-fast inference service. Discover which language wins for speed and consistency."
image:
  path: /assets/img/posts/llm-benchmark-cover.png
  alt: "Python vs Go performance comparison for LLM API clients"
math: false
mermaid: true
render_with_liquid: false
---

Working with Large Language Models (LLMs) can get expensive, especially when you're constantly sending structured data back and forth. Every JSON object, every curly brace, every comma—they all add up in token costs. What if you could cut those costs by 30-60% without changing how your application works?

Enter **TOON** (Token-Oriented Object Notation), a compact data format specifically designed for AI applications that dramatically reduces token usage while maintaining readability and structure.

## What is TOON?

TOON (Token-Oriented Object Notation) is a compact, human-readable format designed for passing structured data to Large Language Models with significantly reduced token usage (30-60% reduction compared to JSON).

Think of it as JSON's efficient cousin—it delivers the same information but strips away all the redundant syntax that makes JSON so verbose. TOON combines YAML's indentation-based structure for nested objects and CSV's tabular format for uniform data rows, optimized specifically for token efficiency in LLM contexts.

## Why TOON Matters

When you're building AI applications, token efficiency isn't just about saving money—it's about performance too. Here's what token savings translate to:

- **Lower API costs** — Fewer tokens = lower bills from OpenAI, Anthropic, or other LLM providers
- **Faster processing** — Less data to process means quicker responses
- **Larger context windows** — You can fit more actual data in the same context window

A real-world test showed compelling results: With a dataset of employees, asking GPT to calculate average salary by department, TOON delivered roughly a 56% reduction in prompt tokens and a noticeable 5-second speed improvement.

## How TOON Works: JSON vs TOON

Let's see the difference with a simple example.

**Standard JSON:**

```json
{
  "users": [
    { "id": 1, "name": "Alice", "role": "admin" },
    { "id": 2, "name": "Bob", "role": "user" },
    { "id": 3, "name": "Charlie", "role": "editor" }
  ]
}
```

**TOON Format:**

```
users[3,]{id,name,role}:
1,Alice,admin
2,Bob,user
3,Charlie,editor
```

Notice how TOON:
- Eliminates braces, brackets, and most quotes
- Declares field names once instead of repeating them
- Uses CSV-style rows for tabular data
- Still remains perfectly readable

## Getting Started with Python

Installing TOON for Python is straightforward:

```bash
pip install python-toon
```

Or if you prefer the official community-driven implementation:

```bash
pip install toon_format
```

## Basic Usage Examples

### Simple Object Encoding

```python
from toon_python import encode

data = {"name": "Alice", "age": 30}
toon_output = encode(data)
print(toon_output)
```

**Output:**

```
name: Alice
age: 30
```

### Tabular Data (Where TOON Shines)

```python
from toon_python import encode

users = [
    {"id": 1, "name": "Alice", "age": 30},
    {"id": 2, "name": "Bob", "age": 25},
    {"id": 3, "name": "Charlie", "age": 35}
]

toon_output = encode(users)
print(toon_output)
```

**Output:**

```
[3,]{id,name,age}:
1,Alice,30
2,Bob,25
3,Charlie,35
```

### Decoding Back to Python

```python
from toon_python import decode

toon_string = """
name: Alice
age: 30
"""

python_data = decode(toon_string)
print(python_data)  # {'name': 'Alice', 'age': 30}
```

## Real-World Use Cases

TOON is particularly useful for:

- **Training data** — Less token overhead for structured training data to fine-tune LLMs
- **Agent frameworks** — Compact data exchange in Agent frameworks
- **MCP workflows** — Faster data serialization between the MCP and AI workflow engines
- **Serverless APIs** — Cost and speed optimization for serverless AI APIs

### Example: LLM Prompt with Employee Data

```python
import toon_python

employees = [
    {"id": 1, "name": "Alice", "department": "Engineering", "salary": 120000},
    {"id": 2, "name": "Bob", "department": "Marketing", "salary": 95000},
    {"id": 3, "name": "Charlie", "department": "Engineering", "salary": 110000}
]

# Convert to TOON for efficient LLM input
toon_data = toon_python.encode(employees)

# Create your prompt
prompt = f"""Analyze this employee data:

{toon_data}

What's the average salary by department?"""

# Send to your LLM (fewer tokens, lower cost!)
```

## Performance and Benchmarks

TOON achieves 73.9% accuracy (vs JSON's 69.7%) while using 39.6% fewer tokens. Not only are you saving tokens, but LLMs actually comprehend TOON data slightly better than JSON in many cases.

> The benchmark results show that TOON isn't just more efficient—it's also more effective for LLM comprehension.
{: .prompt-tip }

## When to Use TOON (and When Not To)

**Use TOON when:**
- Sending structured, tabular data to LLMs
- Working with uniform arrays of objects
- Token costs are a concern
- You need faster LLM processing

**Stick with JSON when:**
- Your data is deeply nested or irregular
- You need strict schema validation
- You're working with non-AI APIs
- The data structure varies significantly between objects

> A hybrid approach may work best: Keep JSON for your application's data exchange format with APIs, but convert to TOON when sending data to LLMs.
{: .prompt-info }

## Advanced Features

The `python-toon` package supports:

- Nested structures
- Multiple data types (datetime, Decimal, UUID, binary data)
- Custom encoding options (indentation, delimiters)
- Array length markers for validation

## The Bottom Line

TOON isn't trying to replace JSON everywhere—it's a specialized tool for a specific problem: efficiently passing data to LLMs. If you're building AI applications that frequently communicate with language models, especially with structured or tabular data, TOON can significantly reduce your costs and improve performance.

The setup takes minutes, the API is simple, and the savings are real. In a world where every token literally costs money, TOON gives you a competitive edge.

Ready to optimize your AI workflows? Install TOON for Python today and start building more cost-effective AI applications.

```bash
pip install python-toon
```

## Further Reading

- [Official TOON Specification](https://github.com/toon-format/toon)
- [Python Implementation](https://github.com/xaviviro/python-toon)
- [TOON Format Documentation](https://mer.vin/2025/11/toon-python-efficient-data-encoding-for-large-language-models/)
