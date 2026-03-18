Here’s your **edited version** — clean, professional, and easy to read (perfect for GitHub README or submission):

---

## **Day 5 NLP: DataAssist Analytics Agent**

**Author:** Sheshikala Mamidisetti
**Week:** 4 (Mar 15–24)
**Topic:** Local LLMs via Ollama & Prompt Engineering for Data Analytics
**Dataset:** Titanic passenger data (via seaborn)

---

## **Overview**

This project extends Day 5 NLP work with Week 4 concepts, focusing on **local Large Language Models (LLMs)** and **advanced prompt engineering** for data analytics tasks using the Titanic dataset.

I used **Ollama** to run **Llama3** and **Mistral** locally, eliminating API costs and internet dependency. Ollama runs a local server (`localhost:11434`) and provides an API similar to OpenAI. Once a model is downloaded, it can be used instantly offline.

---

## **What I Built**

###  Llama3 Pipeline

* Built a complete data analysis pipeline:

  * Loads Titanic dataset
  * Generates statistical summaries
  * Produces:

    * Natural language insights
    * Survival predictions
    * Structured report sections
* Strength: Detailed, multi-paragraph outputs (ideal for reports)

---

###  Mistral Pipeline

* Designed for fast, concise responses:

  * Quick summaries
  * Key metrics
  * Recommendations
* Built a **comparison module**:

  * Runs same prompt on both models
  * Compares:

    * Response time
    * Output length

---

###  Prompt Engineering Techniques

#### 1. Chain of Thought (CoT)

* Forces step-by-step reasoning:

  * Identify data
  * Analyze
  * Interpret
  * Conclude
* Improves accuracy for analytical questions

---

#### 2. ReAct (Reason + Act)

* Combines reasoning with tool usage
* Model decides:

  * What to think
  * What function to call

**Tools implemented:**

* Survival rate
* Descriptive statistics
* Value counts
* Correlation
* Group comparison

---

#### 3. DSPy-style Prompting

* Modular prompt system using **Signatures**
* Implemented (without dspy package):

  * `Predict`
  * `ChainOfThought`
* Demonstrated:

  * Few-shot vs Zero-shot prompting
  * Impact on output quality and style

---

## **Special Feature**

All files include **mock fallback responses**, so:

* Code runs even without Ollama
* Automatically switches to real model output when Ollama is active

---

## **Project Files**

| File                  | Description                                                 |
| --------------------- | ----------------------------------------------------------- |
| `ollama_setup.py`     | Ollama setup, connection check, model listing, benchmarking |
| `llama_pipeline.py`   | Llama3 analysis: insights, predictions, reports             |
| `mistral_pipeline.py` | Mistral: summaries, metrics, comparison                     |
| `cot_prompting.py`    | Chain of Thought prompting                                  |
| `react_prompting.py`  | ReAct agent with tool execution                             |
| `dspy_pipeline.py`    | DSPy-style modular prompting                                |
| `requirements.txt`    | Dependencies                                                |
| `README.md`           | Documentation                                               |

---

## **Key Concepts Learned**

| Concept                    | File                  |
| -------------------------- | --------------------- |
| Ollama local LLM setup     | `ollama_setup.py`     |
| Llama3 report generation   | `llama_pipeline.py`   |
| Mistral concise outputs    | `mistral_pipeline.py` |
| Chain of Thought reasoning | `cot_prompting.py`    |
| ReAct agent with tools     | `react_prompting.py`  |
| DSPy modular prompting     | `dspy_pipeline.py`    |
| Few-shot vs Zero-shot      | `dspy_pipeline.py`    |
| Model benchmarking         | `mistral_pipeline.py` |

---

## **Dataset**

* Titanic dataset loaded via **seaborn**
* If seaborn is unavailable:

  * Automatically uses a **20-row fallback dataset**
* Ensures code runs in any environment without errors

