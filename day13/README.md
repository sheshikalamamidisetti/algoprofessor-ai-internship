# **TimeSeriesHunter — QLoRA Llama 3.1**

**Milestone 7 | IIT Indore AI & Data Science Programme**
**Shashikala Mamidisetti | April 2026** 

---

## **Overview**

**TimeSeriesHunter** fine-tunes Llama 3.1 (8B) into a specialised AI for data science and time series analysis using an industry-standard LLM engineering pipeline. 

---

## **Pipeline**

**QLoRA → SFT → DPO → Quantisation → vLLM Deployment → W&B Evaluation**

---

## **Key Technologies**

* **Llama 3.1 (8B)**
* **QLoRA & LoRA (PEFT)**
* **TRL (SFTTrainer & DPO)**
* **AWQ/GPTQ Quantisation**
* **vLLM**
* **Weights & Biases (W&B)**
* **Hugging Face Transformers**

---

## **Folder Structure**

```
day13/
├── lora_qlora_setup.py
├── sft_trainer.py
├── dpo_preference_tuning.py
├── quantisation_vllm.py
├── wandb_experiment_tracking.py
├── tests/test_day13.py
├── notebooks/timeserieshunter_demo.ipynb
├── outputs/
├── requirements.txt
└── README.md
```

---

## **How to Run**

### **Local Setup**

```bash
cd day13
pip install -r requirements.txt
wandb login
huggingface-cli login
```

### **Execute the Pipeline**

```bash
python lora_qlora_setup.py
python sft_trainer.py --dry-run
python dpo_preference_tuning.py --dry-run
python quantisation_vllm.py --mode demo
python wandb_experiment_tracking.py --milestone
pytest tests/ -v
```

---

## **Deliverables**

* Fine-tuned LoRA adapters
* Quantised deployment-ready model
* Milestone evaluation reports
* Experiment tracking logs
* Demo notebook and unit tests

---

## **Evaluation Tasks**

* Trend Analysis
* Seasonality Detection
* ARIMA Model Selection
* Anomaly Detection
* Forecast Evaluation 

**Pass Criteria:** ≥ 70% average score. 

---

## **IIT Indore Curriculum Alignment**

| Component           | Module                                     |
| ------------------- | ------------------------------------------ |
| Time Series Tasks   | Time Series Analysis                       |
| QLoRA & LoRA        | Deep Learning                              |
| SFT                 | Supervised Learning                        |
| DPO                 | Reinforcement Learning from Human Feedback |
| Quantisation & vLLM | Model Deployment                           |
| W&B Tracking        | ML Experiment Management                   |

---

## **Key Outcomes**

* Specialised LLM for data science and time series analysis
* Efficient fine-tuning using QLoRA
* Optimised deployment through quantisation and vLLM
* Reproducible experiments using W&B



