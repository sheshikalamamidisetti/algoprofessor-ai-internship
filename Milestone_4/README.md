# Milestone 4 — InsightScribe
**AlgoProfessor AI Internship | Sheshikala Mamidisetti | Batch 2026**

## Milestone: M4 — InsightScribe

## Pipeline
```
Upload Audio → Whisper Transcription → Speaker Diarisation
→ 3-Way LLM Summary (Groq FREE + Claude + GPT-4o)
→ PDF Report + Analytics Chart → FastAPI REST Service
```

## Week 5 Requirements Covered
| Requirement | Status |
|------------|--------|
| Whisper audio AI for data briefings 
| AI-assisted data report generation 
| FastAPI + Docker REST analytics service 
| LLM evaluation (3-way comparison) 

## How to Run
1. Open `InsightScribe_Milestone4_Day26.ipynb` in Google Colab
2. Run ALL cells top to bottom
3. Get FREE Groq key from **console.groq.com**
4. Click the public Gradio URL
5. Upload audio → click Process Meeting → see results!

## Folder Structure
```
Milestone_4/
├── data/              — input audio files
├── output/
│   ├── diarised_transcript.json
│   ├── llm_results.json
│   ├── meeting_analytics.png
│   └── meeting_report.pdf
├── utils/
│   ├── __init__.py
│   ├── audio_processor.py
│   ├── llm_comparator.py
│   └── pdf_generator.py
├── InsightScribe_Milestone4_Day26.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Tech Stack
OpenAI Whisper | Groq FREE (Llama3-70B) | Gradio | FastAPI | Pydantic | ReportLab | Docker

## IIT Indore Alignment
- Large Language Models and APIs
- Structured Output Generation
- REST API Development
- Audio Processing and NLP

