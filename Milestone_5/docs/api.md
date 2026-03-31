# API Reference — CodeXcelerate

## Endpoints

### GET /
Returns welcome message.

### GET /health
Returns service health status.

### POST /evaluate
Evaluates LLM response quality.
**Body:** `{"question": "...", "answer": "...", "context": "..."}`

### POST /guardrails/check
Validates text through all 3 NeMo Guardrails rails.
**Body:** `{"text": "..."}`

### GET /pipeline-status
Returns pipeline component status.
