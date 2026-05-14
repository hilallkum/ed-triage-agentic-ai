# ED Triage Agentic AI System

A hybrid emergency department triage system that combines rule-based RPA and an LLM agent to support clinical decision-making with a human-in-the-loop at every step.

## Overview

Manual ED triage relies entirely on nurse judgement with no decision support. This project implements two complementary automation layers using the Manchester Triage System (MTS):

- **RPA layer** (`rpa_triage.py`) — processes structured vital signs (HR, BP, SpO2, Temp, GCS) through deterministic MTS scoring rules
- **AI Agent layer** (`agent_triage.py`) — sends free-text complaints and clinical notes to GPT-4o-mini, returning a suggested MTS level, identified red flags, and clinical reasoning
- **Human-in-the-loop** — nurse reviews and confirms or overrides both outputs before anything is logged

## Why Two Layers?

RPA is fast and consistent for structured data but fails on natural language. It cannot handle negation ("no chest pain" vs "chest pain") or reason about symptom combinations. The LLM agent handles what rule-based systems cannot.

The key case: Patient P001 (67-year-old, chest pain radiating to left arm, BP 90, diaphoresis). RPA assigned Level 2 based on individual vital thresholds. The agent assigned Level 1, recognising the combination as a classic ACS pattern. Neither approach alone is sufficient.

## System Architecture

```
Patient Arrival
      │
      ├── RPA Layer ──────────────── Vital sign scoring → MTS Level + Risk Flag
      │
      └── AI Agent Layer ─────────── Free-text analysis → Level + Red Flags + Reasoning
                    │
                    └── Human-in-the-Loop ── Nurse confirms or overrides → Audit Log
```

## Results on Test Dataset (10 synthetic patients)

- 3 Level 1 (Immediate) cases correctly flagged by RPA
- AI agent correctly escalated P001 from Level 2 → Level 1 based on ACS pattern recognition
- Full audit trail written to `triage_output.xlsx` with timestamps and override status

## Files

```
├── rpa_triage.py          # Rule-based vital sign scoring (MTS)
├── agent_triage.py        # LLM agent via OpenAI API
├── patient_arrivals.xlsx  # Synthetic test dataset (10 patients)
└── triage_output.xlsx     # Output with triage levels, risk flags, timestamps
```

## Setup

```bash
pip install pandas openpyxl openai
export OPENAI_API_KEY=your_key_here
python rpa_triage.py
python agent_triage.py
```

> **Note:** The AI agent requires an OpenAI API key. Without one, it runs in mock demo mode and returns pre-defined responses for demonstration purposes.

## Stack

Python · OpenAI API (GPT-4o-mini) · Pandas · OpenPyXL · Manchester Triage System (MTS)

