# Real-Time Traffic Anomaly Detection and AI Benchmarking Pipeline

This project is a capstone prototype for a real-time traffic anomaly detection and AI benchmarking pipeline. The system uses a FastAPI ingestion gateway, Pydantic schema validation, a Dead Letter Queue, a local LightGBM model, a decision layer, selective Gemini LLM escalation, benchmark logging, and a Streamlit dashboard.

## Project Architecture

The pipeline follows a local-first architecture:

1. Simulated traffic events are generated from the PEMS03 dataset.
2. FastAPI receives incoming traffic payloads.
3. Pydantic validates the payload structure and value ranges.
4. Invalid payloads are stored in the Dead Letter Queue.
5. Valid payloads are passed to the local LightGBM model.
6. The decision layer decides whether the event should be handled locally or escalated to Gemini.
7. Local and LLM results are logged for benchmarking.
8. Streamlit dashboard visualises live traffic and benchmark metrics.

## Main Components

```text
app/
  api_gateway.py       FastAPI ingestion endpoint
  gateway.py           Pydantic validation schema
  dlq.py               Dead Letter Queue logic
  model_service.py     Local LightGBM inference
  decision_layer.py    Local-first escalation rules
  llm_client.py        Gemini API client
  pipeline_service.py  Main processing pipeline
  dashboard.py         Streamlit dashboard

scripts/
  pems_sensor_client.py  Sends simulated PEMS03 traffic events
  sensor_client.py       Basic sensor client

data/
  PEMS03.npz             Traffic dataset

run_server.py            Starts FastAPI server
train_baseline.py        Trains local LightGBM baseline model
benchmark_analyzer.py    Analyses benchmark logs
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a local `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Then add your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
```

Do not commit `.env`.

## Train the Local Model

Run:

```bash
python train_baseline.py
```

This creates or updates the local traffic model used by the pipeline.

## Run the FastAPI Server

Run:

```bash
python run_server.py
```

The API server should start locally.

Health check:

```text
http://127.0.0.1:8000/health
```

## Run the PEMS Sensor Client

In another terminal, run:

```bash
python scripts/pems_sensor_client.py
```

This sends simulated traffic events to the FastAPI gateway.

## Run the Dashboard

In another terminal, run:

```bash
streamlit run app/dashboard.py
```

## Run Benchmark Analysis

After generating benchmark logs, run:

```bash
python benchmark_analyzer.py
```

## Notes

This project is a benchmarking prototype, not a production deployment. The current focus is to evaluate validation, DLQ isolation, local-first inference, selective LLM escalation, latency, throughput, and estimated API cost.

## Experiment Workflow

This section describes the recommended workflow for running the prototype and generating benchmark results.

### 1. Run the FastAPI Server

```bash
python run_server.py
The server exposes the traffic ingestion API at:

http://127.0.0.1:8000/v1/traffic/events
Health check:

http://127.0.0.1:8000/health
2. Train the Local LightGBM Model
python train_baseline.py
This trains the local LightGBM model and generates:

app/traffic_model.txt
reports/model_metrics.json
3. Run Unit Tests
pytest -q
The tests cover:

valid traffic payloads

invalid timestamps

numeric strings

unexpected extra fields

malformed JSON

validation gateway rejection logic

4. Run a Small Validation Test
python scripts/pems_sensor_client.py --dirty-ratio 0.1 --sensors 2 --duration 20 --interval 0.1
This sends traffic events to the FastAPI gateway and injects controlled dirty payloads.

Expected behaviour:

valid payloads return HTTP 200

invalid payloads return HTTP 400 or 422

rejected payloads are written to the DLQ

5. Run an Accident / Escalation Test
python scripts/pems_sensor_client.py --dirty-ratio 0.1 --sensors 2 --duration 20 --time-start 10 --interval 0.1 --accident
This enables a controlled flow-drop scenario to test the decision layer and selective LLM escalation.

6. Run Benchmark Analysis
python benchmark_analyzer.py
This reads the benchmark logs and generates:

reports/benchmark_summary.json
The summary includes:

total requests

DLQ rejections

local pipeline latency

local model latency

LLM escalation rate

actual cloud call rate

estimated cloud cost saving

7. Run the Streamlit Dashboard
streamlit run app/dashboard.py
The dashboard contains three main sections:

Live Monitoring

Benchmark Summary

Local Model Metrics

8. Suggested Dirty-Data Experiment Matrix
python scripts/pems_sensor_client.py --dirty-ratio 0.00 --sensors 2 --duration 60 --interval 0.05 --accident --quiet
python scripts/pems_sensor_client.py --dirty-ratio 0.05 --sensors 2 --duration 60 --interval 0.05 --accident --quiet
python scripts/pems_sensor_client.py --dirty-ratio 0.10 --sensors 2 --duration 60 --interval 0.05 --accident --quiet
python scripts/pems_sensor_client.py --dirty-ratio 0.20 --sensors 2 --duration 60 --interval 0.05 --accident --quiet
These experiments can be used to evaluate how the validation gateway and DLQ behave under different dirty-data ratios.