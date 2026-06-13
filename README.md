# Real-Time Traffic Anomaly Detection and AI Benchmarking Pipeline

## 1. Project Overview

This project implements a real-time traffic anomaly detection and AI benchmarking pipeline.

The system receives simulated traffic sensor data, validates incoming payloads, isolates invalid data into a dead-letter queue, performs fast local traffic-flow prediction using a LightGBM model, and selectively escalates high-risk or uncertain cases to Gemini for natural-language diagnosis.

The main goal of this project is not only to build a prediction model, but to design a reliable software pipeline that can evaluate latency, data quality handling, local-first inference, cloud AI escalation, and estimated cost savings.

## 2. System Architecture

The system follows this workflow:

```text
PEMS traffic data simulator
        ↓
FastAPI API Gateway
        ↓
Pydantic validation
        ↓
Invalid data → Dead Letter Queue
Valid data → Local LightGBM model
        ↓
Decision Layer
        ↓
Local-only handling OR Gemini escalation
        ↓
Benchmark logging
        ↓
Benchmark analyzer
        ↓
Streamlit dashboard
```

## 3. Main Components

```text
app/api_gateway.py        FastAPI traffic event endpoint
app/gateway.py            Pydantic request validation
app/dlq.py                Dead-letter queue handling
app/model_service.py      Local LightGBM model inference
app/decision_layer.py     Selective escalation logic
app/llm_client.py         Gemini cloud diagnosis client
app/pipeline_service.py   Main processing pipeline
app/dashboard.py          Streamlit dashboard

scripts/pems_sensor_client.py   PEMS traffic data simulator
benchmark_analyzer.py           Benchmark result analyzer
reset_experiment.py             Clears runtime experiment logs
train_baseline.py               Basic local model training script
train_pems03_04_test07.py       Cross-dataset LightGBM training script
run_server.py                   FastAPI server entry point
```

## 4. Data Files

The project uses PEMS traffic datasets stored in the `data/` folder:

```text
data/PEMS03.npz
data/PEMS04.npz
data/PEMS07.npz
```

The current local model is trained using PEMS03 and PEMS04, and evaluated on PEMS07 as an independent test dataset.

Model evaluation outputs are saved as:

```text
data/model_metrics.json
data/model_metrics_pems03_04_to_pems07.json
```

Runtime logs such as benchmark events, DLQ logs, and LLM outputs are generated during experiments and should not be treated as fixed source files.

## 5. Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## 6. Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```

A template is provided in:

```text
.env.example
```

The real `.env` file should not be committed or submitted because it may contain private API keys.

## 7. Train the Local Model

To train the cross-dataset LightGBM model using PEMS03 and PEMS04, and evaluate it on PEMS07:

```bash
python train_pems03_04_test07.py
```

This generates:

```text
app/traffic_model.txt
data/model_metrics.json
data/model_metrics_pems03_04_to_pems07.json
```

The model predicts the next traffic flow value using recent historical flow patterns and engineered time-series features.

## 8. Run the FastAPI Backend

Start the backend server:

```bash
python run_server.py
```

The traffic event endpoint is:

```text
POST http://127.0.0.1:8000/v1/traffic/events
```

## 9. Run the Traffic Data Simulator

Open another terminal and run:

```bash
python scripts/pems_sensor_client.py --dirty-ratio 0.00 --sensors 2 --duration 60 --time-start 10 --interval 0.05 --accident --quiet
```

Example dirty-data experiment:

```bash
python scripts/pems_sensor_client.py --dirty-ratio 0.10 --sensors 2 --duration 60 --time-start 10 --interval 0.05 --accident --quiet
```

Important parameters:

```text
--dirty-ratio   Ratio of injected invalid payloads
--sensors       Number of simulated sensors
--duration      Number of time steps
--time-start    Starting time index
--interval      Sending interval between requests
--accident      Injects an artificial traffic-flow drop
--quiet         Reduces console output
```

## 10. Analyze Benchmark Results

After running the simulator, analyze the generated logs:

```bash
python benchmark_analyzer.py
```

The analyzer reports:

```text
Total requests
Valid requests
Invalid requests sent to DLQ
Dirty data interception rate
Local-only processing rate
LLM escalation rate
LLM success / cooldown / failure count
Average and percentile latency
Estimated cloud AI cost saving
```

The summary is saved to:

```text
data/benchmark_summary.json
```

## 11. Run the Dashboard

Start the Streamlit dashboard:

```bash
streamlit run app/dashboard.py
```

The dashboard displays:

```text
Real-time traffic monitoring
Benchmark results
Local model evaluation metrics
DLQ statistics
LLM escalation statistics
Estimated cost savings
```

## 12. Reset Runtime Experiment Files

Before running a new experiment, clear old runtime logs:

```bash
python reset_experiment.py
```

This removes temporary files such as:

```text
data/benchmark_events.jsonl
data/dlq.jsonl
data/live_traffic.json
data/benchmark_summary.json
```

## 13. Run Tests

Run the test suite:

```bash
pytest -q
```

The tests cover API health checks, request validation, invalid payload handling, and basic gateway behavior.

## 14. Notes

This project uses a local-first design. Most traffic events are handled by the local LightGBM model, while only high-risk or uncertain cases are escalated to Gemini.

This reduces unnecessary cloud AI usage while still allowing the system to generate natural-language explanations for important anomaly cases.
