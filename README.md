# Event-Context Eye-Tracking Analysis for ASD Classification

This repository provides the implementation of the event-context-based eye-tracking analysis framework described in the accompanying manuscript. It includes AOI processing, behavioral feature extraction, Random Forest classification, evaluation, and evidence-grounded behavioral reporting.

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Repository Contents

- `src/asd_gaze/`: AOI processing, feature extraction, classification, evaluation, and behavioral reporting.
- `configs/`: Event definitions and final feature metadata.
- `supplementary/`: Summary statistics for the 157 candidate features.
- `examples/`: Synthetic demonstration.
- `tests/`: Unit and integration tests.

## Usage

Run the synthetic demonstration:

```bash
PYTHONPATH=. python examples/synthetic_demo.py
```

Run the tests:

```bash
python -m unittest discover -s tests
```

The optional language-generation component uses Qwen2.5-32B through a separately configured Ollama service.

## Data Availability

Participant-level eye-tracking data, stimulus videos, frame-level AOI annotations, and trained model weights are not included. The repository provides the methods and synthetic examples without the original study data.