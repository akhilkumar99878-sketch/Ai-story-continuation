# StoryForge AI — AI Story Continuation System

Local-first NLP mini-project using DistilGPT2 and PyTorch.

## Features
- Genres: Fantasy, Dark Fantasy, Mystery, Horror, Romance, Science Fiction, Adventure and General
- Short, Medium and Long continuation
- Creativity slider (temperature)
- Separate story opening and generated continuation
- Recent-generation history during the browser session
- Download complete story as TXT
- CUDA GPU inference when available
- Optional TorchServe handler

## Setup & Run Instructions

1. Activate your Python virtual environment (e.g. on Windows):
```powershell
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

3. Run the application:
```powershell
streamlit run app.py
```

Run `streamlit run app.py` inside the virtual environment after `pip install -r requirements.txt`. Open http://localhost:8501 in your browser. The first launch downloads DistilGPT2 and requires an internet connection; subsequent runs use the local Hugging Face cache.

## Model limitations
DistilGPT2 is a small general-purpose model, not a story-specialized model. Genre prompts and sampling settings can influence style but cannot guarantee coherent plots or polished prose. Review generated text.

## TorchServe
A handler and config are included in `torchserve/`. TorchServe is in limited maintenance; validate compatibility in Linux/WSL2 or a compatible container. The handler is provided as a project component but has not been packaged or cloud-deployed by this ZIP.

## AWS requirement
The original assignment brief names Amazon SageMaker as a required deployment component. This updated package runs locally without AWS, but a local-only submission may require faculty approval or an approved AWS account/credits. No billable AWS resources are created by this project.

## Tests
```powershell
pip install pytest
pytest -q
```

## Suggested report testing
Test at least 10 story openings, empty/long input, all three lengths, multiple genres, and low/high creativity. Record outputs and observations. For formal compliance, capture TorchServe and SageMaker evidence if your faculty requires them.
