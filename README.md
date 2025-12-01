# AI Graduate Advisor

AI-assistant that uses faculty profile texts + embeddings to help prospective graduate students find advisors and information.

## Repo layout
- generate_embeddings.py — build embeddings from faculty_texts.json
- build_profile_texts.py — text extraction / cleaning pipeline
- response_engine.py — retrieval + prompt/LLM orchestration
- chatbot_demo.py — a demo script showing how to run the assistant
- data/ — supporting data and assets
- embeddings.npy — (large binary; recommended to remove from repo and store externally / via git-lfs)

## Quick start
1. Create and activate a virtual environment:
   python -m venv .venv
   source .venv/bin/activate

2. Install dependencies:
   pip install -r requirements.txt

3. Create an .env file with your API keys (do not commit .env)
   OPENAI_API_KEY=...

4. Generate embeddings:
   python generate_embeddings.py --input faculty_texts.json --output embeddings.npy

5. Run the demo:
   python chatbot_demo.py

## Testing
- Run tests with pytest:
  pytest

## Notes & suggestions
- Consider storing embeddings outside the repo (S3 or Git LFS).
- Add CI to run linting and tests automatically.
- Modularize response_engine.py into smaller modules for retrieval, embedding, and LLM interface.
