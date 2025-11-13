# Mini Reasoning LLM

A full-stack playground that pairs a FastAPI backend, a tiny PyTorch-powered reasoning model, and a modern React interface. Ask it small arithmetic or pattern questions and it answers with short chain-of-thought style explanations—perfect for demos, hack days, or learning how LLM interfaces are wired together.

## 🚀 Overview

- **What it is:** A production-ready example of an "LLM-like" experience where a handcrafted mini transformer-style model (implemented with PyTorch GRU encoder/decoder) reasons about small numerical tasks.
- **What it can do:**
  - Handle prompts about single-digit arithmetic (addition, subtraction).
  - Recognize parity (even/odd), numeric ordering, and successor questions.
  - Produce lightweight chain-of-thought responses such as “first compute … therefore …”.
- **What it cannot do (yet):**
  - Understand open-ended natural language beyond the curated tasks.
  - Work with large numbers or complex multi-step problems.
  - Replace real LLMs—this is a teaching/demo project!

## 🧰 Tech Stack

- **Backend:** Python 3.11, FastAPI, Uvicorn, PyTorch
- **Model:** Custom GRU-based Seq2Seq network trained on synthetic reasoning data
- **Frontend:** React 18 with Vite build tooling
- **Deployment Target:** Vercel (frontend) + any container-friendly Python host (Render, Fly.io, Railway, etc.)

## 🏗 Architecture

1. The React app collects a prompt and sends it to the backend via `POST /api/complete`.
2. FastAPI validates the request and forwards it to the reasoning engine.
3. The reasoning engine tokenizes the prompt, runs it through a tiny Seq2Seq model, and generates an explanation.
4. The backend returns `{ "completion": "..." }` to the browser, which renders the result.

```
[React UI] --(POST /api/complete)--> [FastAPI] --> [PyTorch Seq2Seq]
     ^                                                    |
     |-------------------- JSON completion ---------------|
```

## 🛠 Setup Guide

### Requirements

- Python 3.11+
- Node.js 18+ (Vite recommends an actively maintained LTS release)
- npm (bundled with Node) or pnpm/yarn if you prefer

### Backend Installation & Local Run

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app.api:app --reload --port 8000
```

The first startup trains the tiny model on synthetic data (~a few seconds). Afterwards the API is ready at `http://localhost:8000`.

### Frontend Installation & Local Run

```bash
cd frontend
cp .env.example .env.local  # Optional: customize API base URL
npm install
npm run dev
```

Open the printed URL (default `http://localhost:5173`) to try the UI. The frontend reads `VITE_API_BASE_URL` from `.env.local`; when omitted it defaults to `http://localhost:8000`.

### Model Initialization Notes

- The backend auto-generates a balanced dataset of arithmetic, parity, and ordering prompts each time it starts.
- Training runs for 30 epochs using Adam; gradients are clipped for stability.
- Because training happens at startup, no extra “model download” step is required.

## 📡 API Documentation

### `POST /api/complete`

- **Request Body**

```json
{
  "prompt": "What is 3 plus 4?"
}
```

- **Successful Response**

```json
{
  "completion": "first, compute 3 + 4 = 7. therefore the answer is 7."
}
```

- **Validation Errors**: Return HTTP 422 with a `detail` message when the prompt is empty or missing.
- **Server Errors**: Return HTTP 500 with a generic `detail` field if the model fails to generate a completion.

A simple health check is also exposed at `GET /`.

## ☁️ Deployment Guide

### Frontend on Vercel

1. Push the repository to GitHub or GitLab.
2. In Vercel, create a new project from the `frontend` folder.
3. Set the build command to `npm run build` and the output directory to `dist` (already encoded in `frontend/vercel.json`).
4. Configure an environment variable `VITE_API_BASE_URL` pointing at your deployed backend URL.
5. Deploy—Vercel will run the Vite build and serve the static assets globally.

### Backend Hosting Options

- **Render / Railway / Fly.io**: Create a new web service from the `backend` folder. Use `uvicorn app.api:app --host 0.0.0.0 --port $PORT` as the start command.
- **Docker**: Containerize by installing requirements and exposing port 8000.
- **Local Tunnel**: For quick demos, run the backend locally and expose it via `ngrok http 8000`.

Update the frontend environment variable (`VITE_API_BASE_URL`) with the deployed backend URL so both services communicate successfully.

### Environment Variables Summary

| Name | Location | Purpose |
|------|----------|---------|
| `VITE_API_BASE_URL` | `frontend/.env.local` (also set in Vercel) | Base URL of the FastAPI backend |

## 🎛 Customization

- **Model behavior:** Modify `backend/app/model.py` to adjust the synthetic dataset or training routine. For example, add multiplication prompts or extend the vocabulary.
- **Training regimen:** Tweak hyperparameters (epochs, hidden size, etc.) to trade off accuracy vs. startup time.
- **Frontend UI:** Edit `frontend/src/App.jsx` and `frontend/src/styles.css` to introduce history views, syntax highlighting, or theming.
- **Scaling up:** Replace the toy model with a larger Transformer by swapping the architecture in `TinySeq2Seq` and loading pre-trained weights.

## 🧩 Troubleshooting & FAQ

- **The frontend says it cannot reach the API.** Double-check that the backend is running and `VITE_API_BASE_URL` matches the reachable URL (including `https://` in production).
- **CORS errors in the browser console.** The FastAPI app enables permissive CORS; ensure you restarted the backend after changes and that proxies are not stripping headers.
- **Model responses look random.** Restart the backend to retrain from scratch or increase training epochs in `ReasoningEngine._train`.
- **Deployment works locally but not on Vercel.** Confirm that the frontend project is built from the `frontend` directory and that environment variables are configured in the Vercel dashboard.
- **How can I inspect requests?** Use your browser’s devtools network tab or `curl -X POST http://localhost:8000/api/complete -H 'Content-Type: application/json' -d '{"prompt": "what is 2 plus 5?"}'`.

Enjoy tinkering with your mini reasoning assistant! 🎉
