# VeriNews AI 🔍

> **AI-powered Fake News Detection** — a full-stack application with a FastAPI backend and a React + Vite frontend.

![Status](https://img.shields.io/badge/status-in%20development-orange)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![React](https://img.shields.io/badge/react-18-61dafb)

---

## 📁 Project Structure

```
VeriNews-AI/
├── backend/
│   ├── app/
│   │   ├── routes/          # API route modules
│   │   ├── services/        # Business logic / ML inference
│   │   ├── models/          # Pydantic schemas & DB models
│   │   ├── utils/           # Shared helpers
│   │   └── main.py          # FastAPI app entry point
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── src/
│   │   ├── components/      # Reusable UI components
│   │   ├── pages/           # Page-level components
│   │   ├── services/        # API call wrappers
│   │   ├── hooks/           # Custom React hooks
│   │   └── assets/          # Static assets
│   ├── vite.config.js
│   └── package.json
│
├── docs/                    # Architecture & design docs
├── screenshots/             # UI screenshots
├── tests/                   # Integration / E2E tests
├── README.md
├── .gitignore
└── LICENSE
```

---

## ⚡ Quick Start

### Prerequisites

| Tool | Minimum Version |
|------|-----------------|
| Python | 3.11+ |
| Node.js | 18+ |
| npm | 9+ |

---

### 1 — Clone the repository

```bash
git clone https://github.com/your-username/VeriNews-AI.git
cd VeriNews-AI
```

---

### 2 — Backend Setup (FastAPI)

```bash
# Navigate to the backend directory
cd backend

# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy the example env file and fill in values
copy .env.example .env       # Windows
# cp .env.example .env       # macOS / Linux

# Start the development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at **http://localhost:8000**  
Interactive docs: **http://localhost:8000/docs**

---

### 3 — Frontend Setup (React + Vite)

Open a **new terminal** in the project root:

```bash
cd frontend

# Install dependencies (already done if you used the scaffold)
npm install

# Copy the example env file
copy .env.example .env.local       # Windows
# cp .env.example .env.local       # macOS / Linux

# Start the Vite dev server
npm run dev
```

The frontend will be available at **http://localhost:5173**

---

### 4 — Verify the Health Endpoint

With both servers running, open a browser or use curl:

```bash
curl http://localhost:8000/api/v1/health
```

Expected response:

```json
{
  "status": "ok",
  "message": "VeriNews AI backend is running."
}
```

The same message is displayed on the frontend home page.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Backend health check |

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18, Vite 5 |
| Backend | Python, FastAPI, Uvicorn |
| Styling | Vanilla CSS (dark, glassmorphic) |
| API Docs | FastAPI / Swagger UI |
| ML (planned) | PyTorch, Transformers, scikit-learn |

---

## 📸 Screenshots

> _Screenshots will be added to the `screenshots/` directory during development._

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
