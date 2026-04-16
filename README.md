# ⚖️ Legal Document Assistant
### Agentic AI Capstone 2026 - Aman Nath Jha

An AI-powered legal assistant for paralegals and junior lawyers, built with **LangGraph**, **ChromaDB**, and **Groq (LLaMA 3.3 70B)**. The agent answers India-specific legal questions using a curated knowledge base, calculates legal deadlines using a DateTime tool, and maintains multi-turn conversation memory — all served through a Streamlit UI.

---

## 🗂️ Repository Structure

```
agentic-ai/
├── agent.py                # Core agent — LangGraph graph, nodes, tools, ChromaDB
├── capstone_streamlit.py   # Streamlit UI — chat interface, metadata display, session state
├── capstone (1).ipynb    # Development notebook — prototyping and experimentation
└── .gitignore
```

---

## 🧠 Architecture

The agent is a **LangGraph StateGraph** with the following nodes:

```
[User Input]
     ↓
  memory         ← Extracts user name, document type, maintains conversation history
     ↓
  router         ← Routes to: retrieve / tool / memory_only
     ↓
┌────┴────┬──────────┐
retrieve  tool    skip (memory_only)
     ↓       ↓       ↓
        answer         ← Synthesises answer from context + tool result
           ↓
         eval          ← Faithfulness scoring (0.0–1.0), retries if score < 0.7
           ↓
         save  ──→ END
```

### Routing Logic
| Route | Triggered When |
|---|---|
| `retrieve` | Any legal question — sections, acts, case law, procedures |
| `tool` | Date arithmetic — deadline calculation, limitation expiry, notice periods |
| `memory_only` | Pure greetings, one-word replies, "what did I ask earlier?" |

### Faithfulness Evaluation
After every `retrieve`-routed answer, the `eval` node scores how grounded the response is in the retrieved context (0.0–1.0). If the score is below **0.7**, the answer node retries (up to 2 times) with a stricter grounding instruction.

---

## 📚 Knowledge Base (15 Documents)

| # | Topic |
|---|---|
| 1 | Void vs Voidable Contracts — ICA 1872, Sections 2(g), 2(i), 19, 19A |
| 2 | Doctrine of Frustration — Section 56, *Satyabrata Ghose v Mugneeram Bangur* |
| 3 | NDA Clauses — Enforceability, carve-outs, residuals clause, IT Act Section 72 |
| 4 | Employment Contracts — Restraint of Trade, Section 27, *Niranjan Shankar Golikari* |
| 5 | CPC — Order VII Rule 11, Res Judicata (Section 11), Res Sub Judice (Section 10) |
| 6 | Limitation Act 1963 — Articles 36/55/113/58, Section 18 acknowledgement, Section 5 |
| 7 | Legal Notice — Section 80 CPC, service modes, WhatsApp notice validity |
| 8 | Bail — Sections 436/437/438/439 CrPC, triple test, *Arnesh Kumar* guidelines |
| 9 | Power of Attorney — GPA vs SPA, *Suraj Lamp* ruling, Registration Act |
| 10 | Confidentiality Clauses — Injunctive relief, Specific Relief Act Section 38, Section 74 |
| 11 | Digital Evidence — Section 65B, *Arjun Panditrao Khotkar*, WhatsApp/email certificates |
| 12 | Arbitration — Sections 8/34/37, Seat vs Venue, *BALCO v Kaiser Aluminium* |
| 13 | Consumer Protection Act 2019 — Jurisdiction tiers, Section 2(7), product liability |
| 14 | IPC — Sections 415/420, 405/406/409, 463/467/468/471, Companies Act Section 447 |
| 15 | Specific Relief Act 2018 Amendment — Mandatory specific performance, Section 20A |

---

## 🛠️ DateTime Calculator Tool

The agent includes a date arithmetic tool triggered automatically for deadline-related questions. It handles:

- **Limitation period expiry** — calculates last date to file suit from a start date
- **Section 18 acknowledgement reset** — calculates new deadline after written acknowledgement
- **Section 80 CPC notice period** — calculates earliest date to file suit (60-day notice)
- **Expiry warning** — alerts if the limitation period has already expired and notes Section 5 availability

**Example trigger:** *"My 3-year limitation period started on March 15, 2022. When is the last date to file?"*

---

## 🚀 Setup & Running

### 1. Clone the repository

```bash
git clone https://github.com/incursio-xd/agentic-ai.git
cd agentic-ai
```

### 2. Install dependencies

```bash
pip install streamlit langchain langchain-groq langchain-google-genai \
            langgraph sentence-transformers chromadb python-dateutil
```

### 3. Configure your API key

**Option A — Streamlit secrets (recommended)**

Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

**Option B — Environment variable**
```bash
export GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

Alternatively, set `GOOGLE_API_KEY` to use **Gemini 1.5 Flash** instead of Groq.

### 4. Run the app

```bash
streamlit run capstone_streamlit.py
```

---

## 💬 Example Questions

**Knowledge base (RAG):**
- *"What is the difference between a void and voidable contract under ICA 1872?"*
- *"Is a post-employment non-compete clause enforceable in India?"*
- *"What are the grounds for setting aside an arbitral award under Section 34?"*
- *"Can WhatsApp messages be used as evidence? What certificate is required?"*

**DateTime tool:**
- *"My limitation period of 3 years started on August 10, 2021. The defendant acknowledged the debt on January 5, 2023. When does my new limitation period end?"*
- *"I served a Section 80 CPC legal notice on February 1, 2025. When can I file suit?"*

---

## ⚙️ Design Decisions

**`@st.cache_resource` for agent initialisation** — The embedding model download and ChromaDB build happen once per session, not on every message. Without this, each user message would trigger a 30–60 second delay.

**`MemorySaver` with `thread_id`** — Each browser session gets a unique UUID as its thread ID, giving the agent multi-turn memory scoped to that session. The "New Conversation" button resets both the thread ID and the display history.

**Notebook vs .py separation** — `capstone (1).ipynb` is the development whiteboard for prototyping. `agent.py` and `capstone_streamlit.py` are the production files.

**Faithfulness eval loop** — After retrieval-based answers, the eval node checks whether the response stays grounded in the retrieved context. If faithfulness < 0.7, the answer node retries with a stricter system prompt (max 2 retries).

---

## ⚠️ Disclaimer

This assistant provides information from its curated knowledge base only. It does not constitute legal advice. Always consult a qualified advocate for legal decisions. The assistant will not fabricate section numbers, case names, or legal provisions not present in its knowledge base.

---

## 👤 Author

**Agentic AI Capstone 2026**  
Aman Nath Jha
