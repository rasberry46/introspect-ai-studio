import os
from io import BytesIO

import pandas as pd
import networkx as nx
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pypdf import PdfReader

from agents import AGENTS

load_dotenv()

app = FastAPI(title="Introspect AI Studio - Fraud Simulation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def extract_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    return "\n".join([page.extract_text() or "" for page in reader.pages])


def ask_llm(system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.7,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


@app.get("/health")
def health():
    return {"status": "ok", "mode": "fraud-simulation"}


@app.get("/")
def root():
    return {"message": "Introspect AI Studio backend is running"}


@app.post("/simulate")
async def simulate(prompt: str = Form(""), file: UploadFile = File(None)):
    text = prompt or ""

    if file:
        content = await file.read()
        if file.filename.lower().endswith(".pdf"):
            text = extract_pdf(content)
        else:
            text = content.decode("utf-8", errors="ignore")

    text = text.strip()

    if not text:
        return {
            "agents": [],
            "disagreement": "No input provided.",
            "summary": "Please paste a fraud/risk scenario or upload a file.",
        }

    text = text[:4000]
    print("INPUT TEXT:", text)

    results = []

    for agent in AGENTS:
        response = ask_llm(
            f"You are {agent['name']}. {agent['persona']}",
            f"""
Analyze this scenario for potential fraud risk:

{text}

Return:
- suspicious signals
- possible fraud patterns
- risk level (Low/Medium/High)
- reasoning
- likely next step if no action is taken
""",
        )

        results.append(
            {
                "agent": agent["name"],
                "persona": agent["persona"],
                "response": response,
            }
        )

    combined = "\n\n".join(
        [f"{r['agent']} ({r['persona']}):\n{r['response']}" for r in results]
    )

    disagreement = ask_llm(
        "You are an expert conflict and fraud review analyst.",
        f"""
Compare these agent responses:

{combined}

Return:
- key disagreements
- why they disagree
- which side is stronger and why
- what additional evidence would resolve the disagreement
""",
    )

    summary = ask_llm(
        "You are a senior fraud risk investigator.",
        f"""
Based on these agent responses:

{combined}

Provide:
- overall fraud risk score (0-100)
- final risk level (Low/Medium/High)
- top warning signs
- recommended action
- short executive summary
""",
    )

    return {
        "agents": results,
        "disagreement": disagreement,
        "summary": summary,
    }


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    content = await file.read()

    # Read CSV
    df = pd.read_csv(BytesIO(content))

    # Clean column names
    df.columns = [str(col).strip() for col in df.columns]

    # Try to auto-detect useful columns
    possible_user_cols = ["user_id", "customer_id", "account_id", "user"]
    possible_merchant_cols = ["merchant", "merchant_id", "merchant_name"]
    possible_device_cols = ["device_id", "device"]
    possible_ip_cols = ["ip_address", "ip", "source_ip"]
    possible_label_cols = ["fraud", "is_fraud", "label", "class"]

    def find_col(options):
        for col in options:
            if col in df.columns:
                return col
        return None

    user_col = find_col(possible_user_cols)
    merchant_col = find_col(possible_merchant_cols)
    device_col = find_col(possible_device_cols)
    ip_col = find_col(possible_ip_cols)
    label_col = find_col(possible_label_cols)

    G = nx.Graph()

    # Limit for MVP performance
    df = df.head(300)

    for idx, row in df.iterrows():
        tx_id = f"tx_{idx}"

        G.add_node(tx_id, label=tx_id, type="transaction")

        if user_col and pd.notna(row[user_col]):
            user = f"user_{row[user_col]}"
            G.add_node(user, label=str(row[user_col]), type="user")
            G.add_edge(tx_id, user, relation="made_by")

        if merchant_col and pd.notna(row[merchant_col]):
            merchant = f"merchant_{row[merchant_col]}"
            G.add_node(merchant, label=str(row[merchant_col]), type="merchant")
            G.add_edge(tx_id, merchant, relation="paid_to")

        if device_col and pd.notna(row[device_col]):
            device = f"device_{row[device_col]}"
            G.add_node(device, label=str(row[device_col]), type="device")
            G.add_edge(tx_id, device, relation="used_device")

        if ip_col and pd.notna(row[ip_col]):
            ip = f"ip_{row[ip_col]}"
            G.add_node(ip, label=str(row[ip_col]), type="ip")
            G.add_edge(tx_id, ip, relation="originated_from")

    # Very simple suspicious-node highlighting
    suspicious_nodes = set()

    for node in G.nodes():
        node_type = G.nodes[node].get("type")
        degree = G.degree(node)

        if node_type in ["device", "ip"] and degree >= 5:
            suspicious_nodes.add(node)

    # If fraud label exists, mark related transaction nodes
    fraud_count = 0
    if label_col:
        for idx, row in df.iterrows():
            raw = str(row[label_col]).strip().lower()
            if raw in ["1", "true", "yes", "fraud"]:
                tx_id = f"tx_{idx}"
                if tx_id in G.nodes:
                    G.nodes[tx_id]["fraud"] = True
                    suspicious_nodes.add(tx_id)
                    fraud_count += 1

    nodes = []
    for n, attrs in G.nodes(data=True):
        nodes.append(
            {
                "id": str(n),
                "label": attrs.get("label", str(n)),
                "type": attrs.get("type", "unknown"),
                "suspicious": n in suspicious_nodes,
                "fraud": attrs.get("fraud", False),
            }
        )

    edges = []
    for u, v, attrs in G.edges(data=True):
        edges.append(
            {
                "source": str(u),
                "target": str(v),
                "relation": attrs.get("relation", "connected_to"),
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "rows_processed": len(df),
            "columns_found": {
                "user_col": user_col,
                "merchant_col": merchant_col,
                "device_col": device_col,
                "ip_col": ip_col,
                "label_col": label_col,
            },
            "fraud_rows_detected": fraud_count,
            "suspicious_nodes": len(suspicious_nodes),
        },
    }