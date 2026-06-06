# !pip install -q bert-score transformers sentencepiece nltk pandas torch

# Imports

import re
import numpy as np
import pandas as pd
import torch

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


# Configuration

CSV_FILE = "data/Evaluation Data/Islamic_classification.csv"

REFERENCE_COL = "التفسير"
PREDICTION_COL = "التوقع"
CATEGORY_COL = "الفئة"

MODEL_NAME = "aubmindlab/bert-base-arabertv02"

device = "cuda" if torch.cuda.is_available() else "cpu"


# Arabic Normalization

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = re.sub(r'[\u064B-\u0652\u0670]', '', text)
    text = re.sub(r'ـ', '', text)

    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)

    text = re.sub(r"[^\w\s]", " ", text)

    text = " ".join(text.split())

    return text.strip()


# ROUGE-L

def rouge_l_score(pred, gold):

    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()

    m = len(gold_tokens)
    n = len(pred_tokens)

    if m == 0 or n == 0:
        return 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):

            if gold_tokens[i] == pred_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(
                    dp[i][j + 1],
                    dp[i + 1][j]
                )

    lcs = dp[m][n]

    precision = lcs / n
    recall = lcs / m

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        /
        (precision + recall)
    )

# BLEU-4

smoothie = SmoothingFunction().method1

def bleu4_score(pred, gold):

    pred_tokens = normalize_text(pred).split()

    gold_tokens = [
        normalize_text(gold).split()
    ]

    if len(pred_tokens) == 0:
        return 0.0

    return sentence_bleu(
        gold_tokens,
        pred_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie
    )


# Load AraBERT

print("Loading AraBERT...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model.to(device)
model.eval()


# Mean Pooling

def mean_pooling(model_output, attention_mask):

    token_embeddings = model_output.last_hidden_state

    mask = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()

    return torch.sum(
        token_embeddings * mask,
        dim=1
    ) / torch.clamp(
        mask.sum(dim=1),
        min=1e-9
    )


# Batch Embedding

def encode_texts(texts, batch_size=16):

    embeddings = []

    for i in range(0, len(texts), batch_size):

        batch = texts[i:i + batch_size]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        encoded = {
            k: v.to(device)
            for k, v in encoded.items()
        }

        with torch.no_grad():

            output = model(**encoded)

            emb = mean_pooling(
                output,
                encoded["attention_mask"]
            )

            emb = torch.nn.functional.normalize(
                emb,
                p=2,
                dim=1
            )

        embeddings.append(
            emb.cpu().numpy()
        )

    return np.vstack(embeddings)


# Fairness

def fairness_gap(scores):

    return float(
        np.max(scores)
        -
        np.min(scores)
    )

def fairness_cv(scores):

    scores = np.array(scores)

    return float(
        np.std(scores)
        /
        (np.mean(scores) + 1e-9)
    )


# Load CSV

df = pd.read_csv(CSV_FILE)

df[REFERENCE_COL] = df[REFERENCE_COL].astype(str)
df[PREDICTION_COL] = df[PREDICTION_COL].astype(str)


# Traditional Metrics (ROUGE & BLEU)

ROUGE = []
BLEU = []

for _, row in df.iterrows():

    gold = row[REFERENCE_COL]
    pred = row[PREDICTION_COL]

    ROUGE.append(
        rouge_l_score(pred, gold)
    )

    BLEU.append(
        bleu4_score(pred, gold)
    )

# BERTScore

print("Calculating BERTScore...")

preds = df[PREDICTION_COL].tolist()
refs = df[REFERENCE_COL].tolist()

P, R, F1 = bert_score(
    preds,
    refs,
    num_layers=12,
    model_type=MODEL_NAME,
    lang="ar",
    device=device
)


# Cosine Similarity

print("Calculating AraBERT embeddings...")

pred_embeddings = encode_texts(preds)
gold_embeddings = encode_texts(refs)

COSINE = []

for p_emb, g_emb in zip(
    pred_embeddings,
    gold_embeddings
):

    COSINE.append(
        float(
            cosine_similarity(
                p_emb.reshape(1, -1),
                g_emb.reshape(1, -1)
            )[0][0]
        )
    )


# Save Metrics

df["ROUGE_L"] = ROUGE
df["BLEU_4"] = BLEU

df["BERTScore_P"] = P.cpu().numpy()
df["BERTScore_R"] = R.cpu().numpy()
df["BERTScore_F1"] = F1.cpu().numpy()

df["Cosine_Similarity"] = COSINE


# Global Results

print("\n" + "=" * 60)

print(f"ROUGE-L:            {np.mean(ROUGE):.4f}")
print(f"BLEU-4:             {np.mean(BLEU):.4f}")

print(f"BERT Precision:     {P.mean().item():.4f}")
print(f"BERT Recall:        {R.mean().item():.4f}")
print(f"BERT F1:            {F1.mean().item():.4f}")

print(f"Cosine Similarity:  {np.mean(COSINE):.4f}")

print("=" * 60)


# Fairness Evaluation

if CATEGORY_COL in df.columns:

    print("\nPER CATEGORY SCORES\n")

    category_scores = (
        df.groupby(CATEGORY_COL)["BERTScore_F1"]
        .mean()
        .sort_values(ascending=False)
    )

    print(category_scores)

    gap = fairness_gap(
        category_scores.values
    )

    cv = fairness_cv(
        category_scores.values
    )

    print("\n" + "=" * 60)

    print(f"Fairness Gap: {gap:.4f}")
    print(f"Fairness CV:  {cv:.4f}")

    print("=" * 60)


# Export

OUTPUT_FILE = "evaluation_results.csv"

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nSaved: {OUTPUT_FILE}")