# !pip install faiss-gpu-cu11==1.10.0
# !pip install --upgrade sentence_transformers

import pandas as pd
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer,CrossEncoder
import ast
import re
from huggingface_hub import snapshot_download
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import requests
from dotenv import load_dotenv
import os


# MODEL

model_name = "Qwen/Qwen2.5-7B-Instruct"

gen_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

gen_model.eval()
torch.set_grad_enabled(False)

# DOWNLOAD RETRIEVER

snapshot_download(
    repo_id="SeragAmin/NAMAA-retriever-cosine-final_60-90",
    repo_type="model",
    local_dir="retriever_model",
    allow_patterns="NAMAA-retriever-cosine-final_contrastive_ara_top70/checkpoint-1985/*"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# LOAD RETRIEVAL MODEL

retrieval_model = SentenceTransformer(
    "retriever_model/NAMAA-retriever-cosine-final_contrastive_ara_top70/checkpoint-1985"
)

retrieval_model.to(device)
retrieval_model.eval()
model = CrossEncoder("yoriis/GTE-tydi-quqa-haqa")
diacritics_pattern = re.compile(r'[\u064B-\u0652\u0670]')


# EMBEDDING

def get_embedding(text):

    with torch.no_grad():

        emb = retrieval_model.encode(
            text,
            convert_to_numpy=True,
            device=device,
            normalize_embeddings=True
        )

    return emb.astype(np.float32)


# INDEX

def build_faiss_index(embeddings):

    embeddings = embeddings.astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(embeddings)

    return index

# LOAD QURAN

quran_passages = []

with open(
    "/kaggle/input/datasets/shaymaamadhetahmed/islamicdomaindata/QH-QA-25_Subtask2_QPC_v1.1.tsv",
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        parts = line.strip().split("\t")

        if len(parts) >= 2:

            passage_id = parts[0]
            passage_text = parts[1]

            quran_passages.append({
                "text": passage_text,
                "source": "quran",
                "id": passage_id
            })


# LOAD HADITH

hadith_passages = []

with open(
    "/kaggle/input/datasets/shaymaamadhetahmed/islamicdomaindata/QH-QA-25_Subtask2_Sahih-Bukhari_v1.0.jsonl",
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        try:

            item = ast.literal_eval(line.strip())

            cleaned_text = diacritics_pattern.sub('', item['hadith'])

            hadith_passages.append({
                "text": cleaned_text,
                "source": "hadith",
                "id": item['hadith_id']
            })

        except Exception as e:
            print(f"Skipping invalid line: {e}")

all_passages = quran_passages + hadith_passages

print(f"Loaded total passages: {len(all_passages)}")


# TEXTS

quran_texts = [p["text"] for p in quran_passages]
hadith_texts = [p["text"] for p in hadith_passages]


# EMBEDDINGS

quran_embeddings = retrieval_model.encode(
    quran_texts,
    convert_to_numpy=True,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
).astype(np.float32)

hadith_embeddings = retrieval_model.encode(
    hadith_texts,
    convert_to_numpy=True,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
).astype(np.float32)

# INDEXING

quran_index = build_faiss_index(quran_embeddings)

hadith_index = build_faiss_index(hadith_embeddings)


# SEARCH


def search(query, k_quran=50, k_hadith=20):

    query_emb = get_embedding(query).reshape(1, -1)

    D_q, I_q = quran_index.search(query_emb, k_quran)

    D_h, I_h = hadith_index.search(query_emb, k_hadith)

    results = []

    for i, score in zip(I_q[0], D_q[0]):

        passage = quran_passages[i]

        results.append({
            "score": float(score),
            "id": passage["id"],
            "source": "quran",
            "text": passage["text"]
        })

    for i, score in zip(I_h[0], D_h[0]):

        passage = hadith_passages[i]

        results.append({
            "score": float(score),
            "id": passage["id"],
            "source": "hadith",
            "text": passage["text"]
        })

    results = sorted(results, key=lambda x: x['score'], reverse=True)

    return results


def predict_Question_rerank_crossencoder(question, model, search_fn, k_retrieve=70, score_threshold=0.15, max_returned=20):
    all_results = []
    # List of Quran & Hadith Passages with Score
    retrieved = search_fn(question)

    # get texts of retrieved passages 
    candidate_texts = [r["text"] for r in retrieved]
 
    # rerank retrieved Quran & Hadith Passages based on the most relevant for question
    reranked = model.rank(query=question, documents=candidate_texts)
   
    # handle the no-answer questions
    filtered = [item for item in reranked if item['score'] >= score_threshold]
    filtered = sorted(filtered, key=lambda x: x['score'], reverse=True)[:max_returned]
   
    # check if zero answer
    if not filtered:
            all_results.append({
               
                "لا توجد اجابة"
            })
        
    # collect top texts
    for item in filtered:
        corpus_id = item['corpus_id']
        score=item['score']
        all_results.append({
                "text": candidate_texts[corpus_id],
                 "score":score
                })

    return all_results



# PROMPT

def get_promptTemplate(question):
    prompt = f"""<|im_start|>system
أنت مساعد استدلال فقهي إسلامي.

المهمة:
استخراج الأدلة الشرعية الصحيحة ثم بناء الحكم الشرعي بناءً عليها.

قواعد إلزامية صارمة:

1) كل نص شرعي (قرآن / حديث / أثر / دعاء مأثور) يجب أن يكون داخل:
[STA] ... [END]

2) ممنوع وجود أي نص شرعي خارج التاجات.

3) لا يجوز إعادة صياغة النصوص الشرعية.

4) جميع الأدلة يجب أن تكون دقيقة مع مصدرها داخل نفس التاج.

5) يمنع إضافة أي نصوص شرعية من عندك.

6) إذا لم يوجد دليل صحيح صريح اكتب فقط:
لا أعلم دليلاً صحيحاً صريحاً في ذلك.

7) الرد باللغة العربية فقط.

8) ممنوع إضافة مقدمات أو عناوين غير مطلوبة.

========================
صيغة الإخراج الإلزامية (مقفولة)
========================

التفسير:
[STA] الدليل الأول (مع المصدر) [END]
و
[STA] الدليل الثاني (مع المصدر) [END]

شرح الأدلة:
شرح مفصل لمعنى الأدلة وعلاقتها بالسؤال.

الاجابة:
الحكم النهائي المباشر.

========================
أمثلة
========================

السؤال: ما حكم الاستغفار للمشرك بعد موته؟

التفسير:
[STA] ما كان للنبي والذين آمنوا أن يستغفروا للمشركين ولو كانوا أولي قربى ... (التوبة: 113) [END]
شرح الأدلة:
الآية تنهى عن الاستغفار للمشرك بعد موته.
الاجابة:
لا يجوز الاستغفار للمشرك بعد موته.

---

السؤال: هل الشيطان يتمثل بالنبي ﷺ؟

التفسير:
[STA] قال رسول الله ﷺ: من رآني في المنام فقد رآني فإن الشيطان لا يتمثل بي (البخاري 6993) [END]
شرح الأدلة:
الحديث ينفي قدرة الشيطان على التمثل بالنبي ﷺ.
الاجابة:
الشيطان لا يتمثل بالنبي ﷺ.

<|im_end|>

<|im_start|>user
السؤال: {question}
<|im_end|>

<|im_start|>assistant
التفسير:
"""
    return prompt



def load_API():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API")
    return api_key
    

def extract_search_query(text: str) -> str:
    api_key = load_API()
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-oss-120b:free",
                "messages": [{"role": "user", "content": f"Create a short semantic Arabic search query from this text.\n\nText:\n{text}\n\nReturn ONLY the search query."}],
                "temperature": 0.0,
                "max_tokens": 50
            },
            timeout=30
        )
        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"].strip()
        
        print("Query extraction failed: Unexpected JSON structure.")
        return text

    except Exception as e:
        print(f"Query extraction error: {e}")
        return text
    

# GREEDY Sampling

def greedy_sampling(logits: torch.Tensor):

    return torch.argmax(logits, dim=-1, keepdim=True)


# CLEAN OUTPUT

def remove_sta_end_tags(text: str):

    text = re.sub(r'\[STA\]\s*', '', text)

    text = re.sub(r'\s*\[END\]', '', text)

    return text.strip()


# RID

def RID(
    question: str,
    max_steps: int = 512,
    threshold: float = 0.7,
):

    device = gen_model.device

    prompt = get_promptTemplate(question)

    input_ids = tokenizer(
        prompt,
        return_tensors="pt"
    ).input_ids.to(device)

    generated_ids = input_ids[0].tolist()

    eos_id = tokenizer.eos_token_id

    STA_TAG = "[STA]"
    END_TAG = "[END]"

    collecting = False
    sta_token_start = None
    buffer_ids = []

    for step in range(max_steps):

        inputs = torch.tensor(
            [generated_ids],
            device=device
        )

        with torch.no_grad():
            outputs = gen_model(inputs)

        logits = outputs.logits[:, -1, :]

        next_token_id = greedy_sampling(logits).item()

        if next_token_id == eos_id:
            break

        generated_ids.append(next_token_id)

        full_decoded = tokenizer.decode(
            generated_ids,
            skip_special_tokens=False
        )

   
        # DETECT [STA]
        

        if not collecting and full_decoded.endswith(STA_TAG):

            collecting = True

            sta_token_start = len(generated_ids)

            buffer_ids = []

            print("START COLLECTING")

            continue

    
        # COLLECT TOKENS
      
        if collecting:

            buffer_ids.append(next_token_id)

            buffer_text = tokenizer.decode(
                buffer_ids,
                skip_special_tokens=False
            )

            # DETECT [END]
           

            if END_TAG in buffer_text:

                collecting = False

                query = buffer_text.split(END_TAG)[0].strip()

                print("=" * 50)
                print("RAW QUERY:", query)

                query = extract_search_query(query)

                print("SEARCH QUERY:", query)

                results = predict_Question_rerank_crossencoder(query, model, search_fn=search, k_retrieve=70)

                if len(results) > 0:

                    best_score = results[0]["score"]

                    print("BEST SCORE:", best_score)

                    if best_score >= threshold:

                        retrieved_text = " ".join(
                            [r["text"] for r in results[:1]]
                        )

                        replacement = (
                            f"[STA] {retrieved_text} [END]"
                        )

                        replacement_ids = tokenizer.encode(
                            replacement,
                            add_special_tokens=False
                        )

                     
                        # REMOVE GENERATED HALLUCINATION
                    
                        sta_ids = tokenizer.encode(
                            STA_TAG,
                            add_special_tokens=False
                        )

                        generated_ids = generated_ids[
                            :sta_token_start - len(sta_ids)
                        ]

                       
                        # INSERT RETRIEVED TEXT
                       

                        generated_ids.extend(replacement_ids)

                        print("RETRIEVAL APPLIED")

                     
                        continue

                    else:

                        print("LOW SCORE -> KEEP GENERATED")

                buffer_ids = []
                sta_token_start = None

    full_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    prompt_text = tokenizer.decode(
        input_ids[0].tolist(),
        skip_special_tokens=True
    )

    if full_text.startswith(prompt_text):

        full_text = full_text[len(prompt_text):].strip()

    return full_text





question = "ما هى اركان الاسلام ؟"

explanation = RID(
    question,
    max_steps=512,
    threshold=0.7
)

print("\n")
print("=" * 80)
print(remove_sta_end_tags(explanation ))