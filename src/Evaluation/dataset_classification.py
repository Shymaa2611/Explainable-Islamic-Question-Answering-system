# !pip install pandas requests tqdm

import os
import time
import json
import pandas as pd
import requests
from tqdm import tqdm


# CONFIG

API_KEY = "YOUR API"
INPUT_FILE = "/content/combined.csv"
OUTPUT_FILE = "qrcd_qe_islamqa_classification.csv"
CHECKPOINT_FILE = "checkpoint_classification.csv"
MODEL = "deepseek/deepseek-chat-v3-0324"
MAX_RETRIES = 5
SAVE_EVERY = 100
DEBUG = True


# LABELS

ID_TO_CATEGORY = {
    "1": "العقيدة",
    "2": "التفسير وعلوم القرآن",
    "3": "الحديث وعلومه",
    "4": "الفقه وأصوله",
    "5": "فقه الأسرة",
    "6": "الأخلاق والرقائق",
    "7": "الدعوة والعلم",
    "8": "قضايا اجتماعية ونفسية",
    "9": "التاريخ والسيرة",
    "10": "التربية",
    "11": "السياسة الشرعية",
    "12": "غير ذلك"
}


# LOAD DATA

if os.path.exists(CHECKPOINT_FILE):
    print("Loading checkpoint...")
    df = pd.read_csv(CHECKPOINT_FILE)
else:
    print("Loading original dataset...")
    df = pd.read_csv(INPUT_FILE)

    if "topic_islamqa" not in df.columns:
        df["topic_islamqa"] = None


# CACHE

cache = {}

completed = df[df["topic_islamqa"].notna()]

for _, row in completed.iterrows():
    cache[str(row["السؤال"])] = row["topic_islamqa"]

# CLASSIFIER

def classify_islamqa(question):

    question = str(question).strip()

    if question in cache:
        return cache[question]



    prompt = f"""
              
          أنت خبير في تصنيف الأسئلة الإسلامية.

          المهمة:
          صنّف السؤال إلى فئة واحدة فقط من الفئات التالية:

          1 العقيدة
          2 التفسير وعلوم القرآن
          3 الحديث وعلومه
          4 الفقه وأصوله
          5 فقه الأسرة
          6 الأخلاق والرقائق
          7 الدعوة والعلم
          8 قضايا اجتماعية ونفسية
          9 التاريخ والسيرة
          10 التربية
          11 السياسة الشرعية
          12 غير ذلك

          تعريفات مختصرة:

          1: التوحيد، الإيمان، العقائد.
          2: تفسير القرآن وعلومه.
          3: الأحاديث النبوية وعلوم الحديث.
          4: العبادات والمعاملات والأحكام الفقهية.
          5: الزواج والطلاق والميراث والأسرة.
          6: الأخلاق والتزكية والرقائق.
          7: الدعوة وطلب العلم الشرعي.
          8: القضايا الاجتماعية والنفسية.
          9: السيرة النبوية والتاريخ الإسلامي.
          10: التربية والتعليم وبناء الشخصية.
          11: الحكم والشورى والسياسة الشرعية.
          12: أي موضوع آخر.

          أمثلة:

          السؤال: هل يجوز تنظيم النسل بسبب ضعف الحال المادية؟
          الإجابة: 5

          السؤال: ما موقف المسلم من الاختلاف في العقيدة؟
          الإجابة: 1

          السؤال: كيف أتعامل مع الطفل المصاب بالتوحد؟
          الإجابة: 8

          السؤال: ما حكم إخراج زكاة الفطر نقداً؟
          الإجابة: 4

          السؤال: ما معنى قوله تعالى: إياك نعبد وإياك نستعين؟
          الإجابة: 2

          السؤال: من هو خالد بن الوليد؟
          الإجابة: 9

          القواعد:
          - اختر الفئة الأكثر ارتباطاً بالسؤال.
          - إذا احتمل السؤال أكثر من فئة فاختر الفئة الأساسية فقط.
          - أعد الرقم فقط.
          - لا تكتب أي تفسير أو نص إضافي.
          - يجب أن يكون الناتج رقماً واحداً فقط من 1 إلى 12.

          السؤال:
          {question}

          الإجابة:
"""
    for attempt in range(MAX_RETRIES):

        try:

            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0,
                    "max_tokens": 20
                },
                timeout=60
            )

            # Print actual OpenRouter error
            if response.status_code != 200:

                print(
                    f"\nStatus: {response.status_code}"
                )

                print(
                    response.text[:1000]
                )

                time.sleep(2 ** attempt)

                continue

            result = response.json()

            if DEBUG:
                print(
                    json.dumps(
                        result,
                        ensure_ascii=False
                    )[:500]
                )

            if (
                "choices" not in result
                or len(result["choices"]) == 0
            ):
                continue

            prediction = (
                result["choices"][0]
                ["message"]
                .get("content")
            )

            print(
                "RAW:",
                repr(prediction)
            )

            if prediction is None:
                continue

            prediction = (
                str(prediction)
                .strip()
                .replace(".", "")
                .replace("\n", "")
                .replace(" ", "")
            )

            category = ID_TO_CATEGORY.get(
                prediction,
                "غير ذلك"
            )

            cache[question] = category

            return category

        except Exception as e:

            print(
                f"Retry {attempt+1}/{MAX_RETRIES}"
            )

            print(e)

            time.sleep(2 ** attempt)

    return "غير ذلك"

# REMAINING ROWS

remaining_indices = df[
    df["topic_islamqa"].isna()
].index.tolist()

print(
    f"Remaining rows: {len(remaining_indices)}"
)


# MAIN LOOP

for count, idx in enumerate(
    tqdm(
        remaining_indices,
        desc="Classifying"
    ),
    start=1
):

    question = str(
        df.at[idx, "السؤال"]
    )

    label = classify_islamqa(
        question
    )

    df.at[idx, "topic_islamqa"] = label

    time.sleep(0.5)

    if count % SAVE_EVERY == 0:

        df.to_csv(
            CHECKPOINT_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            f"\nCheckpoint saved "
            f"({count} rows)"
        )


# FINAL SAVE

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nFinished!\n")

print(
    df["topic_islamqa"]
    .value_counts(dropna=False)
)

print(
    f"\nSaved to: {OUTPUT_FILE}"
)