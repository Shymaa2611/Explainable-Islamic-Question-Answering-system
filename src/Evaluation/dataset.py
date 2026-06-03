import json
import pandas as pd



def Quran_prep():
        rows = []

        with open("data/Evaluation Data/qrcd_v1.1_test.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)

                question = sample.get("question", "")

                passage = sample.get("passage", "")

                answers = sample.get("answers", [])

                if len(answers) == 0:
                    rows.append({
                        "question": question,
                        "context": passage,
                        "answer": ""
                    })
                    continue

                for ans in answers:

                    if isinstance(ans, dict):
                        answer_text = ans.get("text", "")
                    else:
                        answer_text = str(ans)

                    rows.append({
                        "السؤال": question,
                        "التفسير":  passage + "\n\nالإجابة: " + answer_text
                    })

        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["السؤال"])
        sampled_df = df.sample(n=30, random_state=42)
        sampled_df .to_csv(
            "qrcd_qe.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print(df.head())





