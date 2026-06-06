import string
import pandas as pd
from bert_score import BERTScorer
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download

class EvaluationMetrics:
    def __init__(self):
        MODEL_NAME = "aubmindlab/bert-base-arabertv02"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.model.eval()

        self.scorer = BERTScorer(
        model_type="aubmindlab/bert-base-arabertv02",
        num_layers=12,
        lang="ar"
        )


    def load_model(self):
        snapshot_download(
        repo_id="SeragAmin/NAMAA-retriever-cosine-final_60-90",
        repo_type="model",
        local_dir="retriever_model",
        allow_patterns="NAMAA-retriever-cosine-final_contrastive_ara_top70/checkpoint-1985/*" )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = "retriever_model/NAMAA-retriever-cosine-final_contrastive_ara_top70/checkpoint-1985"
        retrieval_model = SentenceTransformer(model_path, device=device)
        return retrieval_model

    # load test data
    def load_data_csv(self,file_path):
        df = pd.read_csv(file_path)
        data = []

        for _, row in df.iterrows():
            data.append({
                "question": str(row.get("question", "")),
                "answer": str(row.get("answer", "")),
                "generatedAnswer":str(row.get("generatedAnswer"))
            })

        return data
    
    def normalize_text(self,s):
        """remove punctuation, some stopwords and extra whitespace."""
        def remove_stopWords(text):
            terms = []
            stopWords = {'من', 'الى', 'إلى', 'عن', 'على', 'في', 'حتى'}
            for term in text.split():
                if term not in stopWords:
                    terms.append(term)
            return " ".join(terms)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            # Arabic punctuation
            exclude.add('،')
            exclude.add('؛')
            exclude.add('؟')
            return ''.join(ch for ch in text if ch not in exclude)

        return white_space_fix(remove_stopWords(remove_punc(s)))

    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
            input_mask_expanded.sum(dim=1), min=1e-9
        )

    def encode(self,text):
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        with torch.no_grad():
            output = self.model(**encoded)
        embedding = self.mean_pooling(output, encoded["attention_mask"])
        return embedding.numpy()

    # cosine similarity metric
    def cosine_sim(self,truth, prediction):
        e1 = self.encode(truth)
        e2 = self.encode(prediction)
        return cosine_similarity(e1, e2)[0][0]


    # bert score metric
    def compute_bert_score(self,truth, prediction):
        if truth == "" or prediction == "":
            return 0.0, 0.0, 0.0

        P, R, F1 = self.scorer.score([prediction], [truth])

        return (
            float(P.mean().item()),
            float(R.mean().item()),
            float(F1.mean().item())
        )
    
    # exact match metric
    def exact_match_score(self,prediction, ground_truth):
        if len(prediction) == 0: 
            return 0
        return (self.normalize_text(prediction) == self.normalize_text(ground_truth))
  
   # Embedding
    def embedding(self,text):
        endcoded_model=self.load_model()
        embs = endcoded_model.encode(
        text,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True )

        return embs

    # preprocessing of hallucination
    def hallucination_preprocessing(self,df_path): 
        df = pd.read_json(df_path)
        df["prediction"] = df["prediction"].fillna("").astype(str)
        df["keypoints"] = df["keypoints"].apply(
            lambda x: x if isinstance(x, list) else []
        )

        df["keypoints"] = df["keypoints"].apply(
            lambda lst: [str(kp) for kp in lst if kp is not None])
        
        all_predictions = df["prediction"].tolist()
        all_keypoints = [kp for sublist in df["keypoints"] for kp in sublist]
        pred_embs = self.embedding(all_predictions)
        kp_embs=self.embedding(all_keypoints)
        idx = 0
        kp_embs_grouped = []
        for sublist in df["keypoints"]:
            kp_embs_grouped.append(kp_embs[idx: idx + len(sublist)])
            idx += len(sublist)

        df["Prediction_Embedding"] = list(pred_embs)
        df["Keypoints_Embeddings"] = kp_embs_grouped

        return df
        
    # calculate hallucination
    def compute_hallucination_score(self,df_path, threshold=0.5):
        df=self.hallucination_preprocessing(df_path)
        total_hallucination = 0.0
        n = len(df)
        for _, row in df.iterrows():
            pred = np.array(row["Prediction_Embedding"]).reshape(1, -1)
            kps = np.array(row["Keypoints_Embeddings"])
            if len(kps) == 0:
                continue
            sims = cosine_similarity(pred, kps)[0]
            contradictions = np.sum(sims <= threshold)
            total_hallucination += contradictions / len(kps)
        return  total_hallucination/n



   

def main():
    metrics=EvaluationMetrics()
    eval_data = metrics.load_data_csv("")
    P_scores, R_scores, F1_scores , Cosine_Similarity = [], [], [], [] 
    for item in eval_data:
        truth = item["explanation"]
        generated =item["generation"]
        #question=item["question"]
        p, r, f1 = metrics.compute_bert_score(truth, generated)
        score = metrics.cosine_sim(truth, generated)
    
        P_scores.append(p)
        R_scores.append(r)
        F1_scores.append(f1)
        Cosine_Similarity.append(score)
    print(f"Precision: {sum(P_scores)/len(eval_data)}")
    print(f"Recall:    {sum(R_scores)/len(eval_data)}")
    print(f"F1 Score:  {sum(F1_scores)/len(eval_data)}")
    print(f"Cosine_Similarity:  {sum(Cosine_Similarity)/len(eval_data)}")
    hallucination_score= metrics.compute_hallucination_score("", threshold=0.5)
    print("Hallucination Score:", hallucination_score)

if __name__=="__main__":
    main()