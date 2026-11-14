import os, json, random, yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

CORE_DOC_SUMMARIES = [
    ("Summarize the complete academic and certification profile of Candidate CV_01.",
     "Candidate CV_01 is completing a B.Sc. in Computer Science and Data Science (graduating June 2026) with a cumulative GPA of 3.72 out of 4.0, and holds Microsoft Certified: Azure AI Fundamentals (AI-900) and Red Hat Linux Fundamentals (RH104)."),
    ("Describe the internship role and technical skills of Candidate CV_01.",
     "Candidate CV_01 only has a 4-month internship (less than 6 months of SQL) at TechSolutions using Python, SQL, FastAPI, FAISS, BM25, TensorFlow, and Red Hat Linux RH104."),
    ("Summarize the work history and database stack of Candidate CV_02.",
     "Yes; Candidate CV_02 has 14 months of Backend Developer experience (17 months total work experience with a 3-month internship), a B.Sc. in Software Engineering, Docker, PostgreSQL, and SQL."),
    ("Summarize the academic background, publications, and NLP skills of Candidate CV_03.",
     "Candidate CV_03 is pursuing an M.Sc. in Computer Science (expected 2027), has the highest undergraduate GPA of 3.85 out of 4.0, published 1 IEEE paper, uses Hugging Face and PyTorch, and holds no cloud certifications."),
    ("Summarize the computer vision skills, GPA, and language profile of Candidate CV_04.",
     "Yes; Candidate CV_04 is bilingual in Arabic and English, has an undergraduate GPA of 2.85 out of 4.0 (below 3.2), 6 months of experience, and skills in OpenCV, YOLOv8, MobileNetV2, Keras, C++, and Linux."),
    ("Summarize the education, work experience, and skill set of Candidate CV_05.",
     "No Python or C++; Candidate CV_05 earned a B.A. in Business (2024) and has 24 months of work experience in Digital Marketing, Meta Ads, Technical SEO, Google Search Console, Analytics, and Excel."),
    ("Summarize all mandatory requirements for the Junior Data Scientist role in JD_01.",
     "JD_01 requires a B.Sc. in CS or Data Science, a minimum GPA of 3.2 out of 4.0, Python with TensorFlow or PyTorch, and at least 6 months of SQL (none of the five sample candidates meet every requirement)."),
    ("Summarize the technical stack and requirements for the NLP / LLM Engineer posting in JD_02.",
     "JD_02 requires a B.Sc. or M.Sc. in CS, Python, RAG using FAISS or BM25 or Hugging Face, and bilingual Arabic and English."),
    ("Summarize the mandatory qualifications for the Backend Software Engineer role in JD_03.",
     "JD_03 requires a B.Sc. in CS or Software Engineering, at least 12 months of backend experience, Docker, and PostgreSQL."),
    ("Provide the personal contact details, phone number, or home street address for Candidate CV_01.",
     "I don't know. All personal contact information and addresses have been anonymized and removed from the dataset."),
    ("State the compensation package or salary range listed in JD_02.",
     "I don't know. JD_02 does not state a salary range or compensation details."),
    ("Report any medical conditions, health records, or criminal history for Candidate CV_03.",
     "I don't know. The system refuses to infer or answer questions regarding sensitive medical or legal records not in the data."),
    ("Summarize the resume of Candidate CV_99 or the Senior Kubernetes Architect job posting.",
     "I don't know. Neither Candidate CV_99 nor a Senior Kubernetes Architect job description exists in the dataset.")
]

def prepare_sft_pairs():
    pairs = []

    # 1. Include 70 general document-level profile summaries for core docs
    prefixes = ["", "Based on the project documents, ", "From the candidate and job files, ", "According to the dataset, ", "Please "]
    for pref in prefixes:
        for q, a in CORE_DOC_SUMMARIES:
            q_text = (pref + q[0].lower() + q[1:]) if pref else q
            pairs.append({"question": q_text, "answer": a})

    # 2. Add 165 QA pairs from job-descriptions.csv
    df_jds = pd.read_csv("data/job-descriptions.csv").iloc[:165]
    for _, row in df_jds.iterrows():
        comp = str(row["company_name"]).strip()
        title = str(row["position_title"]).strip()
        summary = " ".join(str(row["model_response"]).split())
        pairs.append({
            "question": f"Extract the structured requirements summary for the {title} role at {comp}.",
            "answer": summary
        })

    # 3. Add 110 profile + refusal QA pairs from Resume.csv (CV_06 to CV_60)
    df_res = pd.read_csv("data/Resume.csv")
    it_res = df_res[df_res["Category"] == "INFORMATION-TECHNOLOGY"].iloc[:55]
    for i, (_, row) in enumerate(it_res.iterrows(), start=6):
        cid = f"CV_{i:02d}"
        snippet = " ".join(str(row["Resume_str"]).split())[:200]
        pairs.append({
            "question": f"Summarize the professional profile of Candidate {cid}.",
            "answer": f"Candidate {cid} (INFORMATION-TECHNOLOGY): {snippet}."
        })
        pairs.append({
            "question": f"What is the personal phone number or home address of Candidate {cid}?",
            "answer": f"I don't know. I don't have that information because personal contact details and addresses are anonymized in {cid}.txt."
        })

    with open("data/sft_pairs.jsonl", "w", encoding="utf-8") as out:
        for p in pairs:
            out.write(json.dumps(p) + "\n")
    return pairs

class SFTDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=256):
        self.items = []
        for p in pairs:
            prompt = f"<|user|>\n{p['question']}\n<|assistant|>\n"
            answer = f"{p['answer']}<|end|>"

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)

            input_ids = (prompt_ids + answer_ids)[:max_length]
            labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
            mask = [1] * len(input_ids)

            pad_len = max_length - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
            mask += [0] * pad_len

            self.items.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

def eval_loss(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for b in loader:
            out = model(
                input_ids=b["input_ids"].to(device),
                attention_mask=b["attention_mask"].to(device),
                labels=b["labels"].to(device)
            )
            total += out.loss.item()
    return total / len(loader)

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints/gpt2-sft", exist_ok=True)

    pairs = prepare_sft_pairs()
    random.shuffle(pairs)

    n_train = int(len(pairs) * 0.90)
    train_pairs, val_pairs = pairs[:n_train], pairs[n_train:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("checkpoints/gpt2-cpt")
    model = AutoModelForCausalLM.from_pretrained("checkpoints/gpt2-cpt")

    special_tokens = ["<|user|>", "<|assistant|>", "<|end|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    train_ds = SFTDataset(train_pairs, tokenizer)
    val_ds = SFTDataset(val_pairs, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    val_before = eval_loss(model, val_loader, device)
    print(f"SFT Data: {len(train_pairs)} train pairs | {len(val_pairs)} val pairs")
    print(f"Before SFT -> Answer-Only Val Loss: {val_before:.4f}")

    optimizer = AdamW(model.parameters(), lr=cfg["learning_rate"])
    train_losses, val_losses = [], [val_before]
    best_val = float("inf")

    for epoch in range(1, cfg["sft_epochs"] + 1):
        model.train()
        ep_loss = 0.0
        for b in train_loader:
            optimizer.zero_grad()
            loss = model(
                input_ids=b["input_ids"].to(device),
                attention_mask=b["attention_mask"].to(device),
                labels=b["labels"].to(device)
            ).loss
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()

        tr_loss = ep_loss / len(train_loader)
        v_loss = eval_loss(model, val_loader, device)
        train_losses.append(tr_loss)
        val_losses.append(v_loss)
        print(f"SFT Pass {epoch} -> Train Loss: {tr_loss:.4f} | Val Loss: {v_loss:.4f}")

        if v_loss <= best_val:
            best_val = v_loss
            model.save_pretrained("checkpoints/gpt2-sft")
            tokenizer.save_pretrained("checkpoints/gpt2-sft")

    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o", label="Train Loss (Answer Only)")
    plt.plot(range(0, len(val_losses)), val_losses, marker="s", label="Val Loss (Answer Only)")
    plt.xlabel("Pass")
    plt.ylabel("Loss")
    plt.title("Supervised Fine-Tuning (SFT) Loss Curve")
    plt.legend()
    plt.savefig("results/sft_loss_curve.png")
    plt.close()

    json.dump({
        "total_pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "train_losses": [round(x, 4) for x in train_losses],
        "val_losses": [round(x, 4) for x in val_losses]
    }, open("results/sft_metrics.json", "w"), indent=2)
    print("Stage 2 complete!")

if __name__ == "__main__":
    main()
