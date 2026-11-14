import os, glob, math, json, shutil, random, yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

PROMPTS = [
    "CANDIDATE ID: CV_01\nEDUCATION\nB.Sc. in Computer Science and Data Science\nGraduation Date:",
    "CANDIDATE ID: CV_02\nWORK EXPERIENCE (Total Work Experience: 17 months)\nBackend Developer |",
    "CANDIDATE ID: CV_03\nPUBLICATIONS & CERTIFICATIONS\n- Publications:",
    "JOB ID: JD_01\nPosition Title: Junior Data Scientist\nMANDATORY REQUIREMENTS:\n1. Education:",
    "JOB ID: JD_02\nPosition Title: NLP / LLM Engineer\n2. Core Technical Stack:"
]

def prepare_cpt_files():
    os.makedirs("data/cpt", exist_ok=True)

    for fname in ["CV_01.txt", "CV_02.txt", "CV_03.txt", "CV_04.txt", "CV_05.txt", "JD_01.txt", "JD_02.txt", "JD_03.txt"]:
        shutil.copy(f"data/samples/{fname}", f"data/cpt/{fname}")

    df_res = pd.read_csv("data/Resume.csv")
    it_res = df_res[df_res["Category"] == "INFORMATION-TECHNOLOGY"].iloc[:55]
    for i, (_, row) in enumerate(it_res.iterrows(), start=6):
        text = " ".join(str(row["Resume_str"]).split())[:2000]
        with open(f"data/cpt/CV_{i:02d}.txt", "w", encoding="utf-8") as out:
            out.write(f"CANDIDATE ID: CV_{i:02d}\n{text}\n")

    df_jds = pd.read_csv("data/job-descriptions.csv").iloc[:12]
    for i, (_, row) in enumerate(df_jds.iterrows(), start=4):
        text = " ".join(str(row["job_description"]).split())[:2000]
        with open(f"data/cpt/JD_{i:02d}.txt", "w", encoding="utf-8") as out:
            out.write(f"JOB ID: JD_{i:02d}\nPosition: {row['position_title']}\n{text}\n")

class CPTDataset(Dataset):
    def __init__(self, docs, tokenizer, block_size=512):
        full_text = f"\n\n{tokenizer.eos_token}\n\n".join(docs)
        ids = tokenizer.encode(full_text)
        self.total_tokens = len(ids)
        self.chunks = [
            torch.tensor(ids[i : i + block_size], dtype=torch.long)
            for i in range(0, len(ids) - block_size + 1, block_size)
        ]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        x = self.chunks[idx]
        return {"input_ids": x, "labels": x.clone()}

def eval_ppl(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for b in loader:
            total_loss += model(input_ids=b["input_ids"].to(device), labels=b["labels"].to(device)).loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss, math.exp(avg_loss)

def get_completions(model, tokenizer, device):
    model.eval()
    results = []
    for p in PROMPTS:
        inp = tokenizer(p, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=40, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        results.append({"prompt": p, "completion": text})
    return results

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints/gpt2-cpt", exist_ok=True)

    prepare_cpt_files()
    files = sorted(glob.glob("data/cpt/*.txt"))
    docs = [open(f, encoding="utf-8").read() for f in files]

    # Split 90% train (weighting the 8 core docs) and 10% held-out validation (7 docs)
    train_docs, val_docs = (docs[:8] * 8) + docs[8:68], docs[68:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"]).to(device)

    train_ds = CPTDataset(train_docs, tokenizer, cfg["block_size"])
    val_ds = CPTDataset(val_docs, tokenizer, cfg["block_size"])
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    val_loss_before, ppl_before = eval_ppl(model, val_loader, device)
    comp_before = get_completions(model, tokenizer, device)
    print(f"Before CPT -> Val Loss: {val_loss_before:.4f} | Perplexity: {ppl_before:.2f}")

    optimizer = AdamW(model.parameters(), lr=cfg["learning_rate"])
    train_losses, val_losses = [], [val_loss_before]
    best_val = float("inf")

    for epoch in range(1, cfg["cpt_epochs"] + 1):
        model.train()
        ep_loss = 0.0
        for b in train_loader:
            optimizer.zero_grad()
            loss = model(input_ids=b["input_ids"].to(device), labels=b["labels"].to(device)).loss
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()

        tr_loss = ep_loss / len(train_loader)
        v_loss, v_ppl = eval_ppl(model, val_loader, device)
        train_losses.append(tr_loss)
        val_losses.append(v_loss)
        print(f"Pass {epoch} -> Train Loss: {tr_loss:.4f} | Val Loss: {v_loss:.4f} | PPL: {v_ppl:.2f}")

        model.save_pretrained(f"checkpoints/gpt2-cpt/pass_{epoch}")
        tokenizer.save_pretrained(f"checkpoints/gpt2-cpt/pass_{epoch}")

        if v_loss <= best_val:
            best_val = v_loss
            model.save_pretrained("checkpoints/gpt2-cpt")
            tokenizer.save_pretrained("checkpoints/gpt2-cpt")
        else:
            break

    best_model = AutoModelForCausalLM.from_pretrained("checkpoints/gpt2-cpt").to(device)
    val_loss_after, ppl_after = eval_ppl(best_model, val_loader, device)
    comp_after = get_completions(best_model, tokenizer, device)
    print(f"After CPT -> Val Loss: {val_loss_after:.4f} | Perplexity: {ppl_after:.2f}")

    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o", label="Train Loss")
    plt.plot(range(0, len(val_losses)), val_losses, marker="s", label="Val Loss")
    plt.xlabel("Pass")
    plt.ylabel("Loss")
    plt.title("Continued Pretraining Loss Curve")
    plt.legend()
    plt.savefig("results/cpt_loss_curve.png")
    plt.close()

    json.dump({
        "train_tokens": train_ds.total_tokens,
        "val_tokens": val_ds.total_tokens,
        "perplexity_before": round(ppl_before, 2),
        "perplexity_after": round(ppl_after, 2),
        "completions_before": comp_before,
        "completions_after": comp_after
    }, open("results/cpt_metrics.json", "w"), indent=2)
    print("Stage 1 complete!")

if __name__ == "__main__":
    main()
