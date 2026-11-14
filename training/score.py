import os, re, csv, time, json, yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REFUSAL_IDS = {27, 28, 29, 30}
REFUSAL_WORDS = ["i don't know", "i don't have that information", "anonymized", "not state", "refuses", "neither"]

KEY_FACTS = {
    1: ["3.72"], 2: ["june 2026"], 3: ["ai-900", "rh104"], 4: ["14"], 5: ["postgresql", "sql"],
    6: ["m.sc", "2027"], 7: ["1", "ieee"], 8: ["opencv", "yolov8", "mobilenetv2", "keras"],
    9: ["2.85"], 10: ["24"], 11: ["business", "2024"], 12: ["3.2"], 13: ["12"],
    14: ["faiss", "bm25", "rag"], 15: ["yes", "arabic"], 16: ["yes", "3.72", "3.2"],
    17: ["6", "4", "sql"], 18: ["yes", "14", "docker", "postgresql"], 19: ["17", "6"],
    20: ["cv_01", "cv_03"], 21: ["2.85", "3.2"], 22: ["cv_03", "3.85"],
    23: ["no", "marketing", "seo"], 24: ["cv_01", "cv_04"], 25: ["cv_01", "ai-900"],
    26: ["none", "sql", "gpa"]
}

def load_test_questions():
    os.makedirs("tests", exist_ok=True)
    lines = [l.strip() for l in open("eval/seed_test_set.md", encoding="utf-8") if l.strip()]
    questions = []
    for line in lines[1:]:
        c1 = line.find(",")
        qid = int(line[:c1])
        rest = line[c1 + 1:]
        c_last = rest.rfind(",")
        diff = rest[c_last + 1:].strip()
        rest = rest[:c_last]
        c_src = rest.rfind(",")
        src = rest[c_src + 1:].strip()
        qa = rest[:c_src]
        sep = "?," if "?," in qa else ".,"
        idx = qa.find(sep)
        questions.append({
            "id": qid,
            "question": qa[:idx + 1].strip(),
            "expected_answer": qa[idx + 2:].strip(),
            "source": src,
            "difficulty": diff
        })
    with open("tests/test_questions.csv", "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=["id", "question", "expected_answer", "source", "difficulty"])
        writer.writeheader()
        writer.writerows(questions)
    return questions

def ask_model(model, tokenizer, question, max_new_tokens, device):
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    latency = time.time() - t0
    text = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=False)
    for stop in ["<|end|>", "<|user|>", "<|endoftext|>"]:
        if stop in text:
            text = text.split(stop)[0]
    return " ".join(text.split()), latency

def grade_answer(qid, expected, answer):
    ans = answer.lower().strip()
    exp = expected.lower().strip()
    refused = any(w in ans for w in REFUSAL_WORDS)

    if qid in REFUSAL_IDS:
        if refused:
            return 1.0, 1.0, 1.0, True, "None", "Correctly refused unanswerable question"
        return 0.0, 0.0, 0.0, False, "Refusal error", "Failed to refuse; made up answer"

    if refused:
        return 0.0, 0.0, 0.0, False, "Refusal error", "Refused an answerable question"

    if len(ans) < 2 or "<|assistant|>" in ans or ans.count("the ") > 8:
        return 0.0, 0.0, 0.0, False, "Format failure", "Ignored QA format or repeated text"

    keys = KEY_FACTS.get(qid, [])
    hits = sum(1 for k in keys if k in ans) / max(1, len(keys))
    s1 = 1.0 if (exp in ans or hits >= 0.75) else (0.5 if hits >= 0.35 else 0.0)

    exp_words = set(re.findall(r"[a-z0-9_.]+", exp))
    ans_words = set(re.findall(r"[a-z0-9_.]+", ans))
    overlap = len(exp_words & ans_words) / max(1, len(exp_words))
    s2 = 1.0 if (hits >= 0.65 or overlap >= 0.55) else (0.5 if (hits >= 0.25 or overlap >= 0.25) else 0.0)

    final_s = round((s1 + s2) / 2.0 * 2) / 2.0
    if final_s == 1.0:
        return s1, s2, 1.0, False, "None", "Correct answer"
    if final_s == 0.5:
        return s1, s2, 0.5, False, "Knowledge missing", "Partly correct; missed some details"

    err = "Hallucination" if re.search(r"\d+|cv_\d+|jd_\d+", ans) else "Knowledge missing"
    note = "Invented wrong facts" if err == "Hallucination" else "Fact in docs but model did not learn it"
    return s1, s2, 0.0, False, err, note

def run_evaluation(tag, model_path, questions, cfg, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()

    rows, latencies = [], []
    total_score, refusal_correct, agreements = 0.0, 0, 0
    errors = {"Format failure": 0, "Hallucination": 0, "Knowledge missing": 0, "Refusal error": 0, "Test-case problem": 0}

    for q in questions:
        qid = q["id"]
        ans, lat = ask_model(model, tokenizer, q["question"], cfg["max_new_tokens"], device)
        latencies.append(lat)
        s1, s2, score, ref_ok, err_cat, note = grade_answer(qid, q["expected_answer"], ans)

        if s1 == s2:
            agreements += 1
        total_score += score
        if qid in REFUSAL_IDS and ref_ok:
            refusal_correct += 1
        if err_cat in errors:
            errors[err_cat] += 1

        rows.append({
            "id": qid, "question": q["question"], "expected_answer": q["expected_answer"],
            "model_answer": ans, "score": score, "error_category": err_cat, "note": note
        })

    with open(f"results/{tag}_results.csv", "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    acc = round((total_score / len(questions)) * 100, 1)
    ref_acc = round((refusal_correct / len(REFUSAL_IDS)) * 100, 1)
    avg_lat = round(sum(latencies) / len(latencies), 2)
    agree = round((agreements / len(questions)) * 100, 1)
    print(f"[{tag.upper()}] Accuracy: {acc}% | Refusal Accuracy: {ref_acc}% | Avg Latency: {avg_lat}s | Grader Agreement: {agree}%")
    return {"accuracy": acc, "refusal_accuracy": ref_acc, "latency": avg_lat, "agreement": agree, "errors": errors}

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    questions = load_test_questions()

    cpt_m = json.load(open("results/cpt_metrics.json"))
    sft_m = json.load(open("results/sft_metrics.json"))
    ppl_base, ppl_cpt = cpt_m["perplexity_before"], cpt_m["perplexity_after"]

    base_res = run_evaluation("base", cfg["model_name"], questions, cfg, device)
    cpt_res = run_evaluation("cpt", "checkpoints/gpt2-cpt", questions, cfg, device)
    sft_res = run_evaluation("sft", "checkpoints/gpt2-sft", questions, cfg, device)

    scoreboard = f"""# Project Scoreboard — Resume & Job Description Extractor

| Version | Held-out perplexity | Accuracy | Refusal accuracy | Avg. latency (s) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Small model, base | {ppl_base} | {base_res['accuracy']}% | {base_res['refusal_accuracy']}% | {base_res['latency']} | GPT-2 small (124M) |
| + continued pretraining | {ppl_cpt} | {cpt_res['accuracy']}% | {cpt_res['refusal_accuracy']}% | {cpt_res['latency']} | {cfg['cpt_epochs']} passes, lr={cfg['learning_rate']} |
| + SFT | {ppl_cpt} | {sft_res['accuracy']}% | {sft_res['refusal_accuracy']}% | {sft_res['latency']} | {sft_m['total_pairs']} pairs, answer-only loss |
"""
    open("SCOREBOARD.md", "w", encoding="utf-8").write(scoreboard)
    json.dump({"base": base_res, "cpt": cpt_res, "sft": sft_res}, open("results/scoreboard_summary.json", "w"), indent=2)

    print("\n" + scoreboard)
    print("SFT Error Breakdown:", sft_res["errors"])

if __name__ == "__main__":
    main()
