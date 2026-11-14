# Project Scoreboard — Resume & Job Description Extractor

| Version | Held-out perplexity | Accuracy | Refusal accuracy | Avg. latency (s) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Small model, base | 113.3 | 0.0% | 0.0% | 0.84 | GPT-2 small (124M) |
| + continued pretraining | 91.3 | 0.0% | 0.0% | 0.7 | 3 passes, lr=5e-05 |
| + SFT | 91.3 | 21.7% | 75.0% | 0.74 | 340 pairs, answer-only loss |
