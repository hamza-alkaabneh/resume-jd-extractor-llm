# Resume & Job Description Extractor (Phase 1)

**Course:** Special Topics in Data Science — Building LLM Systems: From Models to Production  
**Phase:** Phase 1 — Problem and Data (`v0.1-phase1`)

## Three Core Phase 1 Questions
1. **Who is the user, and what task do they need help with?**  
   Technical recruiters who need to extract structured qualifications from candidate CVs and verify mandatory Job Description constraints.
2. **What data will the system use to help them, and do you already have it?**  
   A corpus of 60 anonymized resumes and 15 technical Job Descriptions (with 5 sample CVs and 3 sample JDs committed in `/data/samples/`).
3. **How will you know the system works?**  
   By evaluating exact/semantic extraction accuracy, source citation precision, and abstention accuracy on our 30-row ground-truth benchmark (`/eval/seed_test_set.csv`).

## Repository Structure
- `docs/proposal.pdf` — 2-Page Project Proposal
- `docs/data_card.pdf` — 1-Page Data Card
- `eval/seed_test_set.csv` — 30-Row Seed Test Set (Easy, Medium, Hard, and Refusal cases)
- `data/samples/` — 5 Anonymized Sample CVs (`CV_01`–`CV_05`) and 3 Job Descriptions (`JD_01`–`JD_03`)
