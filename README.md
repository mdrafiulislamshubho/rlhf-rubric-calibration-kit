# RLHF Rubric Calibration Kit

Deterministic, zero-dependency Python toolkit for auditing inter-annotator agreement (IAA), rubric score drift, and Bangla-English code-switch consistency in Reinforcement Learning from Human Feedback (RLHF) preference datasets.

**Author:** [MD Rafiul Islam Shubho](https://orcid.org/0009-0009-9956-4511) — Co-Founder, Managing Director & Applied AI / RLHF Evaluation Specialist, [Vision Flow Group](https://visionflowgroup.com/)

## Why This Exists

Multilingual RLHF and Speech-to-Speech (S2S) evaluation pipelines frequently suffer from silent annotator drift when bilingual raters score code-switched Bangla-English prompts across subjective rubrics (fluency, factuality, prosody, instruction adherence). This toolkit provides deterministic statistical checks without external ML dependencies.

## Capabilities

- **Pairwise Cohen's Kappa** (`cohens_kappa`) for two-rater preference agreement auditing.
- **Multi-Rater Fleiss' Kappa** (`fleiss_kappa`) across `N` items and `K` categorical rubric labels.
- **High-Drift Flagging** for rubric score deltas `>= 2.0` on 5-point evaluation scales.
- **Bengali/Latin Script Balance & Unicode NFC Verification** (`script_mix_profile`) to surface hidden encoding or code-switching anomalies before model fine-tuning.

## Quick Start

```bash
# Run built-in bilingual RLHF self-test
python3 rubric_auditor.py --demo

# Audit a JSONL annotation batch
python3 rubric_auditor.py --input annotations.jsonl
```

### Expected JSONL Schema

```json
{"id": "bn-001", "rater_a": "A_better", "rater_b": "A_better", "score_a": 4.5, "score_b": 4.5, "response": "বাংলা AI evaluation rubric calibration"}
```

## Related Work & Verification

- **ORCID:** [0009-0009-9956-4511](https://orcid.org/0009-0009-9956-4511)
- **Crunchbase:** [MD Rafiul Islam Shubho](https://www.crunchbase.com/person/md-rafiul-islam-shubho)
- **Companion Benchmark:** [bangla-s2s-prosody-bench](https://github.com/mdrafiulislamshubho/bangla-s2s-prosody-bench)
- **Personal Portfolio:** [mdrafiulislamshubho.github.io](https://mdrafiulislamshubho.github.io/)
- **Organization:** [Vision Flow Group](https://visionflowgroup.com/md-rafiul-islam-shubho/)

## License

MIT License
