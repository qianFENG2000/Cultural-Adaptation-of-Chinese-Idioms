# Cultural Adaptation of Chinese Idioms

This repository contains the code, data, prompts, model outputs, automatic evaluation results, and human evaluation results for the project:

**Cultural Adaptation of Chinese Idioms in Round-Trip Translation with Large Language Models**

The project investigates how Chinese idioms are transformed when they are translated from Chinese into English and then back into Chinese by large language models. The main goal is not to improve translation quality, but to use round-trip translation as a diagnostic framework for analyzing semantic preservation, idiomaticity, and cultural meaning.

---

## 1. Project overview

Chinese idioms, or *chengyu*, are compact expressions that often contain figurative, conventionalized, and culturally embedded meanings. Because their meanings are not always directly recoverable from their literal form, they are challenging for machine translation and large language models.

---

## 2. Main tasks

The experiments include two main tasks.

### Task 1: Idiom-level round-trip translation

Task 1 uses standalone Chinese idioms as input.

```text
Chinese idiom -> English translation -> Chinese idiom
```

Example:

```text
画蛇添足 -> gild the lily / draw a snake and add feet -> 画蛇添足
```

The purpose of Task 1 is to test whether models can translate a standalone idiom and then recover a valid Chinese idiom after back-translation.

---

### Task 2: Sentence-level round-trip translation

Task 2 uses Chinese sentences containing idioms as input.

```text
Chinese sentence with idiom -> English translation -> Chinese sentence
```

Task 2 has three English-to-Chinese back-translation conditions:

```text
C1 / v3_v1: natural Chinese sentence
C2 / v3_v2: Chinese sentence with an idiom
C3 / v3_v3: Chinese sentence with the original idiom
```

The purpose of Task 2 is to test how idiom meaning is preserved or transformed when the idiom appears in context.

---

## 3. Repository structure

```text
adaptation_idioms/
├── data/
├── prompts/
├── scripts/
├── outputs/
├── eval/
└── human_eval/
```

Each folder has its own `readme.txt` file with more detailed explanations.

---

## 4. Folder descriptions

### data/

The `data/` folder contains the idiom datasets used in the experiments.

Typical files include:

```text
idiom_base.json
idiom_sample.json
idiom_sample_ann.json
idiom_eval_human.json
```

Main purpose:

- store the full idiom dataset;
- store sampled idioms;
- store manually annotated polarity and transparency labels;
- store the final subset used for human evaluation.

---

### prompts/

The `prompts/` folder contains prompt templates used by the translation script.

Current prompt files include:

```text
task1_zh_en_v1.txt
task1_en_zh_v1.txt
task2_zh_en_v3.txt
task2_en_zh_v1.txt
task2_en_zh_v2.txt
task2_en_zh_v3.txt
```

These prompts define the instructions for each task, direction, and prompt condition.

---

### scripts/

The `scripts/` folder contains all Python scripts used in the project.

Main script categories:

```text
Sampling scripts
Translation generation scripts
Automatic evaluation scripts
Human evaluation preparation scripts
Human evaluation analysis scripts
Table and figure generation scripts
```

Important scripts include:

```text
run_sample.py
run_sample_eval.py
run_translation.py
run_evaluation.py
merge_summary_auto_eval.py
export_auto_eval_tables.py
plot_auto_eval_figures.py
summarize_invalid_and_wrong_language.py
build_idiom_human_eval_dataset.py
summarize_human_eval.py
calculate_kappa.py
calculate_average.py
generate_human_eval_tables.py
plot_human_eval_grouped_bars.py
```

---

### outputs/

The `outputs/` folder contains raw and cleaned model-generated outputs.

It is organized by task:

```text
outputs/
├── task1/
└── task2/
```

Typical cleaned output files include:

```text
outputs/task1/task1_en_zh_{model}_v1_v1_clean.jsonl
outputs/task2/task2_en_zh_{model}_v3_v1_clean.jsonl
outputs/task2/task2_en_zh_{model}_v3_v2_clean.jsonl
outputs/task2/task2_en_zh_{model}_v3_v3_clean.jsonl
```

The cleaned outputs are used for automatic evaluation and human evaluation preparation.

---

### eval/

The `eval/` folder contains automatic evaluation results.

It is organized as:

```text
eval/
├── task1/
├── task2/
└── merged/
```

Main contents:

- item-level automatic evaluation results;
- model-level automatic evaluation summaries;
- merged automatic evaluation tables;
- invalid and wrong-language output summaries;
- automatic evaluation figures.

---

### human_eval/

The `human_eval/` folder contains human evaluation materials and results.

Main contents:

```text
human_eval/input/
human_eval/task1/
human_eval/task2/
human_eval/human_pic/
merged_human_eval.csv
merged_human_eval_avg_by_eval_id.csv
overall_weighted_kappa.csv
detailed_weighted_kappa.csv
human_overall_scores.csv
task-specific human evaluation tables
```

The `input/` folder also contains human evaluation scoring guidelines ending with:

```text
_guidline.txt
```

---

## 5. Models

The project evaluates six open-source instruction-tuned LLMs:

```text
llama31   Llama-3.1-8B-Instruct
mistral   Mistral-7B-Instruct-v0.3
deepseek  deepseek-llm-7b-chat
gemma2    gemma-2-9b-it
glm4      glm-4-9b-chat-hf
qwen25    Qwen2.5-7B-Instruct
```

In human evaluation files, model names are anonymized as:

```text
llama31  -> M01
mistral  -> M02
deepseek -> M03
gemma2   -> M04
glm4     -> M05
qwen25   -> M06
```

---

## 6. Evaluation

The project uses both automatic evaluation and human evaluation.

### Automatic evaluation

Automatic metrics include:

```text
BLEU
ROUGE
BERTScore
SBERT cosine similarity
COMETKiwi
MetricX-QE
COMET
```

General evaluation directions:

```text
zh_en: Chinese input vs. English translation
en_zh: English translation vs. Chinese back-translation
zh_zh: original Chinese input vs. final Chinese back-translation
```

COMETKiwi and MetricX-QE are used as reference-free evaluation metrics. Reference-based COMET is mainly used for Chinese-to-Chinese round-trip evaluation, where the original Chinese input can serve as a same-language reference.

---

### Human evaluation

Human evaluation scores model outputs on task-specific dimensions.

Common dimensions include:

```text
Code-Switching
Grammar
Fluency
Adequacy
```

Task-specific dimensions include:

```text
English Rendering Strategy
Idiom Form
Idiom-related Meaning Translation
Idiom-related Meaning Preservation
Idiom Reintroduction
Target Idiom Use
```

Human evaluation also includes inter-annotator agreement using quadratic weighted Cohen's kappa.

---

## 7. Recommended running order

The full project pipeline is:

```text
Step 1   Prepare sampled data
Step 2   Run LLM translation
Step 3   Clean model outputs
Step 4   Run automatic evaluation
Step 5   Merge automatic evaluation summaries
Step 6   Export automatic evaluation tables
Step 7   Generate automatic evaluation figures
Step 8   Prepare human evaluation files
Step 9   Conduct human annotation
Step 10  Merge human evaluation annotations
Step 11  Calculate inter-annotator agreement
Step 12  Average annotator scores
Step 13  Generate human evaluation tables
Step 14  Generate human evaluation figures
```

---

## 8. Main commands

### 8.1 Sample idioms

```bash
python scripts/run_sample.py
python scripts/run_sample_eval.py
```

---

### 8.2 Run Task 1 translation

```bash
python scripts/run_translation.py --task task1 --direction zh_en --model all --prompt_version v1

python scripts/run_translation.py --task task1 --direction en_zh --model all --zh_en_prompt_version v1 --prompt_version v1
```

---

### 8.3 Run Task 2 translation

```bash
python scripts/run_translation.py --task task2 --direction zh_en --model all --prompt_version v3

python scripts/run_translation.py --task task2 --direction en_zh --model all --zh_en_prompt_version v3 --prompt_version v1

python scripts/run_translation.py --task task2 --direction en_zh --model all --zh_en_prompt_version v3 --prompt_version v2

python scripts/run_translation.py --task task2 --direction en_zh --model all --zh_en_prompt_version v3 --prompt_version v3
```

---

### 8.4 Clean generated outputs

The downstream scripts require cleaned output files ending with:

```text
_clean.jsonl
```

The cleaning script is not included in the current uploaded folder, so this step must be completed separately before evaluation.

---

### 8.5 Run automatic evaluation

Task 1:

```bash
python scripts/run_evaluation.py --task task1 --eval_type zh_en --model all --zh_en_prompt_version v1 --en_zh_prompt_version v1

python scripts/run_evaluation.py --task task1 --eval_type en_zh --model all --zh_en_prompt_version v1 --en_zh_prompt_version v1

python scripts/run_evaluation.py --task task1 --eval_type zh_zh --model all --zh_en_prompt_version v1 --en_zh_prompt_version v1
```

Task 2:

```bash
python scripts/run_evaluation.py --task task2 --eval_type zh_en --model all --zh_en_prompt_version v3 --en_zh_prompt_version v1

python scripts/run_evaluation.py --task task2 --eval_type en_zh --model all --zh_en_prompt_version v3 --en_zh_prompt_version v1

python scripts/run_evaluation.py --task task2 --eval_type zh_zh --model all --zh_en_prompt_version v3 --en_zh_prompt_version v1

python scripts/run_evaluation.py --task task2 --eval_type en_zh --model all --zh_en_prompt_version v3 --en_zh_prompt_version v2

python scripts/run_evaluation.py --task task2 --eval_type zh_zh --model all --zh_en_prompt_version v3 --en_zh_prompt_version v2

python scripts/run_evaluation.py --task task2 --eval_type en_zh --model all --zh_en_prompt_version v3 --en_zh_prompt_version v3

python scripts/run_evaluation.py --task task2 --eval_type zh_zh --model all --zh_en_prompt_version v3 --en_zh_prompt_version v3
```

---

### 8.6 Merge and export automatic evaluation results

```bash
python scripts/merge_summary_auto_eval.py
python scripts/export_auto_eval_tables.py
python scripts/plot_auto_eval_figures.py
python scripts/summarize_invalid_and_wrong_language.py
```

---

### 8.7 Prepare and analyze human evaluation

```bash
python scripts/build_idiom_human_eval_dataset.py
python scripts/summarize_human_eval.py
python scripts/calculate_kappa.py
python scripts/calculate_average.py
python scripts/generate_human_eval_tables.py
python scripts/plot_human_eval_grouped_bars.py
```

---

## 9. Important notes

### Cleaned outputs are required

Most evaluation scripts require `*_clean.jsonl` files. If the cleaned files are missing, the evaluation scripts will fail.

### Some generated files are large

Model outputs and evaluation results may be large. Depending on the GitHub repository policy, it may be better to keep very large generated files outside the repository or track them with Git LFS.

### Prompt versions matter

Prompt files should not be overwritten after experiments are run. If the prompt changes, create a new prompt version instead.

### Human evaluation files should be preserved

The original annotation files, merged human evaluation files, and kappa files are important for reproducibility.

---

## 10. Reproducibility

Random sampling scripts use a fixed random seed:

```text
RANDOM_SEED = 42
```

This helps ensure that sampled idioms can be reproduced.

For full reproducibility, keep the following fixed:

```text
data files
prompt files
model versions
generation parameters
cleaning rules
evaluation scripts
human evaluation guidelines
```

---

## 11. Citation and data source note

The idiom data used in this project is based on an open-source Chinese idiom dataset compiled from the `chinese-xinhua` GitHub repository.

This dataset should be described as a public dictionary-like computational resource, not as the official Xinhua Dictionary.

