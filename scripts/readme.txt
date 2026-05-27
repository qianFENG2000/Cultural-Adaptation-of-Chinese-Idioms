# Scripts README

This README explains the recommended running order of the scripts in the `scripts/` folder and how each script fits into the whole project pipeline.

Overall pipeline:

1. Prepare idiom samples for human evaluation
2. Run LLM translation experiments
3. Clean generated outputs
4. Run automatic evaluation
5. Summarize automatic evaluation results
6. Prepare human evaluation files
7. Merge and summarize human evaluation annotations
8. Generate tables and figures



============================================================
 Full running order
============================================================

Recommended order:

Step 1   run_sample.py
Step 2   manually annotate idiom_sample.json
Step 3   run_sample_eval.py
Step 4   run_translation.py
Step 5   manually clean generated translation outputs
Step 6   run_evaluation.py
Step 7   merge_summary_auto_eval.py
Step 8   export_auto_eval_tables.py
Step 9   plot_auto_eval_figures.py
Step 10  summarize_invalid_and_wrong_language.py
Step 11  build_idiom_human_eval_dataset.py
Step 12  manually conduct human annotation
Step 13  summarize_human_eval.py
Step 14  calculate_kappa.py
Step 15  calculate_average.py
Step 16  generate_human_eval_tables.py
Step 17  plot_human_eval_grouped_bars.py

============================================================
Part A. Data sampling
============================================================

------------------------------------------------------------
Step 1. Randomly sample idioms
------------------------------------------------------------

Script:

python scripts/run_sample.py

Input:

data/idiom_base.json

Output:

data/idiom_sample.json

Purpose:

This script randomly samples 100 idioms from the full idiom dataset.

The output keeps the following fields:

- sentence
- index
- source_idiom

------------------------------------------------------------
Step 2. Manually annotate sampled idioms
------------------------------------------------------------

After running `run_sample.py`, manually add annotation labels to the sampled idioms.

Expected file:

data/idiom_sample_ann.json

This file should contain at least:

- index
- source_idiom
- sentence
- polarity
- transparency

Example labels:

polarity: positive / neutral / negative
transparency: high / low

------------------------------------------------------------
Step 3. Select idioms for human evaluation
------------------------------------------------------------

Script:

python scripts/run_sample_eval.py

Input:

data/idiom_sample_ann.json

Output:

data/idiom_eval_human.json

Purpose:

This script samples idioms by `polarity + transparency` groups for human evaluation.

Important:

The current script uses:

SAMPLE_SIZE = 2

This means it samples 2 idioms from each polarity-transparency group.

If the final thesis setting requires 5 idioms per group, change it to:

SAMPLE_SIZE = 5

before running the script.

============================================================
Part B. Translation generation
============================================================

------------------------------------------------------------
Step 4. Run LLM translation
------------------------------------------------------------

Script:

python scripts/run_translation.py

This script supports:

- task1
- task2


and two directions:

- zh_en
- en_zh

In the main thesis pipeline, the important tasks are usually:

- task1: idiom-level round-trip translation
- task2: sentence-level round-trip translation

------------------------------------------------------------
Step 4.1. Task 1: Chinese idiom to English
------------------------------------------------------------

Command:

python scripts/run_translation.py \
  --task task1 \
  --direction zh_en \
  --model all \
  --prompt_version v1

Input:

data/idiom_base.json

Output:

outputs/task1/task1_zh_en_{model}_v1.jsonl

------------------------------------------------------------
Step 4.2. Task 1: English back to Chinese idiom
------------------------------------------------------------

Command:

python scripts/run_translation.py \
  --task task1 \
  --direction en_zh \
  --model all \
  --zh_en_prompt_version v1 \
  --prompt_version v1

Input:

outputs/task1/task1_zh_en_{model}_v1_clean.jsonl

Output:

outputs/task1/task1_en_zh_{model}_v1_v1.jsonl

------------------------------------------------------------
Step 4.3. Task 2: Chinese sentence to English
------------------------------------------------------------

Command:

python scripts/run_translation.py \
  --task task2 \
  --direction zh_en \
  --model all \
  --prompt_version v3

Input:

data/idiom_base.json

Output:

outputs/task2/task2_zh_en_{model}_v3.jsonl

------------------------------------------------------------
Step 4.4. Task 2.1: English back to natural Chinese sentence
------------------------------------------------------------

Command:

python scripts/run_translation.py \
  --task task2 \
  --direction en_zh \
  --model all \
  --zh_en_prompt_version v3 \
  --prompt_version v1

Input:

outputs/task2/task2_zh_en_{model}_v3_clean.jsonl

Output:

outputs/task2/task2_en_zh_{model}_v3_v1.jsonl

This corresponds to:

Task 2.1 / C1: natural Chinese sentence

------------------------------------------------------------
Step 4.5. Task 2.2: English back to Chinese sentence with an idiom
------------------------------------------------------------

Command:

python scripts/run_translation.py \
  --task task2 \
  --direction en_zh \
  --model all \
  --zh_en_prompt_version v3 \
  --prompt_version v2

Output:

outputs/task2/task2_en_zh_{model}_v3_v2.jsonl

This corresponds to:

Task 2.2 / C2: Chinese sentence with an idiom

------------------------------------------------------------
Step 4.6. Task 2.3: English back to Chinese sentence with the original idiom
------------------------------------------------------------

Command:

python scripts/run_translation.py \
  --task task2 \
  --direction en_zh \
  --model all \
  --zh_en_prompt_version v3 \
  --prompt_version v3

Output:

outputs/task2/task2_en_zh_{model}_v3_v3.jsonl

This corresponds to:

Task 2.3 / C3: Chinese sentence with the original idiom

------------------------------------------------------------
Step 5. Clean generated outputs
------------------------------------------------------------

Important:

The later evaluation scripts expect cleaned files with filenames ending in:

_clean.jsonl

For example:

outputs/task1/task1_en_zh_llama31_v1_v1_clean.jsonl
outputs/task2/task2_en_zh_llama31_v3_v1_clean.jsonl
outputs/task2/task2_en_zh_llama31_v3_v2_clean.jsonl
outputs/task2/task2_en_zh_llama31_v3_v3_clean.jsonl

Manually clean.

The cleaned files should include invalidity and language-status fields such as:

- is_invalid_en
- invalid_type_en
- language_status_en
- is_invalid_zh
- invalid_type_zh
- language_status_zh

============================================================
Part C. Automatic evaluation
============================================================

------------------------------------------------------------
Step 6. Run automatic evaluation
------------------------------------------------------------

Script:

python scripts/run_evaluation.py

This script evaluates cleaned round-trip outputs using automatic metrics.

Metrics include:

- BLEU
- ROUGE
- BERTScore
- SBERT cosine similarity
- COMETKiwi
- MetricX-QE
- COMET

Note:

COMET is only used for zh_zh evaluation.

------------------------------------------------------------
Step 6.1. Task 1 automatic evaluation
------------------------------------------------------------

Task 1 zh_en:

python scripts/run_evaluation.py \
  --task task1 \
  --eval_type zh_en \
  --model all \
  --zh_en_prompt_version v1 \
  --en_zh_prompt_version v1

Task 1 en_zh:

python scripts/run_evaluation.py \
  --task task1 \
  --eval_type en_zh \
  --model all \
  --zh_en_prompt_version v1 \
  --en_zh_prompt_version v1

Task 1 zh_zh:

python scripts/run_evaluation.py \
  --task task1 \
  --eval_type zh_zh \
  --model all \
  --zh_en_prompt_version v1 \
  --en_zh_prompt_version v1

Outputs:

eval/task1/task1_detail_{eval_type}_{model}_v1_v1.json
eval/task1/task1_summary_{eval_type}_v1_v1.json

------------------------------------------------------------
Step 6.2. Task 2 automatic evaluation
------------------------------------------------------------

Task 2 zh_en:

python scripts/run_evaluation.py \
  --task task2 \
  --eval_type zh_en \
  --model all \
  --zh_en_prompt_version v3 \
  --en_zh_prompt_version v1

Task 2.1 / C1 en_zh:

python scripts/run_evaluation.py \
  --task task2 \
  --eval_type en_zh \
  --model all \
  --zh_en_prompt_version v3 \
  --en_zh_prompt_version v1

Task 2.1 / C1 zh_zh:

python scripts/run_evaluation.py \
  --task task2 \
  --eval_type zh_zh \
  --model all \
  --zh_en_prompt_version v3 \
  --en_zh_prompt_version v1

Task 2.2 / C2 en_zh:

python scripts/run_evaluation.py \
  --task task2 \
  --eval_type en_zh \
  --model all \
  --zh_en_prompt_version v3 \
  --en_zh_prompt_version v2

Task 2.2 / C2 zh_zh:

python scripts/run_evaluation.py \
  --task task2 \
  --eval_type zh_zh \
  --model all \
  --zh_en_prompt_version v3 \
  --en_zh_prompt_version v2

Task 2.3 / C3 en_zh:

python scripts/run_evaluation.py \
  --task task2 \
  --eval_type en_zh \
  --model all \
  --zh_en_prompt_version v3 \
  --en_zh_prompt_version v3

Task 2.3 / C3 zh_zh:

python scripts/run_evaluation.py \
  --task task2 \
  --eval_type zh_zh \
  --model all \
  --zh_en_prompt_version v3 \
  --en_zh_prompt_version v3

Outputs:

eval/task2/task2_detail_{eval_type}_{model}_{zh_en_prompt_version}_{en_zh_prompt_version}.json
eval/task2/task2_summary_{eval_type}_{zh_en_prompt_version}_{en_zh_prompt_version}.json

============================================================
Part D. Automatic evaluation summaries, tables, and figures
============================================================

------------------------------------------------------------
Step 7. Merge automatic evaluation summaries
------------------------------------------------------------

Script:

python scripts/merge_summary_auto_eval.py

Input:

eval/task1/*summary*.json
eval/task2/*summary*.json

Output:

eval/merged/summary_auto_eval.csv

Purpose:

This script merges all automatic evaluation summary JSON files into one long-format CSV table.


------------------------------------------------------------
Step 8. Export automatic evaluation tables
------------------------------------------------------------

Script:

python scripts/export_auto_eval_tables.py

Input:

eval/merged/summary_auto_eval.csv

Outputs:

eval/merged/auto_eval_full_table.csv
eval/merged/grouped_table_index.csv
eval/merged/{task}_{eval_type}_{zh_en_prompt_version}_{en_zh_prompt_version}.csv

Purpose:

This script creates cleaned CSV tables for reporting automatic evaluation results.

------------------------------------------------------------
Step 9. Plot automatic evaluation figures
------------------------------------------------------------

Script:

python scripts/plot_auto_eval_figures.py

Input:

eval/merged/auto_eval_full_table.csv

Outputs:

eval/merged/auto_pic/fig1_auto_zh_en.pdf
eval/merged/auto_pic/fig2_auto_en_zh.pdf
eval/merged/auto_pic/fig3_auto_zh_zh.pdf

Purpose:

This script creates line plots for automatic evaluation results.

------------------------------------------------------------
Step 10. Summarize invalid outputs and wrong-language outputs
------------------------------------------------------------

Script:

python scripts/summarize_invalid_and_wrong_language.py

Input:

outputs/task1/*_clean.jsonl
outputs/task2/*_clean.jsonl

Outputs:

eval/merged/overall_distribution_by_file_eval_type.csv
eval/merged/invalid_type_breakdown_by_task_eval_condition_model.csv

Purpose:

This script summarizes invalid outputs and wrong-language outputs by task, evaluation type, condition, and model.

============================================================
Part E. Human evaluation preparation
============================================================

------------------------------------------------------------
Step 11. Build human evaluation datasets
------------------------------------------------------------

Script:

python scripts/build_idiom_human_eval_dataset.py

Input:

data/idiom_eval_human.json
outputs/task1/task1_en_zh_{model}_v1_v1_clean.jsonl
outputs/task2/task2_en_zh_{model}_v3_v1_clean.jsonl
outputs/task2/task2_en_zh_{model}_v3_v2_clean.jsonl
outputs/task2/task2_en_zh_{model}_v3_v3_clean.jsonl

Outputs:

human_eval/input/task1_zh_en_human_eval.json
human_eval/input/task1_zh_en_human_eval.xlsx

human_eval/input/task1_en_zh_human_eval.json
human_eval/input/task1_en_zh_human_eval.xlsx

human_eval/input/task1_zh_zh_human_eval.json
human_eval/input/task1_zh_zh_human_eval.xlsx

human_eval/input/task2_zh_en_v3_v1_human_eval.json
human_eval/input/task2_zh_en_v3_v1_human_eval.xlsx

human_eval/input/task2_en_zh_v3_v1_human_eval.json
human_eval/input/task2_en_zh_v3_v1_human_eval.xlsx

human_eval/input/task2_zh_zh_v3_v1_human_eval.json
human_eval/input/task2_zh_zh_v3_v1_human_eval.xlsx

human_eval/input/task2_en_zh_v3_v2_human_eval.json
human_eval/input/task2_en_zh_v3_v2_human_eval.xlsx

human_eval/input/task2_zh_zh_v3_v2_human_eval.json
human_eval/input/task2_zh_zh_v3_v2_human_eval.xlsx

human_eval/input/task2_en_zh_v3_v3_human_eval.json
human_eval/input/task2_en_zh_v3_v3_human_eval.xlsx

human_eval/input/task2_zh_zh_v3_v3_human_eval.json
human_eval/input/task2_zh_zh_v3_v3_human_eval.xlsx

Purpose:

This script prepares anonymized human evaluation files.

The model names are anonymized as:

llama31  -> M01
mistral  -> M02
deepseek -> M03
gemma2   -> M04
glm4     -> M05
qwen25   -> M06

------------------------------------------------------------
Step 12. Conduct human annotation
------------------------------------------------------------

The generated Excel files in:

human_eval/input/

should be given to annotators.

After annotation, rename the files with annotator IDs:

_a1.xlsx
_a2.xlsx

Example:

task1_zh_en_human_eval_a1.xlsx
task1_zh_en_human_eval_a2.xlsx
task2_en_zh_v3_v1_human_eval_a1.xlsx
task2_en_zh_v3_v1_human_eval_a2.xlsx

Then place the annotated files into:

human_eval/task1/
human_eval/task2/

The script `summarize_human_eval.py` expects annotated Excel files in these folders.

============================================================
Part F. Human evaluation summaries, agreement, tables, and figures
============================================================

------------------------------------------------------------
Step 13. Merge human evaluation annotations
------------------------------------------------------------

Script:

python scripts/summarize_human_eval.py

Inputs:

human_eval/task1/*.xlsx
human_eval/task2/*.xlsx
data/idiom_sample_ann.json

Output:

human_eval/merged_human_eval.csv

Purpose:

This script merges all annotated Excel files into one CSV file.

It also maps anonymous model IDs back to model names.

------------------------------------------------------------
Step 14. Calculate inter-annotator agreement
------------------------------------------------------------

Script:

python scripts/calculate_kappa.py

Input:

human_eval/merged_human_eval.csv

Outputs:

human_eval/overall_weighted_kappa.csv
human_eval/detailed_weighted_kappa.csv

Purpose:

This script calculates quadratic weighted Cohen's kappa between annotators `a1` and `a2`.

It computes agreement for:

- code_switching
- grammar
- fluency
- adequacy
- task_specific_dim

------------------------------------------------------------
Step 15. Average human scores by evaluation item
------------------------------------------------------------

Script:

python scripts/calculate_average.py

Input:

human_eval/merged_human_eval.csv

Output:

human_eval/merged_human_eval_avg_by_eval_id.csv

Purpose:

This script averages scores from different annotators for the same `eval_id`.

The averaged file is used for later human evaluation tables.

------------------------------------------------------------
Step 16. Generate human evaluation tables
------------------------------------------------------------

Script:

python scripts/generate_human_eval_tables.py

Input:

human_eval/merged_human_eval_avg_by_eval_id.csv

Outputs:

human_eval/human_overall_scores.csv
human_eval/human_task1_zh_en.csv
human_eval/human_task1_en_zh.csv
human_eval/human_task1_zh_zh.csv
human_eval/human_task2_zh_en.csv
human_eval/human_task2_c1.csv
human_eval/human_task2_c2.csv
human_eval/human_task2_c3.csv
human_eval/human_transparency_overall.csv
human_eval/human_transparency_task_specific.csv
human_eval/human_task2_idiom_diagnostic.csv

Purpose:

This script creates human evaluation result tables for thesis writing and presentation.

------------------------------------------------------------
Step 17. Plot human evaluation figures
------------------------------------------------------------

Script:

python scripts/plot_human_eval_grouped_bars.py

Output directory:

human_eval/human_pic/

Outputs include:

fig_human_task1_zh_en_bar.pdf
fig_human_task1_en_zh_bar.pdf
fig_human_task1_zh_zh_bar.pdf
fig_human_task2_zh_en_bar.pdf
fig_human_task2_c1_faceted_bar.pdf
fig_human_task2_c2_faceted_bar.pdf
fig_human_task2_c3_faceted_bar.pdf
fig_human_transparency_overall_bar.pdf
fig_human_transparency_task_specific_bar.pdf

Purpose:

This script creates grouped bar charts for human evaluation results.

Important:

The current script contains hard-coded human evaluation values.
If the human evaluation scores change, the hard-coded values in this script should be updated, or the script should be modified to read directly from the generated CSV tables.

============================================================
3. Recommended minimal command sequence
============================================================

For the main thesis pipeline, the core commands are:

# 1. Sample idioms
python scripts/run_sample.py

# 2. After manually creating data/idiom_sample_ann.json
python scripts/run_sample_eval.py

# 3. Run Task 1 translation
python scripts/run_translation.py --task task1 --direction zh_en --model all --prompt_version v1
python scripts/run_translation.py --task task1 --direction en_zh --model all --zh_en_prompt_version v1 --prompt_version v1

# 4. Run Task 2 translation
python scripts/run_translation.py --task task2 --direction zh_en --model all --prompt_version v3
python scripts/run_translation.py --task task2 --direction en_zh --model all --zh_en_prompt_version v3 --prompt_version v1
python scripts/run_translation.py --task task2 --direction en_zh --model all --zh_en_prompt_version v3 --prompt_version v2
python scripts/run_translation.py --task task2 --direction en_zh --model all --zh_en_prompt_version v3 --prompt_version v3

# 5. Clean generated outputs manually or with a separate cleaning script
# Make sure all required files are saved as *_clean.jsonl

# 6. Run automatic evaluation
python scripts/run_evaluation.py --task task1 --eval_type zh_en --model all --zh_en_prompt_version v1 --en_zh_prompt_version v1
python scripts/run_evaluation.py --task task1 --eval_type en_zh --model all --zh_en_prompt_version v1 --en_zh_prompt_version v1
python scripts/run_evaluation.py --task task1 --eval_type zh_zh --model all --zh_en_prompt_version v1 --en_zh_prompt_version v1

python scripts/run_evaluation.py --task task2 --eval_type zh_en --model all --zh_en_prompt_version v3 --en_zh_prompt_version v1
python scripts/run_evaluation.py --task task2 --eval_type en_zh --model all --zh_en_prompt_version v3 --en_zh_prompt_version v1
python scripts/run_evaluation.py --task task2 --eval_type zh_zh --model all --zh_en_prompt_version v3 --en_zh_prompt_version v1
python scripts/run_evaluation.py --task task2 --eval_type en_zh --model all --zh_en_prompt_version v3 --en_zh_prompt_version v2
python scripts/run_evaluation.py --task task2 --eval_type zh_zh --model all --zh_en_prompt_version v3 --en_zh_prompt_version v2
python scripts/run_evaluation.py --task task2 --eval_type en_zh --model all --zh_en_prompt_version v3 --en_zh_prompt_version v3
python scripts/run_evaluation.py --task task2 --eval_type zh_zh --model all --zh_en_prompt_version v3 --en_zh_prompt_version v3

# 7. Automatic evaluation summaries and figures
python scripts/merge_summary_auto_eval.py
python scripts/export_auto_eval_tables.py
python scripts/plot_auto_eval_figures.py
python scripts/summarize_invalid_and_wrong_language.py

# 8. Human evaluation preparation
python scripts/build_idiom_human_eval_dataset.py

# 9. After human annotators complete Excel files
python scripts/summarize_human_eval.py
python scripts/calculate_kappa.py
python scripts/calculate_average.py
python scripts/generate_human_eval_tables.py
python scripts/plot_human_eval_grouped_bars.py

