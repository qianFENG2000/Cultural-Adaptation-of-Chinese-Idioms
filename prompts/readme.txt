README for prompts/

This folder stores the prompt templates used for the translation experiments.

The folder currently contains the following prompt files:

prompts/
├── task1_zh_en_v1.txt
├── task1_en_zh_v1.txt
├── task2_zh_en_v3.txt
├── task2_en_zh_v1.txt
├── task2_en_zh_v2.txt
└── task2_en_zh_v3.txt


1. Purpose of this folder
-------------------------

The prompts/ folder contains text prompt templates used by scripts/run_translation.py.

Each prompt file defines the instruction given to the model for a specific task, translation direction, and prompt version.

The general naming pattern is:

{task}_{direction}_{version}.txt

Where:

- task: task1 or task2
- direction: zh_en or en_zh
- version: prompt version, such as v1, v2, or v3


2. Task and direction naming
----------------------------

task1:
Idiom-level round-trip translation.

The input is a standalone Chinese idiom.

The full process is:

Chinese idiom -> English translation -> Chinese idiom


task2:
Sentence-level round-trip translation.

The input is a Chinese sentence containing an idiom.

The full process is:

Chinese sentence containing an idiom -> English translation -> Chinese sentence


zh_en:
Chinese-to-English translation.

en_zh:
English-to-Chinese back-translation.


3. Prompt files
---------------

task1_zh_en_v1.txt

This prompt is used for Task 1 Chinese-to-English translation.

Input:
- source_text

Expected model output:
- English translation of the Chinese idiom

Used by:

python scripts/run_translation.py --task task1 --direction zh_en --prompt_version v1


task1_en_zh_v1.txt

This prompt is used for Task 1 English-to-Chinese back-translation.

Input:
- source_text

Expected model output:
- Chinese idiom

Used by:

python scripts/run_translation.py --task task1 --direction en_zh --zh_en_prompt_version v1 --prompt_version v1


task2_zh_en_v3.txt

This prompt is used for Task 2 Chinese-to-English sentence translation.

Input:
- source_text

Expected model output:
- English translation of the Chinese sentence containing an idiom

Used by:

python scripts/run_translation.py --task task2 --direction zh_en --prompt_version v3


task2_en_zh_v1.txt

This prompt is used for Task 2 English-to-Chinese back-translation under condition C1.

Condition:
- C1 / v3_v1

Expected model output:
- A natural Chinese sentence

This condition does not require the model to include a Chinese idiom.

Used by:

python scripts/run_translation.py --task task2 --direction en_zh --zh_en_prompt_version v3 --prompt_version v1


task2_en_zh_v2.txt

This prompt is used for Task 2 English-to-Chinese back-translation under condition C2.

Condition:
- C2 / v3_v2

Expected model output:
- A Chinese sentence containing an idiom

This condition requires the model to reintroduce idiomaticity, but the idiom does not necessarily have to be the original source idiom.

Used by:

python scripts/run_translation.py --task task2 --direction en_zh --zh_en_prompt_version v3 --prompt_version v2


task2_en_zh_v3.txt

This prompt is used for Task 2 English-to-Chinese back-translation under condition C3.

Condition:
- C3 / v3_v3

Expected model output:
- A Chinese sentence containing the original source idiom

This condition is the most constrained Task 2 back-translation setting.

Used by:

python scripts/run_translation.py --task task2 --direction en_zh --zh_en_prompt_version v3 --prompt_version v3


4. Relationship with run_translation.py
---------------------------------------

The script scripts/run_translation.py loads prompt templates from this folder according to the task, direction, and prompt version.

For example:

- task1 + zh_en + v1 loads prompts/task1_zh_en_v1.txt
- task1 + en_zh + v1 loads prompts/task1_en_zh_v1.txt
- task2 + zh_en + v3 loads prompts/task2_zh_en_v3.txt
- task2 + en_zh + v1 loads prompts/task2_en_zh_v1.txt
- task2 + en_zh + v2 loads prompts/task2_en_zh_v2.txt
- task2 + en_zh + v3 loads prompts/task2_en_zh_v3.txt

If a prompt file is missing or renamed, run_translation.py may fail to run.


5. Main experimental settings
-----------------------------

The main thesis experiments use the following prompt settings:

Task 1:

- Chinese-to-English: task1_zh_en_v1.txt
- English-to-Chinese: task1_en_zh_v1.txt

This corresponds to output files such as:

outputs/task1/task1_en_zh_{model}_v1_v1_clean.jsonl


Task 2:

- Chinese-to-English: task2_zh_en_v3.txt
- English-to-Chinese C1: task2_en_zh_v1.txt
- English-to-Chinese C2: task2_en_zh_v2.txt
- English-to-Chinese C3: task2_en_zh_v3.txt

This corresponds to output files such as:

outputs/task2/task2_en_zh_{model}_v3_v1_clean.jsonl
outputs/task2/task2_en_zh_{model}_v3_v2_clean.jsonl
outputs/task2/task2_en_zh_{model}_v3_v3_clean.jsonl


6. Recommended usage
--------------------

To run Task 1 Chinese-to-English translation:

python scripts/run_translation.py --task task1 --direction zh_en --model all --prompt_version v1


To run Task 1 English-to-Chinese back-translation:

python scripts/run_translation.py --task task1 --direction en_zh --model all --zh_en_prompt_version v1 --prompt_version v1


To run Task 2 Chinese-to-English translation:

python scripts/run_translation.py --task task2 --direction zh_en --model all --prompt_version v3


To run Task 2 C1 English-to-Chinese back-translation:

python scripts/run_translation.py --task task2 --direction en_zh --model all --zh_en_prompt_version v3 --prompt_version v1


To run Task 2 C2 English-to-Chinese back-translation:

python scripts/run_translation.py --task task2 --direction en_zh --model all --zh_en_prompt_version v3 --prompt_version v2


To run Task 2 C3 English-to-Chinese back-translation:

python scripts/run_translation.py --task task2 --direction en_zh --model all --zh_en_prompt_version v3 --prompt_version v3



