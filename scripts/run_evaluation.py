import os
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple

import jieba
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
from bert_score import score as bertscore_score
from comet import download_model, load_from_checkpoint
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 1. Global paths
# =========================

METRICX_REPO_PATH = "/dss/dsshome1/07/ru27nut2/metricx"


# =========================
# 2. Helpers
# =========================

def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load data from a JSON array file or a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()

    if not txt:
        return []

    if txt.startswith("["):
        data = json.loads(txt)
        if not isinstance(data, list):
            raise ValueError(f"{path} looks like a JSON array but is not a list.")
        return data

    rows = []
    for line in txt.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def average(values: List[float]) -> float:
    """Return the arithmetic mean."""
    if not values:
        return None
    return sum(values) / len(values)


def tokenize_zh_for_rouge(text: str) -> str:
    """
    Segment Chinese text for ROUGE.
    rouge_score expects whitespace-separated tokens.
    """
    tokens = jieba.lcut(text.strip())
    tokens = [tok for tok in tokens if tok.strip()]
    return " ".join(tokens)


def is_false_flag(row: Dict[str, Any], field_name: str) -> bool:
    """Return True only when row[field_name] is the string 'False'."""
    return str(row.get(field_name, "")).strip() == "False"


def get_skip_reason(
    row: Dict[str, Any],
    reference_field: str,
    prediction_field: str,
    eval_type: str
) -> str | None:
    """
    Return skip reason if the row should NOT be evaluated.
    Return None if the row should be evaluated.
    """
    if "index" not in row:
        return "missing_index"

    if reference_field not in row:
        return "missing_reference_field"

    if prediction_field not in row:
        return "missing_prediction_field"

    invalid_en_false = is_false_flag(row, "is_invalid_en")
    invalid_zh_false = is_false_flag(row, "is_invalid_zh")

    if eval_type == "zh_en":
        if not invalid_en_false:
            return "invalid_en"

    elif eval_type == "en_zh":
        if not invalid_zh_false:
            return "invalid_zh"

    elif eval_type == "zh_zh":
        if (not invalid_en_false) and (not invalid_zh_false):
            return "invalid_en_and_invalid_zh"
        elif not invalid_en_false:
            return "invalid_en"
        elif not invalid_zh_false:
            return "invalid_zh"

    return None


def build_record_pairs(
    rows: List[Dict[str, Any]],
    reference_field: str,
    prediction_field: str,
    eval_type: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int, int]:
    """
    Build aligned record pairs and detail rows.

    Returns:
      - valid_records: rows used for evaluation
      - detail_rows: all rows with included_in_eval + skip_reason
      - num_total
      - num_valid
      - num_invalid
    """
    valid_records = []
    detail_rows = []

    num_total = 0
    num_valid = 0
    num_invalid = 0

    for row in rows:
        num_total += 1

        skip_reason = get_skip_reason(
            row=row,
            reference_field=reference_field,
            prediction_field=prediction_field,
            eval_type=eval_type
        )

        included_in_eval = skip_reason is None

        detail_item = {
            "index": row.get("index"),
            "reference": row.get(reference_field),
            "prediction": row.get(prediction_field),
            "included_in_eval": included_in_eval,
            "skip_reason": skip_reason,
            "bleu": None,
            "rouge1": None,
            "rouge2": None,
            "rougeL": None,
            "bertscore_f1": None,
            "sbert_cosine": None,
            "cometkiwi": None,
            "metricx_qe": None,
            "comet": None,
        }

        if included_in_eval:
            num_valid += 1
            valid_records.append({
                "index": row["index"],
                "reference": row[reference_field],
                "prediction": row[prediction_field],
                "row": row,
            })
        else:
            num_invalid += 1

        detail_rows.append(detail_item)

    return valid_records, detail_rows, num_total, num_valid, num_invalid


def compute_metricx_qe(
    records: List[Dict[str, Any]],
    tmp_dir: str,
    file_tag: str
) -> List[float]:
    """
    Compute MetricX-QE scores.
    Input format:
      - source: reference
      - hypothesis: prediction
    """
    ensure_dir(tmp_dir)

    metricx_input_file = os.path.join(tmp_dir, f"metricx_input_{file_tag}.jsonl")
    metricx_output_file = os.path.join(tmp_dir, f"metricx_output_{file_tag}.jsonl")

    with open(metricx_input_file, "w", encoding="utf-8") as f:
        for r in records:
            obj = {
                "source": r["reference"],
                "hypothesis": r["prediction"],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = METRICX_REPO_PATH + ":" + env.get("PYTHONPATH", "")

    command = [
        "python",
        "-m",
        "metricx23.predict",
        "--tokenizer", "google/mt5-xl",
        "--model_name_or_path", "google/metricx-23-qe-xl-v2p0",
        "--max_input_length", "1024",
        "--batch_size", "1",
        "--input_file", metricx_input_file,
        "--output_file", metricx_output_file,
        "--qe",
    ]

    subprocess.run(command, check=True, env=env)

    scores = []
    with open(metricx_output_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            scores.append(item["prediction"])

    if len(scores) != len(records):
        raise ValueError(
            f"MetricX-QE score count mismatch for {file_tag}: "
            f"{len(scores)} vs {len(records)}"
        )

    return scores


def compute_sbert_scores(
    records: List[Dict[str, Any]],
    sbert_model: SentenceTransformer
) -> List[float]:
    """Compute multilingual SBERT cosine similarity."""
    references = [r["reference"] for r in records]
    predictions = [r["prediction"] for r in records]

    ref_emb = sbert_model.encode(
        references,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    pred_emb = sbert_model.encode(
        predictions,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    sims = cosine_similarity(ref_emb, pred_emb)
    scores = [float(sims[i, i]) for i in range(len(records))]
    return scores


# =========================
# 3. Metric initialization
# =========================

BLEU_METRIC = BLEU(tokenize="flores200", effective_order=True)

ROUGE_METRIC = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=False
)

COMET_KIWI_MODEL_PATH = download_model("Unbabel/wmt22-cometkiwi-da")
COMET_KIWI_MODEL = load_from_checkpoint(COMET_KIWI_MODEL_PATH)

# Reference-based COMET, only for zh_zh
COMET_REF_MODEL_PATH = download_model("Unbabel/wmt22-comet-da")
COMET_REF_MODEL = load_from_checkpoint(COMET_REF_MODEL_PATH)

# Multilingual SBERT
SBERT_MODEL = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")


# =========================
# 4. Config by task and eval type
# =========================

def get_input_file(
    task: str,
    model_name: str,
    zh_en_prompt_version: str,
    en_zh_prompt_version: str
) -> str:
    """Return the input file path for evaluation."""
    if task == "task1":
        return (
            f"outputs/task1/"
            f"task1_en_zh_{model_name}_{zh_en_prompt_version}_{en_zh_prompt_version}_clean.jsonl"
        )
    else:
        return (
            f"outputs/{task}/"
            f"{task}_en_zh_{model_name}_{zh_en_prompt_version}_{en_zh_prompt_version}_clean.jsonl"
        )


def get_fields(task: str, eval_type: str) -> Tuple[str, str]:
    """
    Return (reference_field, prediction_field) for each task/eval_type.

    task1:
      zh_en -> source_idiom vs translation_en
      en_zh -> translation_en vs translation_zh
      zh_zh -> source_idiom vs translation_zh

    task2/task3_3:
      zh_en -> sentence vs translation_en
      en_zh -> translation_en vs translation_zh
      zh_zh -> sentence vs translation_zh
    """
    if task == "task1":
        if eval_type == "zh_en":
            return "source_idiom", "translation_en"
        elif eval_type == "en_zh":
            return "translation_en", "translation_zh"
        elif eval_type == "zh_zh":
            return "source_idiom", "translation_zh"
    else:
        if eval_type == "zh_en":
            return "sentence", "translation_en"
        elif eval_type == "en_zh":
            return "translation_en", "translation_zh"
        elif eval_type == "zh_zh":
            return "sentence", "translation_zh"

    raise ValueError(f"Unsupported combination: task={task}, eval_type={eval_type}")


def build_comet_ref_input(task: str, records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Build input for reference-based COMET, only for zh_zh.

    task1:
      src = source_idiom
      mt  = translation_zh
      ref = source_idiom

    task2/task3_3:
      src = sentence
      mt  = translation_zh
      ref = sentence
    """
    comet_input = []
    for r in records:
        row = r["row"]

        if task == "task1":
            src = row["source_idiom"]
            ref = row["source_idiom"]
        else:
            src = row["sentence"]
            ref = row["sentence"]

        mt = row["translation_zh"]

        comet_input.append({
            "src": src,
            "mt": mt,
            "ref": ref,
        })

    return comet_input


# =========================
# 5. Evaluation core
# =========================

def evaluate_records(
    task: str,
    records: List[Dict[str, Any]],
    eval_type: str,
    tmp_dir: str,
    file_tag: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Evaluate valid records and return:
      - scored_results
      - avg_result
    """
    if not records:
        return [], {
            "bleu": None,
            "rouge1": None,
            "rouge2": None,
            "rougeL": None,
            "bertscore_f1": None,
            "sbert_cosine": None,
            "cometkiwi": None,
            "metricx_qe": None,
            "comet": None,
        }

    # 1) Sentence-level BLEU
    bleu_scores = []
    for r in records:
        s = BLEU_METRIC.sentence_score(r["prediction"], [r["reference"]]).score
        bleu_scores.append(s)

    # 2) ROUGE
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for r in records:
        if eval_type == "zh_zh":
            reference_text = tokenize_zh_for_rouge(r["reference"])
            prediction_text = tokenize_zh_for_rouge(r["prediction"])
        else:
            reference_text = r["reference"]
            prediction_text = r["prediction"]

        rs = ROUGE_METRIC.score(reference_text, prediction_text)
        rouge1_scores.append(rs["rouge1"].fmeasure)
        rouge2_scores.append(rs["rouge2"].fmeasure)
        rougeL_scores.append(rs["rougeL"].fmeasure)

    # 3) BERTScore
    predictions = [r["prediction"] for r in records]
    references = [r["reference"] for r in records]

    _, _, bert_f1 = bertscore_score(
        predictions,
        references,
        model_type="xlm-roberta-large",
        rescale_with_baseline=False
    )
    bert_f1_scores = [x.item() for x in bert_f1]

    # 4) Multilingual SBERT cosine similarity
    sbert_scores = compute_sbert_scores(records, SBERT_MODEL)

    # 5) COMETKiwi
    cometkiwi_input = [{"src": r["reference"], "mt": r["prediction"]} for r in records]
    cometkiwi_output = COMET_KIWI_MODEL.predict(
        cometkiwi_input,
        batch_size=16,
        gpus=1
    )
    cometkiwi_scores = cometkiwi_output.scores

    # 6) MetricX-QE
    metricx_scores = compute_metricx_qe(records, tmp_dir=tmp_dir, file_tag=file_tag)

    # 7) Reference-based COMET only for zh_zh
    comet_scores = [None] * len(records)
    if eval_type == "zh_zh":
        comet_input = build_comet_ref_input(task, records)
        comet_output = COMET_REF_MODEL.predict(
            comet_input,
            batch_size=16,
            gpus=1
        )
        comet_scores = comet_output.scores

    # 8) Per-sentence scored results
    scored_results = []
    for i, r in enumerate(records):
        scored_results.append({
            "index": r["index"],
            "reference": r["reference"],
            "prediction": r["prediction"],
            "bleu": bleu_scores[i],
            "rouge1": rouge1_scores[i],
            "rouge2": rouge2_scores[i],
            "rougeL": rougeL_scores[i],
            "bertscore_f1": bert_f1_scores[i],
            "sbert_cosine": sbert_scores[i],
            "cometkiwi": cometkiwi_scores[i],
            "metricx_qe": metricx_scores[i],
            "comet": comet_scores[i],
        })

    # 9) Average result
    avg_result = {
        "bleu": average(bleu_scores),
        "rouge1": average(rouge1_scores),
        "rouge2": average(rouge2_scores),
        "rougeL": average(rougeL_scores),
        "bertscore_f1": average(bert_f1_scores),
        "sbert_cosine": average(sbert_scores),
        "cometkiwi": average(cometkiwi_scores),
        "metricx_qe": average(metricx_scores),
        "comet": average([x for x in comet_scores if x is not None]) if eval_type == "zh_zh" else None,
    }

    return scored_results, avg_result


def merge_scores_into_detail(
    detail_rows: List[Dict[str, Any]],
    scored_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Fill metric scores back into detail rows for included samples."""
    score_map = {item["index"]: item for item in scored_results}

    merged = []
    for item in detail_rows:
        new_item = dict(item)

        if new_item["included_in_eval"] and new_item["index"] in score_map:
            scored = score_map[new_item["index"]]
            new_item["bleu"] = scored["bleu"]
            new_item["rouge1"] = scored["rouge1"]
            new_item["rouge2"] = scored["rouge2"]
            new_item["rougeL"] = scored["rougeL"]
            new_item["bertscore_f1"] = scored["bertscore_f1"]
            new_item["sbert_cosine"] = scored["sbert_cosine"]
            new_item["cometkiwi"] = scored["cometkiwi"]
            new_item["metricx_qe"] = scored["metricx_qe"]
            new_item["comet"] = scored["comet"]

        merged.append(new_item)

    return merged


# =========================
# 6. Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["task1", "task2", "task3_3"])
    parser.add_argument(
        "--eval_type",
        required=True,
        choices=["zh_en", "en_zh", "zh_zh"],
        help="Evaluation type."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["llama31", "mistral", "deepseek", "gemma2", "glm4", "qwen25", "all"],
        help="Choose one model or 'all'."
    )
    parser.add_argument(
        "--zh_en_prompt_version",
        required=True,
        help="Version used in zh_en output."
    )
    parser.add_argument(
        "--en_zh_prompt_version",
        required=True,
        help="Version used in en_zh output."
    )
    args = parser.parse_args()

    model_names = (
        ["llama31", "mistral", "deepseek", "gemma2", "glm4", "qwen25"]
        if args.model == "all"
        else [args.model]
    )

    eval_dir = os.path.join("eval", args.task)
    ensure_dir(eval_dir)

    metricx_tmp_dir = os.path.join(eval_dir, "metricx_tmp")
    ensure_dir(metricx_tmp_dir)

    reference_field, prediction_field = get_fields(args.task, args.eval_type)

    avg_all_results = []

    for model_name in model_names:
        input_file = get_input_file(
            task=args.task,
            model_name=model_name,
            zh_en_prompt_version=args.zh_en_prompt_version,
            en_zh_prompt_version=args.en_zh_prompt_version
        )

        if not os.path.exists(input_file):
            print(f"Skipped {model_name}: file not found -> {input_file}")
            continue

        print(f"\n===== Evaluating {model_name} =====")
        print("Input file:", input_file)

        rows = load_json_or_jsonl(input_file)

        valid_records, detail_rows, num_total, num_valid, num_invalid = build_record_pairs(
            rows=rows,
            reference_field=reference_field,
            prediction_field=prediction_field,
            eval_type=args.eval_type
        )

        print("Number of total rows:", num_total)
        print("Number of valid rows:", num_valid)
        print("Number of invalid rows:", num_invalid)

        file_tag = (
            f"{args.task}_{args.eval_type}_{model_name}_"
            f"{args.zh_en_prompt_version}_{args.en_zh_prompt_version}"
        )

        scored_results, avg_result = evaluate_records(
            task=args.task,
            records=valid_records,
            eval_type=args.eval_type,
            tmp_dir=metricx_tmp_dir,
            file_tag=file_tag
        )

        detail_results = merge_scores_into_detail(
            detail_rows=detail_rows,
            scored_results=scored_results
        )

        avg_result_full = {
            "task": args.task,
            "eval_type": args.eval_type,
            "model": model_name,
            "zh_en_prompt_version": args.zh_en_prompt_version,
            "en_zh_prompt_version": args.en_zh_prompt_version,
            "num_total": num_total,
            "num_valid": num_valid,
            "num_invalid": num_invalid,
            **avg_result
        }
        avg_all_results.append(avg_result_full)

        detail_path = os.path.join(
            eval_dir,
            f"{args.task}_detail_{args.eval_type}_{model_name}_{args.zh_en_prompt_version}_{args.en_zh_prompt_version}.json"
        )

        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(detail_results, f, ensure_ascii=False, indent=2)

        print("Saved:", detail_path)

    summary_path = os.path.join(
        eval_dir,
        f"{args.task}_summary_{args.eval_type}_{args.zh_en_prompt_version}_{args.en_zh_prompt_version}.json"
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(avg_all_results, f, ensure_ascii=False, indent=2)

    print("Saved:", summary_path)
    print("\nAll evaluations finished.")


if __name__ == "__main__":
    main()