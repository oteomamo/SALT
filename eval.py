# -*- coding: utf-8 -*-
"""
Evaluate compressed LongBench JSONL files with either backend.

One evaluator, two interchangeable generation backends:

    --backend vllm   (default)
    --backend hf                

Both backends share the SAME prompt templates, metrics, and post-processing,
so vllm and hf scores for one file are directly comparable. This is the
standalone evaluator for already-compressed outputs.

Examples
--------
    # vLLM (default)
    python eval.py --compressed gov_report_compressed.jsonl \
        --model meta-llama/Llama-3.1-8B-Instruct

    # HF backend, portable single-GPU
    python eval.py --backend hf --compressed gov_report_compressed.jsonl \
        --model meta-llama/Llama-3.1-8B-Instruct --gpu 0
"""

import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Metric back-ends are optional and only needed for the datasets that use them.
try:
    from rouge import Rouge
    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False

try:
    from fuzzywuzzy import fuzz
    _FUZZ_AVAILABLE = True
except ImportError:
    _FUZZ_AVAILABLE = False


# =============================================================================
# LongBench task definitions (prompt / metric / generation length)
# =============================================================================
DATASET2PROMPT = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, "
        "and a question. Answer the question as concisely as you can, using a "
        "single phrase if possible. Do not provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story as concisely as you can, "
        "using a single phrase if possible. Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question "
        "as concisely as you can, using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the "
        'article, write "unanswerable". If the question is a yes/no question, '
        'answer "yes", "no", or "unanswerable". Do not provide any explanation.'
        "\n\nArticle: {context}\n\n"
        " Answer the question based on the above article as concisely as you "
        "can, using a single phrase or sentence if possible. If the question "
        "cannot be answered based on the information in the article, write "
        '"unanswerable". If the question is a yes/no question, answer "yes", '
        '"no", or "unanswerable". Do not provide any explanation.'
        "\n\nQuestion: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\n"
        "Now, answer the following question based on the above text, only give "
        "me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page "
        "summary of the report.\n\n"
        "Report:\n{context}\n\n"
        "Now, write a one-page summary of the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question "
        "or instruction. Answer the query in one or more sentences.\n\n"
        "Transcript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one "
        "or more sentences.\n\n"
        "Query: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all "
        "news. \n\nNews:\n{context}\n\n"
        "Now, write a one-page summary of all the news.\n\nSummary:"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the "
        "answer and do not output any other words. The following are some "
        "examples.\n\n{context}\n\n{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following are "
        "some examples.\n\n{context}\n\n{input}"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them "
        "may be duplicates. Please carefully read these paragraphs and determine "
        "how many unique paragraphs there are after removing duplicates. In "
        "other words, how many non-repeating paragraphs are there in total?\n\n"
        "{context}\n\n"
        "Please enter the final count of unique paragraphs after removing "
        "duplicates. The output format should only contain the number, such as "
        "1, 2, 3, and so on.\n\n"
        "The final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please "
        "determine which paragraph the abstract is from.\n\n"
        "{context}\n\n"
        "The following is an abstract.\n\n{input}\n\n"
        "Please enter the number of the paragraph that the abstract is from. "
        'The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\n'
        "The answer is: "
    ),
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
    "trec": (
        "Please determine the type of the question below. Here are some "
        "examples of questions.\n\n{context}\n{input}"
    ),
}

DATASET2METRIC = {
    "narrativeqa": "qa_f1", "qasper": "qa_f1", "multifieldqa_en": "qa_f1",
    "hotpotqa": "qa_f1", "2wikimqa": "qa_f1", "musique": "qa_f1",
    "gov_report": "rouge", "qmsum": "rouge", "multi_news": "rouge",
    "triviaqa": "qa_f1", "samsum": "rouge",
    "passage_count": "count", "passage_retrieval_en": "retrieval",
    "lcc": "code_sim", "repobench-p": "code_sim", "trec": "classification",
}

DATASET2GENLEN = {
    "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64,
    "hotpotqa": 32, "2wikimqa": 32, "musique": 32,
    "gov_report": 512, "qmsum": 512, "multi_news": 512,
    "triviaqa": 32, "samsum": 128, "passage_count": 32,
    "passage_retrieval_en": 32, "lcc": 64, "repobench-p": 64, "trec": 64,
}

# Continuation-style tasks: no chat template, first-line post-processing, stop on newline.
COMPLETION_STYLE = {"triviaqa", "samsum", "lcc", "repobench-p", "trec"}

# Stable display / iteration order (LongBench English subset).
ALL_DATASETS = [
    "gov_report", "multi_news", "qmsum",
    "hotpotqa", "2wikimqa", "musique", "narrativeqa",
    "multifieldqa_en", "qasper", "passage_count",
    "passage_retrieval_en", "lcc", "repobench-p",
    "samsum", "trec", "triviaqa",
]

# EOS strings recognised across the chat models used in the paper's experiments.
EXTRA_EOS_STRINGS = ("<|eot_id|>", "<|end_of_text|>", "</s>")


# =============================================================================
# Metrics (LongBench-faithful)
# =============================================================================
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction, ground_truth, **kwargs):
    return f1_score(normalize_answer(prediction).split(),
                    normalize_answer(ground_truth).split())


def rouge_score(prediction, ground_truth, **kwargs):
    if not _ROUGE_AVAILABLE:
        raise RuntimeError("ROUGE metric needs the 'rouge' package: pip install rouge")
    if not prediction.strip() or not ground_truth.strip():
        return 0.0
    try:
        scores = Rouge().get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return scores["rouge-l"]["f"]


def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs.get("all_classes") or []
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list.copy():
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def retrieval_score(prediction, ground_truth, **kwargs):
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if len(numbers) == 0:
        return 0.0
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth_id))
    return right_num / len(numbers)


def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    if len(numbers) == 0:
        return 0.0
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth))
    return right_num / len(numbers)


def code_sim_score(prediction, ground_truth, **kwargs):
    if not _FUZZ_AVAILABLE:
        raise RuntimeError("code_sim metric needs 'fuzzywuzzy': "
                           "pip install fuzzywuzzy python-Levenshtein")
    all_lines = prediction.lstrip("\n").split("\n")
    prediction = ""
    for line in all_lines:
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            prediction = line
            break
    return fuzz.ratio(prediction, ground_truth) / 100


METRIC_FUNCS = {
    "qa_f1": qa_f1_score, "rouge": rouge_score,
    "classification": classification_score, "retrieval": retrieval_score,
    "count": count_score, "code_sim": code_sim_score,
}


def post_process_prediction(prediction, dataset):
    prediction = (
        prediction.split(".assistant")[0]
        .split("\n\nQuestion")[0]
        .split("</s>")[0]
        .split("(Document")[0]
        .split("\n\nAnswer")[0]
        .split("(Passage")[0]
        .strip()
    )
    if dataset in ["trec", "triviaqa", "samsum"]:
        prediction = prediction.lstrip("\n").split("\n")[0]
    return prediction


def score_sample(prediction, ground_truths, dataset, all_classes=None):
    metric_name = DATASET2METRIC.get(dataset, "qa_f1")
    metric_fn = METRIC_FUNCS[metric_name]
    best = 0.0
    best_gt = ""
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    for gt in ground_truths:
        try:
            s = metric_fn(prediction, gt, all_classes=all_classes)
        except Exception:
            s = 0.0
        if s > best:
            best = s
            best_gt = gt
    return best, best_gt


# =============================================================================
# Prompt construction (shared by both backends)
# =============================================================================
def build_user_prompt(context, inp, dataset):
    template = DATASET2PROMPT.get(
        dataset, "Read the following and respond.\n\n{context}\n\n{input}\n\nAnswer:")
    return template.format(context=context, input=inp)


def apply_chat(tokenizer, dataset, user_prompt):
    """Wrap a user prompt in the model's chat template, except for the
    continuation-style datasets which are scored as raw completions.

    Returns (prompt_text, used_chat_template).
    """
    if dataset in COMPLETION_STYLE:
        return user_prompt, False
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False, add_generation_prompt=True), True
    except Exception:
        return user_prompt, False


def middle_truncate(tokenizer, prompt, max_len):
    """LongBench-style middle truncation to `max_len` tokens (keep the head and
    tail, drop the middle). Applied to the raw prompt *before* the chat template
    so the template's special tokens are never sliced. max_len <= 0 disables it.
    """
    if max_len <= 0:
        return prompt
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if len(ids) <= max_len:
        return prompt
    h = max_len // 2
    return (tokenizer.decode(ids[:h], skip_special_tokens=True)
            + tokenizer.decode(ids[-h:], skip_special_tokens=True))


def detect_dataset_for_file(jsonl_path, override=None):
    if override:
        return override.lower()
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                ds = (rec.get("dataset") or "").lower()
                if ds in DATASET2PROMPT:
                    return ds
                break
    except Exception:
        pass
    stem = Path(jsonl_path).stem.lower()
    if stem in DATASET2PROMPT:
        return stem
    for name in DATASET2PROMPT:
        if stem.startswith(name + "_") or stem.startswith(name + "-"):
            return name
    return None


# =============================================================================
# Generation backends
# =============================================================================
class VLLMBackend:
    """Batched decoding with vLLM."""
    kind = "vllm"

    def __init__(self, args):
        # Expose the right GPU(s) *before* vLLM spins up its workers.
        if args.tensor_parallel_size > 1:
            os.environ.setdefault(
                "CUDA_VISIBLE_DEVICES",
                ",".join(str(i) for i in range(args.tensor_parallel_size)))
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        from vllm import LLM, SamplingParams  # lazy: only vllm users need it
        self._SamplingParams = SamplingParams

        # 0 = no cap: chunked prefill streams long inputs and vLLM uses the
        # model's full context window (faithful to the original evaluator).
        self.max_input_len = args.max_input_len
        max_gen = max(DATASET2GENLEN.values())
        max_model_len = (None if args.max_input_len <= 0
                         else args.max_input_len + max_gen)
        devices = os.environ.get("CUDA_VISIBLE_DEVICES", str(args.gpu))
        print(f"Loading model (vLLM): {args.model} "
              f"(tensor_parallel_size={args.tensor_parallel_size}, GPUs=[{devices}])")
        self.llm = LLM(model=args.model, dtype="auto",
                       tensor_parallel_size=args.tensor_parallel_size,
                       max_model_len=max_model_len,
                       gpu_memory_utilization=args.gpu_mem_util,
                       enable_chunked_prefill=True,
                       trust_remote_code=True, enforce_eager=True)
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, user_prompts, ds_name):
        gen_len = DATASET2GENLEN.get(ds_name, 128)
        prompts = [apply_chat(self.tokenizer, ds_name,
                              middle_truncate(self.tokenizer, p, self.max_input_len))[0]
                   for p in user_prompts]
        stop = list(EXTRA_EOS_STRINGS)
        if ds_name in COMPLETION_STYLE:
            stop.append("\n")
        sp = self._SamplingParams(temperature=0.0, max_tokens=gen_len, stop=stop)
        outputs = self.llm.generate(prompts, sp)
        return [o.outputs[0].text.strip() for o in outputs]


class HFBackend:
    """Single-process decoding with HF transformers (portable; no vLLM)."""
    kind = "hf"

    def __init__(self, args):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self._torch = torch

        self.device = args.device
        self.max_input_len = args.max_input_len if args.max_input_len > 0 else 120000
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}.get(args.dtype, torch.bfloat16)

        print(f"Loading model (HF): {args.model} on {self.device} "
              f"[{args.dtype}, attn={args.attn_impl}]")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, device_map=self.device,
            attn_implementation=args.attn_impl)
        self.model.eval()

    def _stop_conditions(self, ds_name):
        tok = self.tokenizer
        eos_ids = []
        if tok.eos_token_id is not None:
            eos_ids.append(int(tok.eos_token_id))
        for s in EXTRA_EOS_STRINGS:
            try:
                tid = tok.convert_tokens_to_ids(s)
                if (isinstance(tid, int) and tid >= 0
                        and tid != tok.unk_token_id and tid not in eos_ids):
                    eos_ids.append(tid)
            except Exception:
                pass
        stop_strings = ["\n"] if ds_name in COMPLETION_STYLE else None
        return eos_ids, stop_strings

    def generate(self, user_prompts, ds_name):
        torch = self._torch
        gen_len = DATASET2GENLEN.get(ds_name, 128)
        eos_ids, stop_strings = self._stop_conditions(ds_name)
        preds = []
        for p in user_prompts:
            p = middle_truncate(self.tokenizer, p, self.max_input_len)
            prompt, used_chat = apply_chat(self.tokenizer, ds_name, p)
            inp = self.tokenizer(prompt, return_tensors="pt",
                                add_special_tokens=(not used_chat)).to(self.device)
            ctx_len = inp["input_ids"].shape[-1]
            gen_kwargs = dict(max_new_tokens=gen_len, num_beams=1,
                              do_sample=False, temperature=1.0,
                              min_length=ctx_len + 1, eos_token_id=eos_ids,
                              pad_token_id=self.tokenizer.pad_token_id)
            if stop_strings:
                gen_kwargs["stop_strings"] = stop_strings
                gen_kwargs["tokenizer"] = self.tokenizer
            with torch.no_grad():
                out = self.model.generate(**inp, **gen_kwargs)
            new_tokens = out[0][ctx_len:]
            preds.append(self.tokenizer.decode(
                new_tokens, skip_special_tokens=True).strip())
        return preds


def make_backend(args):
    return VLLMBackend(args) if args.backend == "vllm" else HFBackend(args)


# =============================================================================
# Evaluation
# =============================================================================
def eval_jsonl(jsonl_path, ds_name, backend, verbose=False, ctx_preview_chars=300):
    with open(jsonl_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    metric_name = DATASET2METRIC.get(ds_name, "qa_f1")
    print(f"    {ds_name:<28} ({len(samples)} samples, {metric_name})...",
          end="" if not verbose else "\n", flush=True)

    user_prompts = [build_user_prompt(s.get("context", ""), s.get("input", ""), ds_name)
                    for s in samples]

    t0 = time.time()
    raw_preds = backend.generate(user_prompts, ds_name)
    elapsed = time.time() - t0

    scores, per_sample = [], []
    for i, (sample, raw) in enumerate(zip(samples, raw_preds)):
        prediction = post_process_prediction(raw, ds_name)
        answers = sample.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]
        score, best_gt = score_sample(prediction, answers, ds_name,
                                      sample.get("all_classes"))
        scores.append(score)
        stats = sample.get("compression_stats", {})
        per_sample.append({
            "sample_id": sample.get("_id", ""),
            "score": round(score * 100, 2),
            "prediction": prediction[:200],
            "compression_ratio": stats.get("compression_ratio"),
        })

        if verbose:
            sid = sample.get("_id", f"sample_{i}")
            q = sample.get("input", "")
            ctx = sample.get("context", "")
            ctx_words = len(ctx.split())
            zero_flag = "  [ZERO]" if score == 0.0 else ""
            print(f"\n  --- [{i+1}/{len(samples)}] {sid[:24]}  "
                  f"score={score*100:6.2f}{zero_flag}")
            print(f"    Q:    {q[:200]}")
            print(f"    GOLD: {answers}")
            print(f"    BEST_MATCH_GT: {best_gt!r}")
            print(f"    PRED: {prediction[:300]!r}")
            print(f"    CTX:  ({ctx_words} words) {ctx[:ctx_preview_chars]!r}...")

    avg_score = round(100 * sum(scores) / len(scores), 2) if scores else 0.0
    n_zero = sum(1 for s in scores if s == 0.0)

    if verbose:
        print(f"\n    {ds_name}: {avg_score:.2f}  ({n_zero} zeros, {elapsed:.0f}s)")
    else:
        print(f" {avg_score:6.2f}  ({n_zero} zeros, {elapsed:.0f}s)")

    return {ds_name: {
        "metric": metric_name, "score": avg_score, "n_samples": len(samples),
        "n_zero": n_zero, "time_seconds": round(elapsed, 1),
        "per_sample": per_sample,
    }}


def eval_folder(data_dir, backend, target_datasets=None, verbose=False):
    data_dir = Path(data_dir)
    if target_datasets is None:
        target_datasets = [f.stem for f in sorted(data_dir.glob("*.jsonl"))
                           if f.stem in DATASET2PROMPT]
    results = {}
    for ds_name in target_datasets:
        jsonl_path = data_dir / f"{ds_name}.jsonl"
        if not jsonl_path.exists():
            continue
        results.update(eval_jsonl(jsonl_path, ds_name, backend, verbose=verbose))
    return results


def print_comparison(all_results, target_names, file=None):
    def out(s=""):
        print(s, file=file)
    ds_w = 28
    col_w = max(max(len(Path(t).name) for t in target_names), 8) + 1
    out(f"\n{'='*100}\n  COMPARISON TABLE\n{'='*100}")
    header = f"  {'Dataset':<{ds_w}}"
    for t in target_names:
        header += f" {Path(t).name:>{col_w}}"
    header += f" {'BEST':>{col_w}}"
    out(header)
    out(f"  {'-'*ds_w}" + f" {'-'*col_w}" * (len(target_names) + 1))
    col_totals = {t: [] for t in target_names}
    for ds in ALL_DATASETS:
        row = f"  {ds:<{ds_w}}"
        best_score = -1
        best_target = ""
        for t in target_names:
            res = all_results.get(t, {})
            score = res.get(ds, {}).get("score")
            if score is not None:
                row += f" {score:>{col_w}.2f}"
                col_totals[t].append(score)
                if score > best_score:
                    best_score = score
                    best_target = Path(t).name
            else:
                row += f" {'-':>{col_w}}"
        row += f" {best_target:>{col_w}}" if best_score >= 0 else ""
        out(row)
    out(f"  {'-'*ds_w}" + f" {'-'*col_w}" * (len(target_names) + 1))
    avg_row = f"  {'AVERAGE':<{ds_w}}"
    best_avg = -1
    best_avg_target = ""
    for t in target_names:
        vals = col_totals[t]
        if vals:
            avg = sum(vals) / len(vals)
            avg_row += f" {avg:>{col_w}.2f}"
            if avg > best_avg:
                best_avg = avg
                best_avg_target = Path(t).name
        else:
            avg_row += f" {'-':>{col_w}}"
    avg_row += f" {best_avg_target:>{col_w}}"
    out(avg_row)
    out()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate compressed LongBench JSONL files (vLLM or HF backend).")
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm",
                        help="Generation backend: 'vllm' (batched, fast) or "
                             "'hf' (portable transformers). Default: vllm.")
    parser.add_argument("--data-dir", type=str, nargs="+", default=None,
                        help="One or more folders of <dataset>.jsonl files.")
    parser.add_argument("--compressed", type=str, nargs="+", default=None,
                        help="One or more compressed (or original) JSONL files.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Force the dataset name for --compressed files "
                             "(otherwise auto-detected from the JSONL).")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated subset of datasets to run for --data-dir.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--max-input-len", type=int, default=0,
                        help="Cap on input tokens. 0 = full context window (vLLM) / "
                             "120000 (HF middle-truncation).")
    parser.add_argument("--comparison", type=str, default=None,
                        help="Write the multi-target comparison table to this file.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip targets whose *_eval.json already exists.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print question/gold/prediction/score per sample.")
    parser.add_argument("--ctx-preview-chars", type=int, default=300,
                        help="Chars of compressed context to show in verbose mode.")

    vllm_grp = parser.add_argument_group("vLLM backend")
    vllm_grp.add_argument("--tensor-parallel-size", type=int, default=1,
                          help="GPUs to shard the model across (vLLM).")
    vllm_grp.add_argument("--gpu-mem-util", type=float, default=0.9)

    hf_grp = parser.add_argument_group("HF backend")
    hf_grp.add_argument("--device", type=str, default="cuda",
                        help="Device for the HF backend (cuda/cpu/cuda:0).")
    hf_grp.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    hf_grp.add_argument("--attn-impl", type=str, default="sdpa",
                        choices=["sdpa", "eager", "flash_attention_2"])

    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index (single-GPU vLLM, or HF CUDA_VISIBLE_DEVICES).")
    args = parser.parse_args()

    if not args.data_dir and not args.compressed:
        parser.error("must provide --data-dir or --compressed")

    targets = []
    if args.data_dir:
        for d in args.data_dir:
            targets.append({"key": str(d), "kind": "folder",
                            "path": Path(d), "ds_name": None})
    if args.compressed:
        for f in args.compressed:
            ds = detect_dataset_for_file(f, args.dataset)
            if ds is None:
                print(f"  WARNING: cannot determine dataset for {f}; skipping.")
                continue
            targets.append({"key": str(f), "kind": "file",
                            "path": Path(f), "ds_name": ds})

    target_datasets = ([d.strip() for d in args.datasets.split(",")]
                       if args.datasets else None)

    def _eval_path(t):
        return (t["path"] / "eval_all.json" if t["kind"] == "folder"
                else t["path"].with_name(t["path"].stem + "_eval.json"))

    targets_to_eval = []
    for t in targets:
        if args.skip_existing and _eval_path(t).exists():
            print(f"  SKIP (exists): {t['key']}")
        else:
            targets_to_eval.append(t)

    if not targets_to_eval and not args.skip_existing:
        print("No valid targets to evaluate.")
        sys.exit(1)

    backend = make_backend(args) if targets_to_eval else None

    all_results = {}
    for t in targets_to_eval:
        print(f"\n{'='*70}\n  EVALUATING [{args.backend}]: {t['key']}  [{t['kind']}"
              + (f", dataset={t['ds_name']}" if t['kind'] == 'file' else "") + "]"
              f"\n{'='*70}")
        if t["kind"] == "folder":
            results = eval_folder(t["path"], backend, target_datasets,
                                  verbose=args.verbose)
        else:
            results = eval_jsonl(t["path"], t["ds_name"], backend,
                                 verbose=args.verbose,
                                 ctx_preview_chars=args.ctx_preview_chars)
        all_results[t["key"]] = results
        eval_out = _eval_path(t)
        with open(eval_out, "w") as f:
            json.dump({"model": args.model, "backend": args.backend,
                       "target": t["key"], "kind": t["kind"], "results": results},
                      f, indent=2, ensure_ascii=False)
        print(f"  Saved: {eval_out}")

    # Pull in any previously-computed results for skipped targets (for the table).
    for t in targets:
        if t["key"] in all_results:
            continue
        eval_out = _eval_path(t)
        if eval_out.exists():
            with open(eval_out) as f:
                data = json.load(f)
            all_results[t["key"]] = data.get("results", {})

    for key, results in all_results.items():
        print(f"\n{'='*70}\n  {Path(key).name}\n{'='*70}")
        print(f"  {'Dataset':<28} {'Metric':<15} {'Score':>8} {'Zeros':>6}")
        print(f"  {'-'*28} {'-'*15} {'-'*8} {'-'*6}")
        for ds in ALL_DATASETS:
            if ds in results:
                r = results[ds]
                print(f"  {ds:<28} {r['metric']:<15} {r['score']:>8.2f} "
                      f"{r.get('n_zero', 0):>6}")

    if len(all_results) > 1:
        target_names = list(all_results.keys())
        print_comparison(all_results, target_names)
        if args.comparison:
            with open(args.comparison, "w") as f:
                print_comparison(all_results, target_names, file=f)
            print(f"\n  Comparison saved: {args.comparison}")


if __name__ == "__main__":
    main()
