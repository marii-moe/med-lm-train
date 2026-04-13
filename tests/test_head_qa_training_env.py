import asyncio
import importlib.util
import sys
import time
from pathlib import Path

from datasets import Dataset
from medarc_verifiers.prompts import AnswerFormat


REPO_ROOT = Path(__file__).resolve().parents[1]
HEAD_QA_PATH = REPO_ROOT / "environments" / "head_qa" / "head_qa.py"


def _timing_state() -> dict:
    return {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0, "start_time": time.time()}


def _load_head_qa_module():
    spec = importlib.util.spec_from_file_location("local_head_qa_env", HEAD_QA_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load head_qa module")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("local_head_qa_env", module)
    spec.loader.exec_module(module)
    return module


def _fake_head_qa_dataset(size: int, prefix: str) -> Dataset:
    rows = []
    for idx in range(size):
        rows.append(
            {
                "qtext": f"{prefix} question {idx}?",
                "answers": [
                    {"aid": 1, "atext": f"{prefix} 1 {idx}"},
                    {"aid": 2, "atext": f"{prefix} 2 {idx}"},
                    {"aid": 3, "atext": f"{prefix} 3 {idx}"},
                    {"aid": 4, "atext": f"{prefix} 4 {idx}"},
                ],
                "ra": 1,
            }
        )
    return Dataset.from_list(rows)


def test_head_qa_training_dataset_supports_xml_boxed_and_json(monkeypatch) -> None:
    head_qa = _load_head_qa_module()

    def fake_load_dataset(path: str, lang: str, split: str):
        assert path == "EleutherAI/headqa"
        assert lang == "en"
        return _fake_head_qa_dataset(9 if split == "train" else 1, split)

    monkeypatch.setattr(head_qa, "load_dataset", fake_load_dataset)

    env = head_qa.load_environment(
        use_think=True,
        answer_format=AnswerFormat.XML,
        train_answer_formats="random",
        train_format_seed=11,
    )
    train_dataset = env.get_dataset()
    formats = {row["info"]["answer_format"] for row in train_dataset}

    assert formats == {"xml", "boxed", "json"}


def test_head_qa_training_shuffle_can_differ_from_eval_shuffle(monkeypatch) -> None:
    head_qa = _load_head_qa_module()

    def fake_load_dataset(path: str, lang: str, split: str):
        assert path == "EleutherAI/headqa"
        assert lang == "en"
        return _fake_head_qa_dataset(2 if split == "train" else 1, split)

    monkeypatch.setattr(head_qa, "load_dataset", fake_load_dataset)

    env = head_qa.load_environment(
        shuffle_answers=False,
        training_shuffle_answers=True,
        training_shuffle_seed=23,
    )

    train_row = env.get_dataset()[0]
    eval_row = env.get_eval_dataset()[0]

    assert "options" in train_row["info"]
    assert "options" not in eval_row["info"]


def test_head_qa_group_scoring_and_task_routing(monkeypatch) -> None:
    head_qa = _load_head_qa_module()

    def fake_load_dataset(path: str, lang: str, split: str):
        assert path == "EleutherAI/headqa"
        assert lang == "en"
        return _fake_head_qa_dataset(9 if split == "train" else 1, split)

    monkeypatch.setattr(head_qa, "load_dataset", fake_load_dataset)

    env = head_qa.load_environment(
        use_think=True,
        answer_format=AnswerFormat.XML,
        train_answer_formats=[AnswerFormat.XML, AnswerFormat.BOXED, AnswerFormat.JSON],
        train_format_seed=5,
    )

    completions = {
        "xml": [{"role": "assistant", "content": "<think>x</think><answer>1</answer>"}],
        "boxed": [{"role": "assistant", "content": "<think>x</think>\\boxed{1}"}],
        "json": [{"role": "assistant", "content": '<think>x</think>{"answer":"1"}'}],
    }

    states = []
    seen = set()
    for row in env.get_dataset():
        row_format = row["info"]["answer_format"]
        if row_format in seen:
            continue
        states.append(
            {
                "prompt": row["prompt"],
                "completion": completions[row_format],
                "answer": row["answer"],
                "task": row["task"],
                "info": row["info"],
                "timing": _timing_state(),
                "trajectory": [],
            }
        )
        seen.add(row_format)

    asyncio.run(env.rubric.score_group(states))

    for state in states:
        assert state["reward"] == 1.1
        assert state["metrics"]["multiple_choice_accuracy_reward"] == 1.0
        assert state["metrics"]["format_reward"] == 1.0

    assert env.get_env_for_task("head_qa") is env.train_env
