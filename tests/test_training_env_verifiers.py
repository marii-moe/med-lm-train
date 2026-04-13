from medarc_verifiers.prompts import AnswerFormat
from medarc_rl.verifiers import (
    MetaParser,
    StrictMaybeThinkParser,
    build_parser_bundle,
    choose_training_answer_format,
    normalize_training_answer_formats,
)
from medarc_rl.verifiers.rewards import format_reward, multiple_choice_accuracy_reward


def test_strict_maybe_think_parser_rejects_unclosed_think() -> None:
    parser = StrictMaybeThinkParser()

    assert parser.parse("<think>unfinished") == ""
    assert parser.parse("<think>done</think>final") == "final"


def test_parser_bundle_returns_expected_reward_paths() -> None:
    xml_parser, xml_reward = build_parser_bundle(AnswerFormat.XML, use_think=False)
    json_parser, json_reward = build_parser_bundle(AnswerFormat.JSON, use_think=True)
    boxed_parser, boxed_reward = build_parser_bundle(AnswerFormat.BOXED, use_think=False)

    assert xml_parser.parse_answer("<answer>A</answer>") == "A"
    assert xml_reward([{"role": "assistant", "content": "<answer>A</answer>"}]) > 0.0
    assert json_parser.parse_answer('<think>x</think>{"answer":"B"}') == "B"
    assert json_reward([{"role": "assistant", "content": '{"answer":"B"}'}]) > 0.0
    assert boxed_parser.parse_answer(r"\boxed{C}") == "C"
    assert boxed_reward([{"role": "assistant", "content": r"\boxed{C}"}]) == 1.0


def test_seeded_training_format_selection_is_reproducible() -> None:
    formats = normalize_training_answer_formats(AnswerFormat.XML, "Random")

    first = [
        choose_training_answer_format(row_format_key=f"row-{idx}", train_answer_formats=formats, train_format_seed=7)
        for idx in range(6)
    ]
    second = [
        choose_training_answer_format(row_format_key=f"row-{idx}", train_answer_formats=formats, train_format_seed=7)
        for idx in range(6)
    ]

    assert first == second


def test_meta_parser_rejects_missing_and_unknown_formats() -> None:
    parser = MetaParser(use_think=False)

    try:
        parser.parse_for_row([{"role": "assistant", "content": "<answer>A</answer>"}], {})
    except ValueError as exc:
        assert "Missing answer_format" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing answer_format")

    try:
        parser.parse_for_format([{"role": "assistant", "content": "A"}], "yaml")
    except ValueError as exc:
        assert "Unsupported answer format" in str(exc)
    else:
        raise AssertionError("expected ValueError for unsupported answer_format")


def test_rewards_reject_missing_answer_format_metadata() -> None:
    parser = MetaParser(use_think=False)

    try:
        multiple_choice_accuracy_reward(
            completion=[{"role": "assistant", "content": "<answer>A</answer>"}],
            answer="A",
            parser=build_parser_bundle(AnswerFormat.XML, use_think=False)[0],
            info={},
            meta_parser=parser,
        )
    except ValueError as exc:
        assert "Missing answer_format" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing answer_format")

    try:
        format_reward(
            completion=[{"role": "assistant", "content": "<answer>A</answer>"}],
            info=None,
            meta_parser=parser,
        )
    except ValueError as exc:
        assert "Missing info" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing info")
