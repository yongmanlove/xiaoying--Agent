"""Step 8 测试：CLI 在无 API Key 时以 mock 模式返回合法 JSON。"""
import json

from agentqa.cli import main


def test_question_mode_returns_valid_json(capsys):
    rc = main(["--question", "病假需要提前多久申请？", "--provider", "mock"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] in {
        "answered",
        "insufficient_information",
        "conflict_needs_clarification",
        "tool_error",
        "max_steps_exceeded",
    }
    assert "answer" in out and "citations" in out and "gaps" in out


def test_question_conflict_status(capsys):
    rc = main(["--question", "报销期限多久？", "--provider", "mock"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "conflict_needs_clarification"


def test_case_mode_writes_result_trace_and_markdown(tmp_path):
    case_dir = tmp_path / "case01"
    case_dir.mkdir()
    (case_dir / "input.json").write_text(
        json.dumps({"question": "病假需要提前多久申请？"}, ensure_ascii=False),
        encoding="utf-8",
    )
    rc = main(["--case", str(case_dir / "input.json"), "--provider", "mock"])
    assert rc == 0
    assert (case_dir / "result.json").exists()
    assert (case_dir / "trace.json").exists()
    assert (case_dir / "output.md").exists()
    result = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "answered"


def test_batch_mode_writes_summary(tmp_path):
    qpath = tmp_path / "questions.json"
    qpath.write_text(
        json.dumps(
            [
                {"question": "病假需要提前多久申请？"},
                {"question": "报销期限多久？"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    rc = main(["--batch", str(qpath), "--provider", "mock", "--out", str(out)])
    assert rc == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert len(summary) == 2
    assert summary[0]["status"] == "answered"
    assert summary[1]["status"] == "conflict_needs_clarification"


def test_fail_search_flag_recovers(capsys):
    rc = main(["--question", "病假需要提前多久申请？", "--provider", "mock", "--fail-search", "1"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "answered"
