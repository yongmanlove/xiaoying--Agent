"""Step 2 测试：合成语料的完整性与冲突设计。"""
import re
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"

REQUIRED_FIELDS = [
    "doc_id",
    "title",
    "version",
    "effective_date",
    "department",
    "applies_to",
    "supersedes",
    "topic_key",
    "status",
]


def _parse_frontmatter(text: str):
    if not text.startswith("---"):
        return {}, text
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"')
    return meta, m.group(2)


def load_docs() -> dict:
    docs = {}
    for p in sorted(DOCS_DIR.glob("*.md")):
        meta, body = _parse_frontmatter(p.read_text(encoding="utf-8"))
        docs[meta.get("doc_id")] = {"meta": meta, "body": body, "path": p.name}
    return docs


def test_corpus_has_ten_docs_with_unique_ids():
    docs = load_docs()
    assert len(docs) == 10
    assert set(docs.keys()) == {f"D{i:02d}" for i in range(1, 11)}


def test_every_doc_has_required_frontmatter():
    docs = load_docs()
    for doc_id, d in docs.items():
        for f in REQUIRED_FIELDS:
            assert f in d["meta"], f"{doc_id} 缺少 frontmatter 字段: {f}"


def test_unresolvable_conflict_d01_vs_d02():
    docs = load_docs()
    d01, d02 = docs["D01"], docs["D02"]
    # 同主题、不同事实值
    assert d01["meta"]["topic_key"] == d02["meta"]["topic_key"] == "travel_reimbursement"
    assert "30 天" in d01["body"] and "15 天" in d02["body"]
    # 无适用范围、无取代关系 → 不可解决
    assert d01["meta"]["applies_to"] == "" and d02["meta"]["applies_to"] == ""
    assert d01["meta"]["supersedes"] == "" and d02["meta"]["supersedes"] == ""
    assert d01["meta"]["status"] == d02["meta"]["status"] == "active"


def test_resolvable_conflict_d03_vs_d04():
    docs = load_docs()
    d03, d04 = docs["D03"], docs["D04"]
    assert d03["meta"]["topic_key"] == d04["meta"]["topic_key"] == "annual_leave"
    assert "10 天" in d03["body"] and "15 天" in d04["body"]
    # D04 取代 D03 且生效日期更晚 → 可解决
    assert d04["meta"]["supersedes"] == "D03"
    assert d04["meta"]["effective_date"] > d03["meta"]["effective_date"]
    assert d03["meta"]["status"] == "superseded"
    assert d04["meta"]["status"] == "active"


def test_sick_leave_clear_answer():
    docs = load_docs()
    assert docs["D05"]["meta"]["topic_key"] == "sick_leave"
    assert "24 小时" in docs["D05"]["body"]


def test_same_domain_different_topic_not_conflicting():
    docs = load_docs()
    d01, d10 = docs["D01"], docs["D10"]
    # 同属差旅领域（财务部），但 topic_key 不同 → 不应判为冲突
    assert d01["meta"]["department"] == d10["meta"]["department"]
    assert d01["meta"]["topic_key"] != d10["meta"]["topic_key"]
