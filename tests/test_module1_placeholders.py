from pathlib import Path


def test_module1_placeholder_files_exist_and_are_empty():
    root = Path(__file__).resolve().parents[1]
    expected = [
        root / "apex_sam" / "module1_qar" / "build_expert_database.py",
        root / "apex_sam" / "module1_qar" / "retrieve_support_rank2.py",
    ]
    for path in expected:
        assert path.exists(), f"Missing placeholder file: {path}"
        assert path.read_text(encoding="utf-8") == "", f"Placeholder file must be empty: {path}"
