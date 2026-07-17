"""End-to-end dry-run CLI test with a mocked Hugging Face dataset."""

from pathlib import Path
from typing import Any

from turkish_tts_generation import dataset as dataset_module
from turkish_tts_generation.cli import main
from turkish_tts_generation.io import read_manifest


class FakeDataset:
    column_names = ["id", "text"]

    def __init__(self) -> None:
        self.rows = [{"id": "one", "text": "Merhaba"}, {"id": "two", "text": "Dünya"}]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.rows[index]


def test_cli_dry_run(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setattr(dataset_module, "load_dataset", lambda *_args, **_kwargs: FakeDataset())
    output_root = tmp_path / "outputs"
    config_path = tmp_path / "generation.yaml"
    config_path.write_text(
        f"""
dataset:
  path: org/dataset
  split: test
  text_column: text
  id_column: id
targets:
  - name: baseline
    engine: noop
    model_id: placeholder
    batch_size: 2
output:
  root: {output_root}
  run_name: cli-test
  audio_format: wav
  resume: true
""",
        encoding="utf-8",
    )

    exit_code = main(["--config", str(config_path), "--dry-run"])

    assert exit_code == 0
    assert "planned=2" in capsys.readouterr().out
    records = read_manifest(output_root / "cli-test" / "baseline" / "plan.jsonl")
    assert [record.sample_id for record in records] == ["one", "two"]
    assert not (output_root / "cli-test" / "baseline" / "audio").exists()
