from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


def _extract_include_blocks(markdown_text: str) -> List[str]:
    """
    Return all sample include blocks like:
    {% include sample.html title="..." description="..." audio="/rendered_audio/..." plot="...|..." captions="..." %}
    """
    pattern = re.compile(r"\{\%\s*include\s+sample\.html(?P<attrs>[\s\S]*?)\%\}", re.MULTILINE)
    return [m.group("attrs") for m in pattern.finditer(markdown_text)]


def _parse_attrs(attrs_block: str) -> Dict[str, str]:
    """
    Parse key="value" pairs from the attrs block. Keys are lowercased.
    """
    # Compact whitespace to simplify parsing across lines
    compact = " ".join(attrs_block.split())
    # Match key="value" allowing any non-greedy content inside quotes
    kv_pairs = re.findall(r"(\w+)\s*=\s*\"(.*?)\"", compact)
    return {k.lower(): v for k, v in kv_pairs}


def _site_path_to_workspace_relative(site_path: str) -> str:
    """
    Convert a site-root absolute path like "/rendered_audio/foo.wav" to a
    workspace-relative path string like "rendered_audio\\foo.wav" on Windows
    or "rendered_audio/foo.wav" on POSIX.
    """
    # Remove a single leading slash and normalize separators
    relative = site_path.lstrip("/")
    return str(Path(relative))


def parse_samples_from_markdown_text(markdown_text: str) -> Dict[str, object]:
    """
    Parse the markdown content and extract:
      - target: dict with title, audio
      - samples: list of dicts with title, audio (excluding target)
    """
    blocks = _extract_include_blocks(markdown_text)
    entries: List[Dict[str, str]] = []
    for block in blocks:
        attrs = _parse_attrs(block)
        # Only keep entries that have an audio attribute
        if "audio" not in attrs:
            continue
        title = attrs.get("title", "").strip()
        audio_rel = _site_path_to_workspace_relative(attrs["audio"])  # make workspace-relative
        entries.append({"title": title, "audio": audio_rel})

    # Identify target
    target_entry = next((e for e in entries if e.get("title", "").lower() == "target spectra"), None)
    # Everything else are candidates/samples
    samples = [e for e in entries if e is not target_entry]

    return {
        "target": target_entry,
        "samples": samples,
        "all_entries": entries,
    }


def parse_samples_from_markdown_file(markdown_file: Path | str) -> Dict[str, object]:
    path = Path(markdown_file)
    text = path.read_text(encoding="utf-8")
    return parse_samples_from_markdown_text(text)


def validate_paths(entries: List[Dict[str, str]], base_dir: Path | str = ".") -> List[Dict[str, object]]:
    """
    For each entry with an 'audio' field, add an 'exists' boolean indicating
    whether the file exists relative to base_dir.
    Returns a new list with augmented dicts (does not mutate input).
    """
    base_path = Path(base_dir)
    results: List[Dict[str, object]] = []
    for e in entries:
        audio_path = base_path / e["audio"]
        results.append({**e, "exists": audio_path.exists()})
    return results



