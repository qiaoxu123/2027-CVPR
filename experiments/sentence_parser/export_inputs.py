"""Export text-only model inputs and lookup aliases. Never export GT fields."""
import argparse
import hashlib
import json
from pathlib import Path


def export(source, destination):
    texts, aliases, counts, file_hashes = {}, [], {}, {}
    for path in sorted(source.glob("citynav_*.json")):
        content = path.read_bytes()
        file_hashes[path.name] = hashlib.sha256(content).hexdigest()
        episodes = json.loads(content)
        count = 0
        for index, episode in enumerate(episodes):
            for description_index, text in enumerate(episode["descriptions"]):
                text_id = hashlib.sha256(text.encode()).hexdigest()
                texts.setdefault(text_id, {"text_id": text_id, "instruction": text})
                aliases.append({"split_file": path.name, "episode_index": index,
                                "description_index": description_index, "text_id": text_id,
                                "map": f"{episode['area']}_block_{episode['block']}"})
                count += 1
        counts[path.name] = {"episodes": len(episodes), "instructions": count}
    destination.mkdir(parents=True, exist_ok=True)
    for name, rows in [("instructions", texts.values()), ("aliases", aliases)]:
        (destination / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    manifest = {"split_counts": counts, "unique_texts": len(texts), "lookup_records": len(aliases),
                "input_fields": ["text_id", "instruction"], "source_file_sha256": file_hashes,
                "deduplication": "exact UTF-8 text SHA256; no map or target metadata supplied to model"}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(args.source, args.destination)
