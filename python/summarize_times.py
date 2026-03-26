import argparse
import re
from pathlib import Path


TIME_PATTERN = re.compile(r"\b(?P<protein>\S+)\s+took\s+(?P<seconds>\d+(?:\.\d+)?)s\s+to\s+find\s+(?P<results>\d+)\s+results\.")


def parse_times(log_path: Path) -> list[tuple[str, float, int]]:
    entries: list[tuple[str, float, int]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = TIME_PATTERN.search(line)
        if not match:
            continue
        entries.append(
            (
                match.group("protein"),
                float(match.group("seconds")),
                int(match.group("results")),
            )
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize timing lines from an ActSeekN log file."
    )
    parser.add_argument("logfile", type=Path, help="Path to the log file to parse.")
    args = parser.parse_args()

    entries = parse_times(args.logfile)
    if not entries:
        raise SystemExit("No matching timing lines were found.")

    fastest = min(entries, key=lambda entry: entry[1])
    slowest = max(entries, key=lambda entry: entry[1])
    average = sum(entry[1] for entry in entries) / len(entries)

    print(f"Count: {len(entries)}")
    print(f"Minimum: {fastest[1]:.2f}s ({fastest[0]})")
    print(f"Maximum: {slowest[1]:.2f}s ({slowest[0]})")
    print(f"Average: {average:.2f}s")


if __name__ == "__main__":
    main()
