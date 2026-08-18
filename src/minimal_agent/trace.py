import json
from dataclasses import asdict
from pathlib import Path

from minimal_agent.agent import TraceEvent


class JsonlTraceStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def __call__(self, event: TraceEvent) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(asdict(event), ensure_ascii=False, separators=(",", ":")))
            trace_file.write("\n")

    def read_last_run(self) -> list[TraceEvent]:
        if not self._path.exists():
            return []

        events = [
            TraceEvent(**json.loads(line))
            for line in self._path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if not events:
            return []
        last_run_id = events[-1].run_id
        last_run: list[TraceEvent] = []
        for event in reversed(events):
            if event.run_id != last_run_id:
                break
            last_run.append(event)
        return list(reversed(last_run))
