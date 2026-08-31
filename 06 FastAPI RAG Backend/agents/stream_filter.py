"""Filter CrewAI streaming output down to the final answer."""

import re

_MARKER = re.compile(r"\*{0,2}final answer\*{0,2}\s*:\*{0,2}", re.IGNORECASE)
_REASONING_LINE = re.compile(
    r"^\s*(Thought|Action|Action Input|Observation|Tool|Tool Input|Tool Output)\s*:",
    re.IGNORECASE,
)
_BARE_JSON_LINE = re.compile(r'^\s*(?:\{|\[\s*(?:\{|"|\]))')
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class FinalAnswerStreamFilter:
    """Incrementally detect the final-answer marker and hide reasoning text."""

    FALLBACK_MESSAGE = (
        "I wasn't able to produce an answer for that query. Please try rephrasing."
    )

    def __init__(self) -> None:
        self._buffer = ""
        self._in_think = False
        self._streaming = False

    def feed(self, text: str) -> str:
        """Feed one chunk and return text that is safe to emit."""
        if self._streaming:
            return self._strip_think_tags(text)

        self._buffer += text
        match = _MARKER.search(self._buffer)
        if not match:
            return ""

        self._streaming = True
        tail = self._buffer[match.end():]
        self._buffer = ""
        return self._strip_think_tags(tail.lstrip())

    def _strip_think_tags(self, text: str) -> str:
        """Remove content between think tags and the tags themselves."""
        result = []
        i = 0
        while i < len(text):
            if not self._in_think:
                open_match = _THINK_OPEN.search(text, i)
                if open_match:
                    result.append(text[i:open_match.start()])
                    self._in_think = True
                    i = open_match.end()
                    continue
                else:
                    result.append(text[i:])
                    break
            else:
                close_match = _THINK_CLOSE.search(text, i)
                if close_match:
                    self._in_think = False
                    i = close_match.end()
                    continue
                else:
                    break
        return "".join(result)

    def clean_final_answer(self, text: str) -> str:
        """Remove any non-answer residue from final output text."""
        text = self._strip_tool_trace(text)
        text = _THINK_BLOCK.sub("", text)
        text = _THINK_OPEN.sub("", _THINK_CLOSE.sub("", text))
        text = _MARKER.sub("", text)
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                lines.append(line)
                continue
            if _REASONING_LINE.match(line):
                continue
            if _BARE_JSON_LINE.match(line):
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    @staticmethod
    def _strip_tool_trace(text: str) -> str:
        """Remove JSON tool-call/observation blocks that some models emit before the answer."""
        parts = re.split(r"\}(?!\s*[\{\"]\s)", text, maxsplit=1)
        if len(parts) > 1 and parts[1].strip():
            prose = parts[1].strip()
            if prose and prose[0].isalpha():
                return prose
        lines = text.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped in ("{", "}", "},", "[", "]", "],", "}{"):
                continue
            if stripped.startswith("{") and stripped.endswith("}"):
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                continue
            if stripped.startswith("}") and stripped.endswith("{"):
                continue
            if re.match(r'^"[a-zA-Z_]+"\s*:', stripped):
                continue
            if stripped.startswith("{") or stripped.startswith("}"):
                after = re.sub(r"^[\s\{\}\[\]]+", "", stripped)
                if after and after[0].isalpha():
                    lines[i] = after
                    return "\n".join(lines[i:]).strip()
                continue
            if re.match(r"^[\{\}\[\]],?\s*$", stripped):
                continue
            return "\n".join(lines[i:]).strip()
        return text.strip()

    def flush(self) -> str:
        """Return cleaned buffered text when no final-answer marker appeared."""
        if self._streaming:
            return ""

        cleaned = self.clean_final_answer(self._buffer)
        self._buffer = ""
        return cleaned if cleaned else self.FALLBACK_MESSAGE

    def reset(self) -> None:
        """Start filtering a fresh CrewAI task."""
        self._buffer = ""
        self._in_think = False
        self._streaming = False
