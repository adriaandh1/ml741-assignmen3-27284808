"""Targeted static checks for the supplied academic writing rules."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def strip_latex(source: str) -> str:
    text = re.sub(r"(?m)(?<!\\)%.*$", "", source)
    text = text.split(r"\begin{document}", 1)[-1]
    for environment in (
        "algorithm",
        "algorithmic",
        "equation",
        "figure",
        "figure*",
        "table",
        "table*",
        "tabular",
        "tabularx",
        "thebibliography",
    ):
        escaped = re.escape(environment)
        text = re.sub(
            rf"\\begin\{{{escaped}\}}.*?\\end\{{{escaped}\}}",
            " ",
            text,
            flags=re.DOTALL,
        )
    text = re.sub(r"\\(?:cite|ref|eqref|label|path|url)\{[^}]*\}", " ", text)
    text = re.sub(
        r"\\(?:texttt|textit|emph|section|subsection|caption)\{([^}]*)\}",
        r"\1",
        text,
    )
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", " ", text)
    text = re.sub(r"[{}$&~_^]", " ", text)
    return re.sub(r"\s+", " ", text)


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "main.tex")
    source = path.read_text(encoding="utf-8")
    text = strip_latex(source)
    failures: list[str] = []

    forbidden = {
        "first person plural": r"\b(?:we|our|ours|ourselves)\b",
        "first person singular": r"\b(?:I|me|my|mine|myself)\b",
        "uncertain quantifier": r"\b(?:some|certain)\b",
        "uncertain modal": r"\b(?:may be|can be)\b",
        "ambiguous pronoun": r"\b(?:it|its|this|them|they|which)\b",
        "etcetera": r"\betc\.?\b",
        "contraction": r"\b\w+(?:n't|'re|'ve|'ll|'d|'m)\b",
        "present continuous": r"\b(?:is|are|was|were)\s+\w+ing\b",
    }
    for label, pattern in forbidden.items():
        matches = sorted(set(re.findall(pattern, text, flags=re.IGNORECASE)))
        if matches:
            failures.append(f"{label}: {matches}")

    for match in re.finditer(r"\\(?:section|subsection)\{([^}]*)\}", source):
        heading = match.group(1)
        if heading.endswith("?"):
            failures.append(f"question heading: {heading}")

    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sentence in sentences:
        words = re.findall(r"\b[\w-]+\b", sentence)
        punctuation_probe = re.sub(r"(?<=\d),(?=\d)", "", sentence)
        if len(words) > 38:
            failures.append(f"sentence over 38 words: {' '.join(words[:12])}...")
        if punctuation_probe.count(",") > 1:
            failures.append(f"sentence with more than one comma: {' '.join(words[:12])}...")

    bibliography = re.findall(r"\\bibitem\{([^}]+)\}", source)
    citations = {
        key.strip()
        for group in re.findall(r"\\cite\{([^}]+)\}", source)
        for key in group.split(",")
    }
    if set(bibliography) != citations:
        failures.append("bibliography entries and citations do not match")
    if len(bibliography) != len(set(bibliography)):
        failures.append("duplicate bibliography key")

    if "hyperref" in source or "href" in source:
        failures.append("active-link package or command detected")

    if failures:
        print("Writing-rule checks failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Writing-rule checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
