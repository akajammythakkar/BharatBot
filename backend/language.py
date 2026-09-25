"""Language resolution for cross-lingual chat.

The knowledge base is English-only, so every request is resolved to:
  - input_language:  the language the question is written in (used to decide
                     whether the query must be translated before retrieval)
  - output_language: the language the answer must be written in

Resolution order
  input:  dropdown selection -> script detection -> English
  output: language requested in the prompt ("answer in Gujarati")
          -> dropdown selection -> language of the question -> English
"""
import re
from typing import Optional

DEFAULT_LANGUAGE = "en"
AUTO = "auto"

# Words that name each language, in English (plus common romanisations) and natively.
_LANG_WORDS = {
    "en": r"english|angrezi|angreji|અંગ્રેજી|अंग्रेज़ी|अंग्रेजी",
    "hi": r"hindi|હિન્દી|હિંદી|हिंदी|हिन्दी",
    "gu": r"gujarati|gujrati|ગુજરાતી|गुजराती",
}
_ANY_LANG = "|".join(f"(?P<{code}>{words})" for code, words in _LANG_WORDS.items())

_VERBS = (
    r"answer|reply|respond|response|explain|write|tell|say|give|describe|"
    r"translate|summari[sz]e|output|speak|talk"
)

# "Explain Article 21 in Gujarati", "reply to me in hindi", "answer in English please"
_EN_VERB_PATTERN = re.compile(
    rf"\b(?:{_VERBS})\b[^.?!\n]{{0,60}}?(?P<instr>\b(?:in|into)\s+(?:{_ANY_LANG})(?![A-Za-z]))",
    re.IGNORECASE,
)
# "... Article 21? in Gujarati please" at the very end of the message
_EN_TRAILING_PATTERN = re.compile(
    rf"\b(?:in|into)\s+(?:{_ANY_LANG})(?:\s+(?:please|pls|plz))?\s*[.?!]*\s*$",
    re.IGNORECASE,
)
# Native phrasing: "ગુજરાતીમાં જવાબ આપો", "हिंदी में बताइए", "gujarati ma samjhavo", "hindi mein batao"
_NATIVE_PATTERN = re.compile(
    rf"(?:{_ANY_LANG})\s*(?:માં|में|मे|\bma\b|\bmaa\b|\bmein\b|\bme\b)",
    re.IGNORECASE,
)

_GUJARATI_SCRIPT = re.compile(r"[઀-૿]")
_DEVANAGARI_SCRIPT = re.compile(r"[ऀ-ॿ]")


def normalize(code: Optional[str], supported: dict) -> Optional[str]:
    """Return a supported language code, or None for auto/empty."""
    if code is None:
        return None
    code = code.strip().lower()
    if code in ("", AUTO):
        return None
    if code not in supported:
        raise ValueError(f"Unsupported language '{code}'. Use one of: {', '.join(supported)} or '{AUTO}'.")
    return code


def detect_script_language(text: str) -> Optional[str]:
    """Detect Gujarati/Hindi by script. Returns None when the text is Latin/other."""
    gu = len(_GUJARATI_SCRIPT.findall(text))
    hi = len(_DEVANAGARI_SCRIPT.findall(text))
    if gu == 0 and hi == 0:
        return None
    return "gu" if gu >= hi else "hi"


def find_requested_language(text: str) -> tuple[Optional[str], Optional[str]]:
    """Find an explicit "answer in <language>" instruction in the message.

    Returns (language_code, instruction_phrase) or (None, None). The phrase is the
    part that names the language (e.g. "in Gujarati"), safe to strip from the query.
    """
    for pattern in (_EN_VERB_PATTERN, _EN_TRAILING_PATTERN, _NATIVE_PATTERN):
        match = None
        for match in pattern.finditer(text):
            pass  # keep the last instruction if several are present
        if match:
            for code in _LANG_WORDS:
                if match.group(code):
                    phrase = match.groupdict().get("instr") or match.group(0)
                    # "Answer in English": the verb only carries the instruction, drop it too.
                    if phrase != match.group(0) and not match.group(0)[: -len(phrase)].split()[1:]:
                        phrase = match.group(0)
                    return code, phrase
    return None, None


def resolve_languages(
    message: str,
    input_language: Optional[str],
    output_language: Optional[str],
    supported: dict,
) -> dict:
    """Work out the effective input/output languages for a message."""
    selected_in = normalize(input_language, supported)
    selected_out = normalize(output_language, supported)

    requested_out, instruction = find_requested_language(message)
    # Ignore the instruction itself so "Explain Article 21 in ગુજરાતી" still counts as English input.
    detected_in = detect_script_language(strip_instruction(message, instruction))
    effective_in = selected_in or detected_in or DEFAULT_LANGUAGE

    if requested_out:
        effective_out, out_source = requested_out, "prompt"
    elif selected_out:
        effective_out, out_source = selected_out, "selected"
    elif selected_in or detected_in:
        effective_out, out_source = effective_in, "input"
    else:
        effective_out, out_source = DEFAULT_LANGUAGE, "default"

    return {
        "input_language": effective_in,
        "output_language": effective_out,
        "output_source": out_source,
        "instruction": instruction,
    }


def strip_instruction(message: str, instruction: Optional[str]) -> str:
    """Remove the "answer in X" phrase so it doesn't pollute the retrieval query."""
    if not instruction:
        return message
    stripped = message.replace(instruction, " ", 1)
    stripped = re.sub(r"\s+", " ", stripped).strip(" ,;:-")
    return stripped or message
