"""
CORAL Text Normalization Layer
------------------------------
Standardizes adversarial and unstructured input text into a consistent ASCII 
format prior to TF-IDF vectorization.

Why it exists:
Spam payloads frequently use obfuscation techniques that bypass standard 
TF-IDF vocabularies. This layer normalizes these inputs so the model can 
evaluate their actual semantics:
- Unicode homoglyphs:               𝕗𝕣𝕖𝕖 𝕞𝕠𝕟𝕖𝕪 -> free money
- Full-width characters:            ｆｒｅｅ ｍｏｎｅｙ -> free money
- Emoji semantic signaling:         🎉💰 -> party_popper money_bag
- Zero-width character obfuscation: f​r​e​e -> free
- Character elongation:             heeeelloooo -> hello
- Irregular whitespace handling

Architectural Requirement:
This pipeline must be strictly symmetrical across both the offline training 
script (`train.py`) and the live inference service (`main.py`). Any divergence 
in preprocessing logic will introduce train-serve skew, causing the model to 
silently underperform on production traffic despite passing offline evaluations.
"""

import re
import unicodedata
import emoji
from unidecode import unidecode


# ---------------------------------------------------------------------------
# Step 1: Strip zero-width and invisible characters
# ---------------------------------------------------------------------------
# Spammers insert these between letters of flagged words (e.g. "f-r-e-e" with
# invisible characters) so keyword/TF-IDF matching never sees the real word.
_ZERO_WIDTH_CHARS = re.compile(
    "["
    "\u200b"  # zero width space
    "\u200c"  # zero width non-joiner
    "\u200d"  # zero width joiner
    "\ufeff"  # zero width no-break space (BOM)
    "\u2060"  # word joiner
    "]"
)


def strip_zero_width_chars(text: str) -> str:
    return _ZERO_WIDTH_CHARS.sub("", text)


# ---------------------------------------------------------------------------
# Step 2: Unicode normalization (NFKC) + transliteration to ASCII
# ---------------------------------------------------------------------------
# NFKC ("compatibility composition") folds visually-equivalent Unicode
# representations into a canonical form BEFORE we hand off to unidecode.
# This catches things like full-width characters, ligatures, and some
# stylized Unicode blocks that unidecode alone can be inconsistent with.
#
# unidecode() then transliterates "fancy font" Unicode (mathematical bold,
# fraktur, circled letters, etc.) down to plain ASCII: 𝕗𝕣𝕖𝕖 -> free
def normalize_unicode_to_ascii(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = unidecode(text)
    return text


# ---------------------------------------------------------------------------
# Step 3: Convert emoji to text tokens (do NOT just delete them)
# ---------------------------------------------------------------------------
# Emoji usage is itself a spam signal (🎉💰🔥 cluster around "urgency/reward"
# spam patterns), so we convert to words the TF-IDF vectorizer can learn
# from, rather than stripping emoji and throwing that signal away.
#   🎉 -> "party_popper"
#   💰 -> "money_bag"
def convert_emoji_to_text(text: str) -> str:
    # demojize gives ":party_popper:" - strip colons and keep it as one token
    text = emoji.demojize(text, delimiters=(" ", " "))
    return text


# ---------------------------------------------------------------------------
# Step 4: Collapse excessive character repetition
# ---------------------------------------------------------------------------
# "heeeelloooo" / "freeeeee" -> normalize repeated runs down to max 2 of the
# same character. Capped at 2 (not 1) so legitimate double letters like
# "hello" or "free" survive untouched.
_REPEATED_CHARS = re.compile(r"(.)\1{2,}")  # 3+ repeats of the same char


def collapse_repeated_chars(text: str) -> str:
    return _REPEATED_CHARS.sub(r"\1\1", text)


# ---------------------------------------------------------------------------
# Step 5: Whitespace normalization
# ---------------------------------------------------------------------------
def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def normalize_text(text: str, lowercase: bool = True) -> str:
    """
    Full normalization pipeline. Order matters:
      1. Strip zero-width/invisible chars first (before anything reads the string)
      2. Emoji -> text tokens BEFORE unidecode (unidecode silently drops
         characters it can't transliterate, which includes most emoji -
         run this too late and your emoji signal just vanishes)
      3. Unicode -> ASCII transliteration (handles fancy fonts, full-width chars)
      4. Collapse repeated characters (after steps above normalize the alphabet first)
      5. Normalize whitespace last (previous steps can introduce extra spaces)
    """
    if not isinstance(text, str):
        return ""

    text = strip_zero_width_chars(text)
    text = convert_emoji_to_text(text)
    text = normalize_unicode_to_ascii(text)
    text = collapse_repeated_chars(text)
    text = normalize_whitespace(text)

    if lowercase:
        text = text.lower()

    return text


if __name__ == "__main__":
    # Quick manual smoke test against adversarial examples
    test_cases = [
        "𝕗𝕣𝕖𝕖 𝕞𝕠𝕟𝕖𝕪 𝕔𝕝𝕚𝕔𝕜 𝕟𝕠𝕨",
        "ｆｒｅｅ ｍｏｎｅｙ ｃｌｉｃｋ ｎｏｗ",
        "Congratulations!!! 🎉💰 You've woooooon a free prize!!!",
        "f\u200br\u200be\u200be   money",
        "HELLLLOOOOO there, FREEEEE gift!!!",
        "Hello, how are you today?",
    ]

    for case in test_cases:
        print(f"BEFORE: {case!r}")
        print(f"AFTER:  {normalize_text(case)!r}")
        print("-" * 60)