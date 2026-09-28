"""
Text cleaning and tokenization for AG News and 20 Newsgroups.
"""

import re
from typing import List, Optional, Union

from transformers import AutoTokenizer


# ---- Text cleaning ----

def remove_html(text: str) -> str:
    """Remove HTML tags."""
    return re.sub(r"<[^>]+>", " ", text).strip()


def remove_urls(text: str) -> str:
    """Replace URLs with a space."""
    return re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE).strip()


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines/tabs to a single space."""
    return re.sub(r"\s+", " ", text).strip()


def remove_numbers(text: str, replace_with: str = " ") -> str:
    """Replace digit sequences with a placeholder (or remove)."""
    return re.sub(r"\d+", replace_with, text).strip()


def to_lower(text: str) -> str:
    """Lowercase text."""
    return text.lower()


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_html_tags: bool = True,
    remove_urls_flag: bool = True,
    remove_numbers_flag: bool = False,
    normalize_whitespace: bool = True,
) -> str:
    """
    Apply a configurable text cleaning pipeline.
    """
    if not text or not isinstance(text, str):
        return ""

    if remove_html_tags:
        text = remove_html(text)
    if remove_urls_flag:
        text = remove_urls(text)
    if remove_numbers_flag:
        text = remove_numbers(text)
    if normalize_whitespace:
        text = remove_extra_whitespace(text)
    if lowercase:
        text = to_lower(text)

    return remove_extra_whitespace(text)


def clean_texts(
    texts: List[str],
    **kwargs,
) -> List[str]:
    """Apply clean_text to a list of strings."""
    return [clean_text(t, **kwargs) for t in texts]


# ---- Tokenization ----

def tokenize_with_hf(
    texts: Union[List[str], str],
    tokenizer_name_or_path: str = "distilbert-base-uncased",
    max_length: int = 128,
    padding: Union[bool, str] = "max_length",
    truncation: bool = True,
    return_tensors: Optional[str] = None,
) -> dict:
    """
    Tokenize text(s) with a Hugging Face tokenizer for transformer models.

    Args:
        texts: Single string or list of strings.
        tokenizer_name_or_path: Model name or path (e.g. 'distilbert-base-uncased').
        max_length: Maximum sequence length.
        padding: 'max_length', True (batch longest), or False.
        truncation: Whether to truncate to max_length.
        return_tensors: 'pt' for PyTorch, 'tf' for TensorFlow, None for lists.

    Returns:
        Dict with input_ids, attention_mask, and optionally other keys.
    """
    if isinstance(texts, str):
        texts = [texts]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    encoded = tokenizer(
        texts,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors,
    )
    return encoded


def simple_word_tokenize(text: str, lowercase: bool = True) -> List[str]:
    """
    Simple word tokenizer: split on non-alphanumeric, keep words.
    Useful for bag-of-words or quick baselines without a full tokenizer.
    """
    if lowercase:
        text = text.lower()
    tokens = re.findall(r"\b[a-z0-9]+\b", text)
    return tokens


# ---- Batch preprocessing for datasets ----

def preprocess_for_transformers(
    texts: List[str],
    tokenizer_name_or_path: str = "distilbert-base-uncased",
    max_length: int = 128,
    clean: bool = True,
    clean_kwargs: Optional[dict] = None,
    **tokenize_kwargs,
) -> dict:
    """
    Clean (optional) and tokenize a list of texts for Hugging Face models.
    Returns tokenizer output dict (input_ids, attention_mask, etc.).
    """
    if clean:
        texts = clean_texts(texts, **(clean_kwargs or {}))
    return tokenize_with_hf(
        texts,
        tokenizer_name_or_path=tokenizer_name_or_path,
        max_length=max_length,
        **tokenize_kwargs,
    )


if __name__ == "__main__":
    sample = "  Check out https://example.com and <b>HTML</b> here.   Multiple   spaces. 123  "

    print("Raw:", repr(sample))
    print("Cleaned:", repr(clean_text(sample)))
    print("Cleaned (keep numbers):", repr(clean_text(sample, remove_numbers_flag=False)))
    print("Simple tokens:", simple_word_tokenize(clean_text(sample)))

    print("\nHF tokenizer (first 5 ids):")
    out = tokenize_with_hf([sample], max_length=32, return_tensors=None)
    print("  input_ids[0][:5]:", out["input_ids"][0][:5])
    print("  attention_mask[0][:5]:", out["attention_mask"][0][:5])
