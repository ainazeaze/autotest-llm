"""
60 pure Python utility functions — no I/O, no external dependencies.
Covers: math, strings, sequences, dicts, type coercion, recursion, and more.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable, Iterable, TypeVar

T = TypeVar("T")


def clamp(value: float, lo: float, hi: float) -> float:
    """Return *value* constrained to [lo, hi]. Handles lo > hi by swapping."""
    lo, hi = min(lo, hi), max(lo, hi)
    return max(lo, min(value, hi))


def safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Divide two numbers, returning *fallback* instead of raising on zero."""
    return fallback if denominator == 0 else numerator / denominator


def digits_of(n: int) -> list[int]:
    """Return the decimal digits of *n* as a list (handles 0 and negatives)."""
    return [int(d) for d in str(abs(n))] if n != 0 else [0]


def is_prime(n: int) -> bool:
    """Return True if *n* is a prime number. Works correctly for n <= 1."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def fibonacci(n: int) -> list[int]:
    """Return the first *n* Fibonacci numbers. Returns [] for n <= 0."""
    if n <= 0:
        return []
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]


def percentage(part: float, whole: float, decimals: int = 2) -> float:
    """Return what percent *part* is of *whole*. Returns 0.0 when whole is 0."""
    return round(safe_divide(part * 100, whole), decimals)


def truncate(text: str, max_len: int, suffix: str = "…") -> str:
    """Shorten *text* to *max_len* chars, appending *suffix* if truncated."""
    if max_len < 0:
        raise ValueError("max_len must be >= 0")
    if len(text) <= max_len:
        return text
    cut = max(0, max_len - len(suffix))
    return text[:cut] + suffix


def slugify(text: str) -> str:
    """Convert *text* to a URL-safe slug (lowercase, hyphens, no specials)."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-")


def palindrome(s: str) -> bool:
    """Return True if *s* reads the same forwards and backwards (ignores case/spaces)."""
    cleaned = re.sub(r"[^a-z0-9]", "", s.lower())
    return cleaned == cleaned[::-1]


def count_words(text: str) -> int:
    """Count whitespace-delimited words in *text*. Empty/blank strings return 0."""
    return len(text.split())


def camel_to_snake(name: str) -> str:
    """Convert 'CamelCase' or 'mixedCase' identifiers to 'snake_case'."""
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower()


def mask_string(text: str, visible: int = 4, char: str = "*") -> str:
    """Mask all but the last *visible* characters of *text* (e.g. credit cards)."""
    if visible < 0:
        raise ValueError("visible must be >= 0")
    if len(text) <= visible:
        return text
    return char * (len(text) - visible) + text[-visible:]


def flatten(nested: list) -> list:
    """Recursively flatten an arbitrarily nested list."""
    result: list = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def chunk(seq: list[T], size: int) -> list[list[T]]:
    """Split *seq* into consecutive chunks of at most *size* items."""
    if size <= 0:
        raise ValueError("size must be > 0")
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def dedupe(seq: list[T]) -> list[T]:
    """Remove duplicates from *seq* while preserving original order."""
    seen: set = set()
    result: list[T] = []
    for item in seq:
        key = item
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def most_common(seq: Iterable) -> Any:
    """Return the most frequently occurring element; None for empty input."""
    if not seq:
        return None
    counts = Counter(seq)
    return counts.most_common(1)[0][0]


def running_total(numbers: list[float]) -> list[float]:
    """Return a list where each element is the cumulative sum up to that index."""
    totals: list[float] = []
    acc = 0.0
    for n in numbers:
        acc += n
        totals.append(acc)
    return totals


def rotate(seq: list[T], steps: int) -> list[T]:
    """Rotate *seq* left by *steps* positions (negative rotates right)."""
    if not seq:
        return []
    steps = steps % len(seq)
    return seq[steps:] + seq[:steps]


def zip_to_dict(keys: list, values: list) -> dict:
    """Zip two lists into a dict, dropping extras if lengths differ."""
    return dict(zip(keys, values))


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def invert_dict(d: dict) -> dict:
    """Swap keys and values. Duplicate values become a list of original keys."""
    result: dict = {}
    for k, v in d.items():
        if v in result:
            existing = result[v]
            if isinstance(existing, list):
                existing.append(k)
            else:
                result[v] = [existing, k]
        else:
            result[v] = k
    return result


def flatten_dict(d: dict, sep: str = ".", prefix: str = "") -> dict:
    """Collapse a nested dict into a flat one with compound keys."""
    items: dict = {}
    for k, v in d.items():
        new_key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, sep=sep, prefix=new_key))
        else:
            items[new_key] = v
    return items


def try_int(value: Any, fallback: int = 0) -> int:
    """Coerce *value* to int, returning *fallback* on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def try_float(value: Any, fallback: float = 0.0) -> float:
    """Coerce *value* to float, returning *fallback* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def is_valid_email(address: str) -> bool:
    """Basic structural check for an email address (no DNS lookup)."""
    pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, address))


def memoize(fn: Callable) -> Callable:
    """Simple memoization decorator using a plain dict cache."""
    cache: dict = {}

    def wrapper(*args):
        if args not in cache:
            cache[args] = fn(*args)
        return cache[args]

    wrapper.cache = cache  # type: ignore[attr-defined]
    return wrapper


def pipe(*fns: Callable) -> Callable:
    """Compose functions left-to-right: pipe(f, g)(x) == g(f(x))."""

    def _pipe(value):
        for fn in fns:
            value = fn(value)
        return value

    return _pipe


def retry(fn: Callable, times: int, *args, **kwargs) -> Any:
    """Call *fn* up to *times* attempts, returning the first successful result."""
    last_exc: Exception = RuntimeError("times must be >= 1")
    for _ in range(max(1, times)):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
    raise last_exc


def group_by(seq: Iterable[T], key: Callable[[T], Any]) -> dict[Any, list[T]]:
    """Partition *seq* into a dict of lists keyed by *key(item)*."""
    result: dict = {}
    for item in seq:
        k = key(item)
        result.setdefault(k, []).append(item)
    return result


def clamp_int(value: int, lo: int, hi: int) -> int:
    """Return *value* constrained to [lo, hi] (integer version)."""
    return max(lo, min(value, hi))


def sign(n: float) -> int:
    """Return 1, -1, or 0 depending on the sign of *n*."""
    if n > 0:
        return 1
    if n < 0:
        return -1
    return 0


def gcd(a: int, b: int) -> int:
    """Return the greatest common divisor of *a* and *b*."""
    while b:
        a, b = b, a % b
    return abs(a)


def lcm(a: int, b: int) -> int:
    """Return the least common multiple of *a* and *b*."""
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)


def is_power_of_two(n: int) -> bool:
    """Return True if *n* is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0


def int_to_roman(num: int) -> str:
    """Convert a positive integer (1-3999) to a Roman numeral string."""
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    result = ""
    for i, v in enumerate(val):
        while num >= v:
            result += syms[i]
            num -= v
    return result


def mean(numbers: list[float]) -> float:
    """Return the arithmetic mean of *numbers*. Raises ValueError if empty."""
    if not numbers:
        raise ValueError("mean of empty sequence")
    return sum(numbers) / len(numbers)


def median(numbers: list[float]) -> float:
    """Return the median of *numbers*. Raises ValueError if empty."""
    if not numbers:
        raise ValueError("median of empty sequence")
    s = sorted(numbers)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def variance(numbers: list[float]) -> float:
    """Return the population variance of *numbers*."""
    if not numbers:
        raise ValueError("variance of empty sequence")
    m = mean(numbers)
    return sum((x - m) ** 2 for x in numbers) / len(numbers)


def binary_search(seq: list, target: Any) -> int:
    """Return the index of *target* in sorted *seq*, or -1 if not found."""
    lo, hi = 0, len(seq) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if seq[mid] == target:
            return mid
        elif seq[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def interleave(a: list[T], b: list[T]) -> list[T]:
    """Interleave two lists element-by-element, appending leftovers."""
    result: list[T] = []
    for pair in zip(a, b):
        result.extend(pair)
    result.extend(a[len(b):])
    result.extend(b[len(a):])
    return result


def sliding_window(seq: list[T], size: int) -> list[list[T]]:
    """Return all consecutive sub-lists of length *size*."""
    if size <= 0:
        raise ValueError("size must be > 0")
    return [seq[i: i + size] for i in range(len(seq) - size + 1)]


def take_while(pred: Callable[[T], bool], seq: list[T]) -> list[T]:
    """Return leading elements of *seq* for which *pred* is True."""
    result: list[T] = []
    for item in seq:
        if not pred(item):
            break
        result.append(item)
    return result


def drop_while(pred: Callable[[T], bool], seq: list[T]) -> list[T]:
    """Drop leading elements of *seq* while *pred* is True, return the rest."""
    for i, item in enumerate(seq):
        if not pred(item):
            return seq[i:]
    return []


def partition(pred: Callable[[T], bool], seq: list[T]) -> tuple[list[T], list[T]]:
    """Split *seq* into (items where pred is True, items where pred is False)."""
    yes: list[T] = []
    no: list[T] = []
    for item in seq:
        (yes if pred(item) else no).append(item)
    return yes, no


def count_occurrences(seq: list, target: Any) -> int:
    """Count how many times *target* appears in *seq*."""
    return sum(1 for item in seq if item == target)


def snake_to_camel(name: str) -> str:
    """Convert 'snake_case' to 'camelCase'."""
    parts = name.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def strip_prefix(text: str, prefix: str) -> str:
    """Remove *prefix* from the start of *text* if present."""
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def strip_suffix(text: str, suffix: str) -> str:
    """Remove *suffix* from the end of *text* if present."""
    if suffix and text.endswith(suffix):
        return text[: -len(suffix)]
    return text


def repeat_string(s: str, n: int) -> str:
    """Return *s* repeated *n* times. Returns '' for n <= 0."""
    return s * max(0, n)


def center_text(text: str, width: int, fill: str = " ") -> str:
    """Center *text* in a field of *width* characters using *fill*."""
    return text.center(width, fill)


def is_anagram(a: str, b: str) -> bool:
    """Return True if *a* and *b* are anagrams (case-insensitive, ignores spaces)."""
    clean = lambda s: sorted(s.lower().replace(" ", ""))
    return clean(a) == clean(b)


def longest_common_prefix(strings: list[str]) -> str:
    """Return the longest common prefix shared by all strings in *strings*."""
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def word_frequency(text: str) -> dict[str, int]:
    """Return a dict mapping each word (lowercased) to its frequency in *text*."""
    words = re.findall(r"[a-z]+", text.lower())
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return freq


def compress_spaces(text: str) -> str:
    """Replace consecutive whitespace with a single space and strip ends."""
    return re.sub(r"\s+", " ", text).strip()


def safe_get(d: dict, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dicts with a sequence of keys."""
    current: Any = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


def omit(d: dict, *keys: str) -> dict:
    """Return a copy of *d* without the specified keys."""
    return {k: v for k, v in d.items() if k not in keys}


def pick(d: dict, *keys: str) -> dict:
    """Return a new dict with only the specified keys from *d*."""
    return {k: d[k] for k in keys if k in d}


def merge_lists(*lists: list) -> list:
    """Concatenate any number of lists into one."""
    result: list = []
    for lst in lists:
        result.extend(lst)
    return result


def transpose(matrix: list[list]) -> list[list]:
    """Transpose a 2-D list (list of rows → list of columns)."""
    if not matrix:
        return []
    return [list(row) for row in zip(*matrix)]
