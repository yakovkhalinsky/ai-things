"""Quality tasks for model-bench.

Each task is a dict with:
  id          stable identifier (used in result files)
  category    json | instruction | code | reasoning | extraction | summarize
  quick       True if included in the default --suite quick
  weight      contribution to the weighted quality score
  prompt      sent to /api/generate
  num_predict max tokens to generate
  format      optional Ollama `format` value (e.g. "json")
  grader      callable(text) -> (passed: bool, detail: str)
  reference   a known-good answer; --self-test checks the grader accepts it
  note        one-line description shown in the report

Graders strip  thinking...</think> blocks and markdown fences, so reasoning
models and chatty formatting are tolerated where the task is not specifically
about format.
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path


# --- helpers ---------------------------------------------------------------

def strip_thinking(text: str) -> str:
    """Drop  thinking...</think> blocks emitted by reasoning models."""
    return re.sub(r"(?is)<think>.*?</think>", "", text or "")


def fenced(text: str, lang: str = "python") -> str | None:
    """Return the largest ```lang fenced block, if any."""
    pattern = rf"```(?:{lang}|py)?[ \t]*\n(.*?)```" if lang == "python" else r"```[ \t]*\n(.*?)```"
    blocks = re.findall(pattern, text, re.S)
    return max(blocks, key=len) if blocks else None


def first_json(text: str):
    """Parse the first JSON object/array in text, tolerating fences and prose."""
    text = strip_thinking(text)
    block = fenced(text, "json") or fenced(text)
    candidates = [block] if block else []
    candidates.append(text)
    # Fallback: slice from the first { or [ to the matching last } or ].
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate.strip())
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", strip_thinking(text))


def last_number(text: str) -> float | None:
    nums = re.findall(r"-?\d+(?:\.\d+)?", strip_thinking(text).replace(",", ""))
    return float(nums[-1]) if nums else None


def last_nonempty_line(text: str) -> str:
    lines = [ln.strip() for ln in strip_thinking(text).splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def run_python(code: str, tests: str, timeout: int = 10) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "solution.py"
        path.write_text(code + "\n\n" + tests, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(path)],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout}s"
        output = (proc.stdout + proc.stderr).strip()
        return proc.returncode == 0, output[-1200:]


def grade_code(text: str, tests: str) -> tuple[bool, str]:
    code = fenced(text, "python")
    if code is None:
        # No fence: drop any prose before the first line that starts code.
        plain = strip_thinking(text)
        match = re.search(r"(?m)^[ \t]*(?:import |from |def |class |@)", plain)
        if not match:
            return False, "no Python code found"
        code = plain[match.start():]
    return run_python(code, tests)


# --- graders ---------------------------------------------------------------

def grade_json_basic(text: str) -> tuple[bool, str]:
    data = first_json(text)
    if not isinstance(data, dict):
        return False, "not a JSON object"
    problems = []
    if not isinstance(data.get("name"), str) or "ada" not in data["name"].lower():
        problems.append("name")
    if data.get("age") != 36:
        problems.append("age")
    fields = data.get("fields")
    if not (isinstance(fields, list) and len(fields) == 3 and all(isinstance(x, str) for x in fields)):
        problems.append("fields")
    return (not problems, "ok" if not problems else "bad/missing: " + ", ".join(problems))


def grade_json_nested(text: str) -> tuple[bool, str]:
    data = first_json(text)
    if not isinstance(data, dict):
        return False, "not a JSON object"
    problems = []
    if data.get("order_id") != "A-1042":
        problems.append("order_id")
    if data.get("paid") is not True:
        problems.append("paid")
    items = data.get("items")
    if not (isinstance(items, list) and len(items) == 2):
        problems.append("items")
    elif not all(
        isinstance(i, dict) and isinstance(i.get("sku"), str) and isinstance(i.get("qty"), int)
        for i in items
    ):
        problems.append("items shape")
    total = data.get("total")
    if not (isinstance(total, (int, float)) and abs(total - 59.5) < 0.01):
        problems.append("total")
    return (not problems, "ok" if not problems else "bad/missing: " + ", ".join(problems))


def grade_word_limit(text: str) -> tuple[bool, str]:
    body = strip_thinking(text).strip()
    n = len(words(body))
    if len(body.splitlines()) != 1:
        return False, f"{len(body.splitlines())} lines (want 1)"
    if not 5 <= n <= 25:
        return False, f"{n} words (want 5-25)"
    if "recurs" not in body.lower():
        return False, "explanation does not mention recursion"
    if "```" in body or body.lstrip().startswith(("-", "*", "#")):
        return False, "used code or a list"
    return True, f"{n} words, one line"


def grade_bullet_format(text: str) -> tuple[bool, str]:
    lines = [ln for ln in strip_thinking(text).strip().splitlines() if ln.strip()]
    if len(lines) != 3:
        return False, f"{len(lines)} non-empty lines (want 3)"
    for i, line in enumerate(lines, 1):
        if not line.startswith("- "):
            return False, f"line {i} does not start with '- '"
        if len(words(line)) > 8:
            return False, f"line {i} is {len(words(line))} words (max 8)"
    return True, "3 bullets, each <=8 words"


def grade_prime_list(text: str) -> tuple[bool, str]:
    body = strip_thinking(text).strip()
    match = re.fullmatch(r"\s*(\d+)\s*,\s*(\d+)\s*", body)
    if not match:
        return False, f"not exactly 'n, n': {body[:80]!r}"
    nums = [int(match.group(1)), int(match.group(2))]
    primes = [n for n in nums if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))]
    if len(primes) != 2:
        return False, f"not both prime: {nums}"
    return True, f"primes {nums}"


def grade_fizzbuzz(text: str) -> tuple[bool, str]:
    return grade_code(text, """
assert fizzbuzz(5) == ["1", "2", "Fizz", "4", "Buzz"], fizzbuzz(5)
assert fizzbuzz(15)[14] == "FizzBuzz", fizzbuzz(15)
assert fizzbuzz(3) == ["1", "2", "Fizz"], fizzbuzz(3)
assert fizzbuzz(0) == [], fizzbuzz(0)
print("OK")
""")


def grade_slugify(text: str) -> tuple[bool, str]:
    return grade_code(text, """
assert slugify("Hello, World!") == "hello-world", slugify("Hello, World!")
assert slugify("  --A  B--  ") == "a-b", slugify("  --A  B--  ")
assert slugify("Python 3.12 is GREAT") == "python-3-12-is-great", slugify("Python 3.12 is GREAT")
assert slugify("") == "", slugify("")
assert slugify("a___b...c") == "a-b-c", slugify("a___b...c")
print("OK")
""")


def grade_bugfix(text: str) -> tuple[bool, str]:
    return grade_code(text, """
assert unique([3, 1, 3, 2, 1]) == [3, 1, 2], unique([3, 1, 3, 2, 1])
assert unique([]) == [], unique([])
assert unique(["a", "a", "b"]) == ["a", "b"], unique(["a", "a", "b"])
print("OK")
""")


def grade_math(text: str) -> tuple[bool, str]:
    answer = last_number(text)
    if answer is None:
        return False, "no number in answer"
    return (abs(answer - 6) < 1e-9, f"last number = {answer:g} (want 6)")


def grade_multiple_choice(text: str) -> tuple[bool, str]:
    body = strip_thinking(text).strip()
    letters = re.findall(r"\b([A-D])\b", body)
    if not letters:
        return False, "no answer letter A-D"
    answer = letters[-1]
    return (answer == "D", f"answered {answer} (want D)")


def grade_logic(text: str) -> tuple[bool, str]:
    line = last_nonempty_line(text).lower()
    ok = "ana" in line and "fish" in line and "ben" not in line and "cleo" not in line
    return (ok, f"final line: {line[:100]!r}")


def grade_extract(text: str) -> tuple[bool, str]:
    data = first_json(text)
    if not isinstance(data, dict):
        return False, "not a JSON object"
    problems = []
    if not isinstance(data.get("vendor"), str) or "acme" not in data["vendor"].lower():
        problems.append("vendor")
    if data.get("invoice_no") != "INV-2024-0917":
        problems.append("invoice_no")
    total = data.get("total")
    if not (isinstance(total, (int, float)) and abs(total - 1284.50) < 0.01):
        problems.append("total")
    if data.get("due_date") != "2025-03-15":
        problems.append("due_date")
    return (not problems, "ok" if not problems else "bad/missing: " + ", ".join(problems))


def grade_summary(text: str) -> tuple[bool, str]:
    body = strip_thinking(text).strip()
    n = len(words(body))
    if n > 30:
        return False, f"{n} words (max 30)"
    low = body.lower()
    missing = [fact for fact in ("arpanet", "1969") if fact not in low]
    if missing:
        return False, "missing facts: " + ", ".join(missing)
    if "```" in body or body.lstrip().startswith(("-", "*")):
        return False, "used code or a list"
    return True, f"{n} words, both facts"


def grade_sql(text: str) -> tuple[bool, str]:
    body = strip_thinking(text)
    sql = fenced(body, "sql")
    if sql is None:
        match = re.search(r"(?is)\b(select|with)\b.*", body)
        if not match:
            return False, "no SQL query found"
        sql = match.group(0)
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        return False, "empty SQL"
    con = sqlite3.connect(":memory:")
    try:
        con.executescript(SCHEMA_SQL)
        con.executescript(SEED_SQL)
        rows = con.execute(sql).fetchall()
    except sqlite3.Error as exc:
        return False, f"sqlite: {exc}"
    finally:
        con.close()
    normalized = [(str(r[0]), round(float(r[1]), 2)) for r in rows]
    expected = [("Ada", 250.0), ("Bob", 150.0)]
    return (normalized == expected, f"rows={normalized} want={expected}")


# --- task data -------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL);
"""

SEED_SQL = """
INSERT INTO customers VALUES (1, 'Ada'), (2, 'Bob'), (3, 'Cy');
INSERT INTO orders VALUES
  (1, 1, 100), (2, 1, 150), (3, 2, 150), (4, 3, 20);
"""

TASKS = [
    {
        "id": "json_basic",
        "category": "json",
        "quick": True,
        "weight": 1.0,
        "num_predict": 200,
        "format": "json",
        "note": "flat JSON object with typed fields",
        "prompt": (
            'Return ONLY a JSON object (no markdown, no prose) describing Ada Lovelace '
            'with exactly these keys: "name" (string), "age" (integer, 36), '
            '"fields" (array of exactly 3 strings).'
        ),
        "reference": '{"name": "Ada Lovelace", "age": 36, "fields": ["math", "computing", "logic"]}',
        "grader": grade_json_basic,
    },
    {
        "id": "json_nested",
        "category": "json",
        "quick": False,
        "weight": 1.0,
        "num_predict": 300,
        "format": "json",
        "note": "nested JSON with booleans, numbers and arrays",
        "prompt": (
            'Return ONLY a JSON object with these keys: "order_id" (the string "A-1042"), '
            '"paid" (boolean true), "items" (array of exactly 2 objects, each with a '
            '"sku" string and a "qty" integer), and "total" (the number 59.5).'
        ),
        "reference": (
            '{"order_id": "A-1042", "paid": true, "total": 59.5, '
            '"items": [{"sku": "WID-1", "qty": 2}, {"sku": "GAD-7", "qty": 1}]}'
        ),
        "grader": grade_json_nested,
    },
    {
        "id": "instr_words",
        "category": "instruction",
        "quick": True,
        "weight": 1.0,
        "num_predict": 120,
        "note": "length limit, single line, no list",
        "prompt": (
            "Explain what recursion is in at most 25 words. Do not use code, bullet "
            "points, or a preamble. Answer on a single line."
        ),
        "reference": (
            "Recursion is when a function calls itself to solve smaller instances of "
            "a problem until it reaches a base case."
        ),
        "grader": grade_word_limit,
    },
    {
        "id": "instr_bullets",
        "category": "instruction",
        "quick": True,
        "weight": 1.0,
        "num_predict": 150,
        "note": "exact line count, prefix and per-line word limit",
        "prompt": (
            "List exactly 3 benefits of writing unit tests. Output only three lines, "
            "each starting with '- ' and no longer than 8 words. No heading."
        ),
        "reference": (
            "- Catch bugs early\n"
            "- Document expected behavior\n"
            "- Enable safe refactoring"
        ),
        "grader": grade_bullet_format,
    },
    {
        "id": "instr_prime",
        "category": "instruction",
        "quick": True,
        "weight": 1.0,
        "num_predict": 60,
        "note": "exact output shape with a correctness constraint",
        "prompt": (
            "Give exactly two examples of a prime number, separated by a comma. "
            "Output nothing else: no words, no explanation."
        ),
        "reference": "2, 3",
        "grader": grade_prime_list,
    },
    {
        "id": "code_fizzbuzz",
        "category": "code",
        "quick": True,
        "weight": 1.0,
        "num_predict": 350,
        "note": "write a function that passes unit tests",
        "prompt": (
            "Write a Python function `fizzbuzz(n)` that returns a list of strings for "
            'the integers 1..n inclusive: "FizzBuzz" if divisible by 15, "Fizz" if '
            'divisible by 3, "Buzz" if divisible by 5, otherwise the number as a '
            "string. Return only a Python code block."
        ),
        "reference": (
            "```python\n"
            "def fizzbuzz(n):\n"
            "    out = []\n"
            "    for i in range(1, n + 1):\n"
            "        if i % 15 == 0:\n"
            '            out.append("FizzBuzz")\n'
            "        elif i % 3 == 0:\n"
            '            out.append("Fizz")\n'
            "        elif i % 5 == 0:\n"
            '            out.append("Buzz")\n'
            "        else:\n"
            "            out.append(str(i))\n"
            "    return out\n"
            "```"
        ),
        "grader": grade_fizzbuzz,
    },
    {
        "id": "code_slugify",
        "category": "code",
        "quick": False,
        "weight": 1.0,
        "num_predict": 350,
        "note": "string transformation against edge cases",
        "prompt": (
            "Write a Python function `slugify(s)` that lowercases the string, replaces "
            "every run of non-alphanumeric characters with a single '-', and strips "
            "leading/trailing '-'. Return only a Python code block."
        ),
        "reference": (
            "```python\n"
            "import re\n\n"
            "def slugify(s):\n"
            '    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")\n'
            "```"
        ),
        "grader": grade_slugify,
    },
    {
        "id": "code_bugfix",
        "category": "code",
        "quick": False,
        "weight": 1.0,
        "num_predict": 400,
        "note": "diagnose a bug and preserve insertion order",
        "prompt": (
            "This function should return the unique items in first-occurrence order, "
            "but the order is random:\n\n"
            "```python\n"
            "def unique(items):\n"
            "    return list(set(items))\n"
            "```\n\n"
            "Return only a corrected Python code block that keeps the same name and "
            "signature."
        ),
        "reference": (
            "```python\n"
            "def unique(items):\n"
            "    seen = set()\n"
            "    out = []\n"
            "    for item in items:\n"
            "        if item not in seen:\n"
            "            seen.add(item)\n"
            "            out.append(item)\n"
            "    return out\n"
            "```"
        ),
        "grader": grade_bugfix,
    },
    {
        "id": "reason_math",
        "category": "reasoning",
        "quick": True,
        "weight": 1.0,
        "num_predict": 300,
        "note": "multi-step arithmetic word problem",
        "prompt": (
            "A shop sells pencils in packs of 6. Maya buys 4 packs and gives 6 pencils "
            "to her brother. She then splits the remaining pencils equally among "
            "herself and 2 friends. How many pencils does each person get? "
            "Answer with a single number and nothing else."
        ),
        "reference": "6",
        "grader": grade_math,
    },
    {
        "id": "reason_mc",
        "category": "reasoning",
        "quick": True,
        "weight": 1.0,
        "num_predict": 300,
        "note": "logical entailment multiple choice",
        "prompt": (
            "All roses in the garden are red. Some red flowers are tulips. "
            "Which statement MUST be true?\n"
            "A) All tulips are roses\n"
            "B) Some roses are tulips\n"
            "C) No roses are tulips\n"
            "D) Some flowers in the garden are red\n"
            "Reply with the single letter of the correct answer and nothing else."
        ),
        "reference": "D",
        "grader": grade_multiple_choice,
    },
    {
        "id": "reason_logic",
        "category": "reasoning",
        "quick": False,
        "weight": 1.0,
        "num_predict": 300,
        "note": "small constraint puzzle",
        "prompt": (
            "Three people - Ana, Ben and Cleo - each own a different pet: a cat, a dog "
            "and a fish. Ana does not own the cat. Ben owns the dog. Who owns the fish? "
            "End your answer with one line naming the owner."
        ),
        "reference": "Ana owns the fish.",
        "grader": grade_logic,
    },
    {
        "id": "extract_invoice",
        "category": "extraction",
        "quick": True,
        "weight": 1.0,
        "num_predict": 250,
        "format": "json",
        "note": "pull structured fields out of messy text",
        "prompt": (
            "Extract fields from the invoice below and return ONLY a JSON object with "
            'keys "vendor" (string), "invoice_no" (string), "total" (number, no '
            'currency symbol) and "due_date" (string in YYYY-MM-DD form).\n\n'
            "INVOICE - Acme Widgets Ltd\n"
            "Inv #: INV-2024-0917\n"
            "Bill to: Northwind Traders\n"
            "Widgets (x40) ........ $1,120.00\n"
            "Shipping ............. $164.50\n"
            "TOTAL DUE: $1,284.50\n"
            "Payment due by 15 March 2025."
        ),
        "reference": (
            '{"vendor": "Acme Widgets Ltd", "invoice_no": "INV-2024-0917", '
            '"total": 1284.50, "due_date": "2025-03-15"}'
        ),
        "grader": grade_extract,
    },
    {
        "id": "summarize_facts",
        "category": "summarize",
        "quick": False,
        "weight": 1.0,
        "num_predict": 150,
        "note": "hard length cap with required facts",
        "prompt": (
            "Summarize the text below in at most 30 words. You must mention ARPANET "
            "and 1969. Output prose only, no list.\n\n"
            "The Advanced Research Projects Agency Network (ARPANET) was the first "
            "wide-area packet-switching network. Its first successful message was sent "
            "in 1969 between two computers at UCLA and Stanford. It later became the "
            "technical foundation of the Internet."
        ),
        "reference": (
            "ARPANET, the first packet-switching network, sent its first message in "
            "1969 between UCLA and Stanford, later becoming the Internet's foundation."
        ),
        "grader": grade_summary,
    },
    {
        "id": "sql_top_customers",
        "category": "code",
        "quick": False,
        "weight": 1.0,
        "num_predict": 250,
        "note": "write SQL and execute it against SQLite",
        "prompt": (
            "Given this SQLite schema:\n"
            "  customers(id INTEGER PRIMARY KEY, name TEXT)\n"
            "  orders(id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL)\n\n"
            "Write one SQL query returning the name and total order amount for the two "
            "customers with the highest total amount, highest first. Return only the "
            "SQL query."
        ),
        "reference": (
            "SELECT c.name, SUM(o.amount) AS total FROM customers c "
            "JOIN orders o ON o.customer_id = c.id "
            "GROUP BY c.id ORDER BY total DESC LIMIT 2;"
        ),
        "grader": grade_sql,
    },
]


def selected(suite: str) -> list[dict]:
    return [t for t in TASKS if suite == "full" or t["quick"]]
