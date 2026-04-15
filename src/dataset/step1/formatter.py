"""
Formatter: mengubah Chunk menjadi RawDomainDataPoint dalam tiga format:
- span_corruption : gaya T5 pre-training, zero LLM cost
- qa_generic      : ekstrak QA dari bold/inline code/heading, zero LLM cost
- summarization   : ringkasan via LLM (opsional)
"""
from __future__ import annotations

import math
import random
import re
import time
from dataclasses import dataclass
from typing import List, Optional

from src.dataset.step2.chunker import Chunk
from src.dataset.tokenizer_utils import count_tokens, tokenize, decode_tokens


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RawDomainDataPoint:
    input: str
    target: str
    metadata: dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Hitung token menggunakan IndoT5 tokenizer yang sebenarnya."""
    return count_tokens(text)


def _count_sentinels(text: str) -> int:
    return len(re.findall(r"<extra_id_\d+>", text))


def _clean_text_for_processing(text: str) -> str:
    """
    Bersihkan teks dari noise sebelum diproses formatter.
    Menghapus:
    - Blockquote (baris dimulai dengan > atau mengandung > " di tengah kalimat)
    - Image references (![...](...))
    - Baris yang hanya berisi separator (---, ___, ***)
    - Baris kosong berlebih

    Teks asli tetap disimpan di metadata chunk — ini hanya untuk processing.
    """
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip blockquote (dimulai dengan >)
        if stripped.startswith(">"):
            continue
        # Skip image references
        if re.match(r"!\[.*?\]\(.*?\)", stripped):
            continue
        # Skip horizontal rules
        if re.match(r"^[-_*]{3,}$", stripped):
            continue
        # Hapus inline blockquote dari dalam baris (> "..." yang muncul di tengah)
        line = re.sub(r'>\s*"[^"]*"', "", line)
        # Hapus image references inline
        line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line)
        # Skip baris yang jadi kosong setelah cleaning
        if line.strip():
            cleaned.append(line)

    # Hapus baris kosong berlebih
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned))
    return result.strip()


def _find_code_block_ranges(tokens: List[str]) -> List[tuple[int, int]]:
    """
    Kembalikan list (start, end) index token yang berada di dalam code block.
    Menggunakan state machine per-token untuk menghindari bug char-offset.
    Token yang mengandung ``` menandai batas code block.
    """
    ranges: List[tuple[int, int]] = []
    in_code = False
    start = 0

    for i, tok in enumerate(tokens):
        if "```" in tok:
            if not in_code:
                in_code = True
                start = i
            else:
                ranges.append((start, i))
                in_code = False

    # Jika code block tidak ditutup, tandai sampai akhir
    if in_code:
        ranges.append((start, len(tokens) - 1))

    return ranges


def _in_code_block(idx: int, code_ranges: List[tuple[int, int]]) -> bool:
    return any(start <= idx <= end for start, end in code_ranges)


# ---------------------------------------------------------------------------
# 1. SpanCorruptor
# ---------------------------------------------------------------------------

def corrupt_spans(
    chunk: Chunk,
    corruption_rate: float = 0.15,
    seed: Optional[int] = None,
) -> RawDomainDataPoint:
    """
    Implementasi span corruption gaya T5.

    Input:  teks dengan span diganti sentinel <extra_id_N>
    Target: sentinel diikuti span asli, diakhiri <extra_id_N+1> sebagai end marker

    Property 3: count_sentinels(input) == count_sentinels(target) - 1
    Property 4: masked tokens antara 1 dan ceil(token_count * 0.25)
    Property 5: code block tidak di-mask
    """
    if seed is not None:
        random.seed(seed)

    # Gunakan teks bersih untuk span corruption (tanpa blockquote/image noise)
    clean_text = _clean_text_for_processing(chunk.text)
    # Fallback ke teks asli jika setelah cleaning terlalu pendek
    working_text = clean_text if len(clean_text.split()) >= 10 else chunk.text

    tokens = working_text.split()
    n_tokens = len(tokens)
    code_ranges = _find_code_block_ranges(tokens)

    # Hitung target jumlah token yang di-mask
    n_mask_target = max(1, int(n_tokens * corruption_rate))
    max_mask = math.ceil(n_tokens * 0.25)

    # Kumpulkan kandidat span (tidak overlap, tidak di code block, panjang 2-5)
    spans: List[tuple[int, int]] = []  # (start, end) inklusif
    masked_count = 0
    attempts = 0
    max_attempts = n_tokens * 10

    while masked_count < n_mask_target and attempts < max_attempts and len(spans) < 10:
        attempts += 1
        span_len = random.randint(2, min(5, n_tokens))
        start = random.randint(0, n_tokens - span_len)
        end = start + span_len - 1

        # Cek tidak overlap dengan span yang sudah ada
        overlap = any(not (end < s or start > e) for s, e in spans)
        if overlap:
            continue

        # Cek tidak di dalam code block
        if any(_in_code_block(i, code_ranges) for i in range(start, end + 1)):
            continue

        # Cek tidak melebihi max_mask
        if masked_count + span_len > max_mask:
            continue

        spans.append((start, end))
        masked_count += span_len

    # Pastikan minimal 1 span jika ada token di luar code block
    non_code_tokens = [
        i for i in range(n_tokens) if not _in_code_block(i, code_ranges)
    ]
    if not spans and len(non_code_tokens) >= 2:
        start = non_code_tokens[0]
        end = min(start + 1, non_code_tokens[-1])
        spans.append((start, end))

    # Sort spans by position
    spans.sort(key=lambda x: x[0])

    # Bangun input dan target
    input_tokens: List[str] = []
    target_parts: List[str] = []
    sentinel_idx = 0
    i = 0

    while i < n_tokens:
        # Cek apakah posisi ini adalah awal span
        span_match = next((s for s in spans if s[0] == i), None)
        if span_match:
            start, end = span_match
            sentinel = f"<extra_id_{sentinel_idx}>"
            input_tokens.append(sentinel)
            # Target: sentinel + span asli
            span_text = " ".join(tokens[start:end + 1])
            target_parts.append(f"{sentinel} {span_text}")
            sentinel_idx += 1
            i = end + 1
        else:
            input_tokens.append(tokens[i])
            i += 1

    # End marker di target
    target_parts.append(f"<extra_id_{sentinel_idx}>")

    input_str = " ".join(input_tokens)
    target_str = " ".join(target_parts)

    metadata = {
        "format": "span_corruption",
        "source_file": chunk.source_file,
        "module_name": _extract_module_name(chunk.source_file),
        "section_heading": chunk.section_heading,
        "token_count": chunk.token_count,
        "has_code": chunk.has_code,
    }

    return RawDomainDataPoint(input=input_str, target=target_str, metadata=metadata)


# ---------------------------------------------------------------------------
# 2. QAGenerator
# ---------------------------------------------------------------------------

def extract_qa_pairs(chunk: Chunk) -> List[RawDomainDataPoint]:
    """
    Ekstrak pasangan QA faktual dari chunk menggunakan rule-based heuristics.

    Sumber term:
    - Bold text (**term**) → "Apa itu {term} dalam Python?"  [skip jika di blockquote]
    - Inline code (`term`) → "Jelaskan {term} dalam Python."
    - Heading text → "Apa yang dimaksud dengan {heading}?"

    Property 7: term di input question muncul di target answer
    """
    results: List[RawDomainDataPoint] = []
    seen_terms: set[str] = set()

    # Gunakan teks bersih untuk ekstraksi QA (tanpa blockquote/image noise)
    clean_text = _clean_text_for_processing(chunk.text)
    working_text = clean_text if len(clean_text.strip()) >= 20 else chunk.text

    sentences = _split_into_sentences(working_text)

    # Kumpulkan baris-baris yang merupakan blockquote (dimulai dengan >)
    blockquote_lines = set()
    for line in chunk.text.splitlines():
        stripped = line.strip()
        if stripped.startswith(">"):
            blockquote_lines.add(stripped)

    def _is_in_blockquote(term: str) -> bool:
        """Cek apakah term muncul hanya di dalam baris blockquote.
        Karena working_text sudah bersih dari blockquote, fungsi ini
        hanya perlu cek di teks asli untuk term yang mungkin lolos."""
        pattern = re.compile(r"\*\*" + re.escape(term) + r"\*\*")
        # Jika term ada di working_text (sudah bersih), berarti bukan dari blockquote
        if pattern.search(working_text):
            return False
        return True

    def _is_technical_sentence(sentence: str) -> bool:
        """Cek apakah kalimat mengandung konteks teknis Python (bukan cerita ilustrasi)."""
        technical_keywords = [
            "python", "tipe data", "fungsi", "variabel", "nilai", "objek",
            "method", "class", "list", "dict", "string", "integer", "float",
            "boolean", "tuple", "set", "array", "loop", "kondisi", "operator",
            "immutable", "mutable", "index", "slice", "syntax", "keyword",
            "module", "library", "import", "return", "parameter", "argumen",
            "pemrograman", "kode", "program", "data", "tipe",
        ]
        # Kata-kata yang menandakan konteks cerita/ilustrasi (bukan teknis)
        story_keywords = [
            "kisah", "cerita", "pria", "wanita", "tahun", "kota", "jakarta",
            "evans", "kg", "menit", "jas hujan", "kantor", "hujan",
            "berat badan", "berumur", "bernama",
        ]
        sentence_lower = sentence.lower()
        # Jika mengandung kata cerita, bukan teknis
        if any(kw in sentence_lower for kw in story_keywords):
            return False
        # Skip bullet list yang hanya mendeskripsikan properti non-teknis
        # (pola: "- **Term** — dibentuk dari..." tanpa keyword Python spesifik)
        if sentence.strip().startswith("- **") and "python" not in sentence_lower:
            # Hanya lolos jika mengandung keyword teknis yang kuat
            strong_technical = ["tipe data", "pemrograman", "boolean", "string", "integer",
                                 "float", "list", "dict", "tuple", "set", "immutable", "mutable"]
            if not any(kw in sentence_lower for kw in strong_technical):
                return False
        return any(kw in sentence_lower for kw in technical_keywords)

    # 1. Bold terms — skip jika hanya ada di blockquote atau target tidak teknis
    bold_terms = re.findall(r"\*\*([^*]+)\*\*", working_text)
    for term in bold_terms:
        term_clean = term.strip()
        if not term_clean or term_clean.lower() in seen_terms:
            continue
        # Skip angka murni atau simbol
        if re.match(r"^-?\d+(\.\d+)?([Ee][+-]?\d+)?[°%]?[A-Za-z]*$", term_clean) and len(term_clean) <= 5:
            continue
        # Skip term yang terlalu pendek (< 3 karakter)
        if len(term_clean) < 3:
            continue
        # Skip term yang merupakan angka dengan satuan (60°C, 75kg, dll)
        if re.match(r"^\d+[°%\w]{0,3}$", term_clean):
            continue
        # Skip term yang hanya muncul di blockquote
        if _is_in_blockquote(term_clean):
            continue
        seen_terms.add(term_clean.lower())

        target_sentence = _find_sentence_with_term(sentences, term_clean)
        # Skip jika target tidak mengandung konteks teknis Python
        if not target_sentence or not _is_technical_sentence(target_sentence):
            continue

        qa = RawDomainDataPoint(
            input=f"Apa itu {term_clean} dalam Python?",
            target=target_sentence,
            metadata=_qa_metadata(chunk),
        )
        results.append(qa)

    # 2. Inline code terms (skip jika sudah ada dari bold)
    inline_terms = re.findall(r"`([^`]+)`", working_text)
    for term in inline_terms:
        term_clean = term.strip()
        # Skip jika terlalu panjang (bukan nama/keyword, tapi code snippet)
        if not term_clean or len(term_clean.split()) > 4:
            continue
        # Skip angka murni (1, -20, 3.14, 0, dll)
        if re.match(r"^-?\d+(\.\d+)?([Ee][+-]?\d+)?j?$", term_clean):
            continue
        # Skip simbol/operator tunggal (+, -, *, /, :, {}, [], (), dll)
        if re.match(r"^[+\-*/=<>!&|^~%@#:,.()\[\]{}'\"\\]+$", term_clean):
            continue
        # Skip term yang terlalu pendek (< 2 karakter) atau hanya angka+simbol
        if len(term_clean) < 2:
            continue
        # Skip term yang dimulai dengan # atau ### (heading yang ikut ke inline)
        if term_clean.startswith("#"):
            continue
        # Skip term yang merupakan format Markdown (#### Heading, dll)
        if re.match(r"^#{1,6}\s", term_clean):
            continue
        # Input question harus minimal 10 token — cek dulu
        question = f"Jelaskan {term_clean} dalam Python."
        if len(question.split()) < 5:
            continue
        if term_clean.lower() in seen_terms:
            continue
        seen_terms.add(term_clean.lower())

        target_sentence = _find_sentence_with_term(sentences, term_clean)
        if not target_sentence or not _is_technical_sentence(target_sentence):
            continue
        # Skip target yang dimulai dengan heading atau separator
        if re.match(r"^#{1,6}\s|^---", target_sentence.strip()):
            continue

        qa = RawDomainDataPoint(
            input=question,
            target=target_sentence,
            metadata=_qa_metadata(chunk),
        )
        results.append(qa)

    # 3. Heading text (jika belum ada QA dari bold/inline)
    if chunk.section_heading and not results:
        heading_clean = re.sub(r"^#+\s*", "", chunk.section_heading).strip()
        if heading_clean and heading_clean.lower() not in seen_terms:
            # Cari kalimat yang mengandung heading term
            heading_sentence = _find_sentence_with_term(sentences, heading_clean)
            # Fallback: kalimat pertama yang cukup panjang
            if not heading_sentence:
                heading_sentence = next(
                    (s for s in sentences if len(s.split()) >= 5), None
                )
            if heading_sentence and _is_technical_sentence(heading_sentence):
                qa = RawDomainDataPoint(
                    input=f"Apa yang dimaksud dengan {heading_clean}?",
                    target=_clean_target(heading_sentence),
                    metadata=_qa_metadata(chunk),
                )
                results.append(qa)

    return results


def _split_into_sentences(text: str) -> List[str]:
    """Split teks menjadi kalimat-kalimat, skip baris code block dan heading."""
    # Hapus code block dulu agar tidak ikut di-split
    text_no_code = re.sub(r"```[\s\S]*?```", "", text)
    # Split di titik, tanda tanya, tanda seru
    raw = re.split(r"(?<=[.!?])\s+", text_no_code)
    result = []
    for s in raw:
        s = s.strip()
        if len(s) < 10:
            continue
        # Skip baris yang dimulai dengan heading Markdown
        if re.match(r"^#{1,6}\s", s):
            continue
        # Skip baris separator
        if re.match(r"^[-_*]{3,}$", s):
            continue
        result.append(s)
    return result


def _find_sentence_with_term(sentences: List[str], term: str) -> Optional[str]:
    """Cari kalimat pertama yang mengandung term (case-insensitive)."""
    term_lower = term.lower()
    for s in sentences:
        if term_lower in s.lower():
            return _clean_target(s.strip())
    return None


def _clean_target(text: str) -> str:
    """
    Bersihkan target string dari noise yang bisa menyebabkan model belajar pola salah.

    Menghapus:
    - Separator markdown (---) di awal
    - Heading markdown (## ...) di awal
    - Baris kosong berlebih di awal/akhir

    Ini adalah Fix 3: mencegah target yang dimulai dengan '---' atau heading
    yang menyebabkan model belajar generate separator berulang saat inference.
    """
    # Strip baris separator dan heading di awal
    lines = text.splitlines()
    while lines:
        stripped = lines[0].strip()
        if re.match(r"^[-_*]{3,}$", stripped) or re.match(r"^#{1,6}\s", stripped) or stripped == "":
            lines.pop(0)
        else:
            break
    # Strip baris kosong di akhir
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def _qa_metadata(chunk: Chunk) -> dict:
    return {
        "format": "qa_generic",
        "source_file": chunk.source_file,
        "module_name": _extract_module_name(chunk.source_file),
        "section_heading": chunk.section_heading,
        "token_count": chunk.token_count,
        "has_code": chunk.has_code,
    }


# ---------------------------------------------------------------------------
# 3. Summarizer (membutuhkan LLM)
# ---------------------------------------------------------------------------

SUMMARIZATION_SYSTEM_PROMPT = (
    "Kamu adalah asisten pendidikan Python berbahasa Indonesia.\n"
    "Tugas kamu adalah merangkum teks materi Python berikut menjadi 2-4 kalimat yang padat dan informatif.\n"
    "Pertahankan semua istilah teknis Python (nama fungsi, tipe data, keyword) dalam ringkasan.\n"
    "Gunakan bahasa Indonesia yang jelas dan mudah dipahami siswa.\n"
    "Berikan HANYA ringkasan, tanpa penjelasan tambahan."
)

MIN_TOKENS_FOR_SUMMARIZATION = 100


def summarize_chunk(
    chunk: Chunk,
    llm_client,
    max_retries: int = 2,
) -> Optional[RawDomainDataPoint]:
    """
    Hasilkan ringkasan chunk via LLM.

    Property 6: input selalu diawali "Rangkum teks berikut:"
    """
    if chunk.token_count < MIN_TOKENS_FOR_SUMMARIZATION:
        return None

    input_str = f"Rangkum teks berikut:\n\n{chunk.text}"

    for attempt in range(max_retries + 1):
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content=SUMMARIZATION_SYSTEM_PROMPT),
                HumanMessage(content=input_str),
            ]
            response = llm_client.invoke(messages)
            target = response.content.strip() if hasattr(response, "content") else str(response).strip()

            if not target:
                raise ValueError("Empty response from LLM")

            metadata = {
                "format": "summarization",
                "source_file": chunk.source_file,
                "module_name": _extract_module_name(chunk.source_file),
                "section_heading": chunk.section_heading,
                "token_count": chunk.token_count,
                "has_code": chunk.has_code,
            }
            return RawDomainDataPoint(input=input_str, target=target, metadata=metadata)

        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)  # 1s, 2s
            else:
                print(f"[WARNING] Summarizer gagal setelah {max_retries + 1} percobaan: {e}")
                return None

    return None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _extract_module_name(source_file: str) -> str:
    """Ekstrak nama modul dari path file (nama folder parent terdekat)."""
    from pathlib import Path
    parts = Path(source_file).parts
    # Cari folder yang namanya dimulai dengan angka (pola modul)
    for part in reversed(parts[:-1]):
        if re.match(r"^\d+", part):
            return part.lower()
    # Fallback: parent folder
    return parts[-2].lower() if len(parts) >= 2 else "unknown"
