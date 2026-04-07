"""
Synthetic Generator: memanggil LLM via OpenRouter untuk menghasilkan
pasangan data (question + answer + distractors) dari PromptInput.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter

from src.dataset.step2.prompt_constructor import PromptInput
from src.dataset.step2.validator import RawDataPoint

load_dotenv(override=True)

GENERATION_SYSTEM_PROMPT = """Kamu adalah pembuat soal kuis Python untuk siswa Indonesia.

PENTING: Buat soal HANYA berdasarkan informasi yang ada di dalam teks Konteks yang diberikan. Jangan gunakan pengetahuan di luar konteks.

Berikan output HANYA dalam format berikut (tanpa teks lain, tanpa penjelasan tambahan):
Pertanyaan: <pertanyaan>? Jawaban benar: <jawaban>. Distraktor: 1) <d1> 2) <d2> 3) <d3> 4) <d4>. Misconception tags: <tag1>, <tag2>, <tag3>

Aturan:
- Pertanyaan dalam bahasa Indonesia yang natural
- Jawaban benar harus ada secara eksplisit dalam teks Konteks
- 4 distraktor yang plausible dan pedagogis (menguji miskonsepsi umum siswa)
- Misconception tags: label singkat yang menjelaskan miskonsepsi yang ditargetkan tiap distraktor
  Contoh: tokoh_pemrograman_lain, salah_versi_python, bingung_tipe_data, off_by_one, js_syntax
- Tidak ada teks lain selain format di atas"""


def _build_llm_client() -> ChatOpenRouter:
    """Membuat LLM client dari environment variables."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", "z-ai/glm-4.5-air:free")

    print(f"[DEBUG] model={model}, api_key={'SET' if api_key else 'MISSING'}")

    return ChatOpenRouter(
        model=model,
        api_key=api_key,
        temperature=float(os.getenv("OPENROUTER_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("OPENROUTER_MAX_TOKENS", "2000")),
    )


def _parse_target(raw_text: str) -> Optional[tuple[str, list[str]]]:
    """
    Validasi dan normalisasi output LLM.
    Mengekstrak target string dan misconception_tags dari response.

    Returns:
        (target_string, misconception_tags) jika valid, atau None jika format salah.
    """
    text = raw_text.strip()
    required = ["Pertanyaan:", "Jawaban benar:", "Distraktor:"]
    if not all(marker in text for marker in required):
        return None

    # Ekstrak misconception_tags jika ada
    misconception_tags: list[str] = []
    if "Misconception tags:" in text:
        # Pisahkan target dari misconception tags
        parts = text.split("Misconception tags:", 1)
        target_str = parts[0].strip().rstrip(".")
        tags_raw = parts[1].strip()
        # Parse tags: pisah dengan koma, bersihkan whitespace
        misconception_tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
    else:
        target_str = text

    return target_str, misconception_tags


def generate_datapoint(
    prompt_input: PromptInput,
    llm_client: Optional[ChatOpenRouter] = None,
    max_retries: int = 2,
) -> Optional[RawDataPoint]:
    """
    Memanggil LLM dan mengembalikan RawDataPoint, atau None jika semua retry gagal.

    Args:
        prompt_input: PromptInput dari Prompt Constructor
        llm_client: LangChain ChatOpenAI client (dibuat otomatis jika None)
        max_retries: jumlah maksimal retry setelah gagal

    Returns:
        RawDataPoint jika berhasil, None jika gagal setelah semua retry
    """
    if llm_client is None:
        llm_client = _build_llm_client()

    messages = [
        SystemMessage(content=GENERATION_SYSTEM_PROMPT),
        HumanMessage(content=prompt_input.input),
    ]

    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            response = llm_client.invoke(messages)
            raw_text = response.content

            target = _parse_target(raw_text)
            if target is None:
                # Format salah — retry
                last_error = ValueError(
                    f"Format output LLM tidak valid: '{raw_text[:100]}'"
                )
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                continue

            target_str, misconception_tags = target

            # Bangun metadata dari chunk + params
            metadata = {
                "difficulty": prompt_input.params.difficulty,
                "question_type": prompt_input.params.question_type,
                "concept": prompt_input.params.concept,
                "misconception_tags": misconception_tags,
                "source_file": prompt_input.chunk.source_file,
                "section": prompt_input.chunk.section_heading,
                "source": "synthetic",
                "validated": False,
            }

            return RawDataPoint(
                input=prompt_input.input,
                target=target_str,
                metadata=metadata,
                source="synthetic",
            )

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                # Exponential backoff: 1s, 2s, 4s
                wait = 2 ** attempt
                print(f"[WARNING] LLM error (attempt {attempt + 1}): {e}. Retry in {wait}s...")
                time.sleep(wait)

    print(f"[ERROR] generate_datapoint gagal setelah {max_retries + 1} percobaan: {last_error}")
    return None


def generate_batch(
    prompt_inputs: list[PromptInput],
    llm_client: Optional[ChatOpenRouter] = None,
    max_retries: int = 2,
    delay_between: float = 0.5,
) -> tuple[list[RawDataPoint], int]:
    """
    Generate batch data points dari list PromptInput.

    Returns:
        (results, failed_count)
    """
    if llm_client is None:
        llm_client = _build_llm_client()

    results: list[RawDataPoint] = []
    failed = 0

    for i, prompt_input in enumerate(prompt_inputs):
        result = generate_datapoint(prompt_input, llm_client, max_retries)
        if result is not None:
            results.append(result)
        else:
            failed += 1

        # Rate limiting
        if i < len(prompt_inputs) - 1 and delay_between > 0:
            time.sleep(delay_between)

    return results, failed
