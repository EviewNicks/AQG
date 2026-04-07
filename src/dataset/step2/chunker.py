"""
Chunker: memotong file Markdown menjadi chunk 250-400 token.
Setiap chunk mempertahankan code block utuh dan dilengkapi metadata.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# Estimasi token: jumlah kata × 1.3 (pendekatan sederhana tanpa tokenizer penuh)
def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


@dataclass
class Chunk:
    text: str               # teks chunk (konteks materi)
    source_file: str        # path file Markdown asal
    section_heading: str    # heading terdekat di atas chunk
    token_count: int        # estimasi jumlah token
    has_code: bool          # apakah chunk mengandung code block


def _has_code_block(text: str) -> bool:
    return "```" in text


def _split_at_sentence_boundary(text: str, max_tokens: int) -> List[str]:
    """Split teks panjang di batas kalimat atau baris blockquote agar tidak melebihi max_tokens."""
    # Untuk blockquote (baris dimulai dengan >), split per baris
    lines = text.strip().splitlines()
    is_blockquote = all(l.strip().startswith(">") or l.strip() == "" for l in lines if l.strip())

    if is_blockquote:
        # Split blockquote per baris
        parts: List[str] = []
        current = ""
        for line in lines:
            candidate = (current + "\n" + line).strip() if current else line
            if _estimate_tokens(candidate) <= max_tokens:
                current = candidate
            else:
                if current:
                    parts.append(current)
                current = line
        if current:
            parts.append(current)
        return parts if parts else [text]

    # Teks biasa: split di batas kalimat
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    parts = []
    current = ""

    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if _estimate_tokens(candidate) <= max_tokens:
            current = candidate
        else:
            if current:
                parts.append(current)
            # Jika satu kalimat saja sudah > max_tokens, tetap masukkan apa adanya
            current = sentence

    if current:
        parts.append(current)

    return parts if parts else [text]


def _extract_code_blocks(text: str) -> List[str]:
    """Ekstrak semua code block dari teks sebagai list string."""
    return re.findall(r"```[\s\S]*?```", text)


def _split_section_preserving_code(
    heading: str, body: str, max_tokens: int, source_file: str
) -> List[Chunk]:
    """
    Split satu section (heading + body) menjadi chunk-chunk.
    Code block tidak pernah dipotong — selalu dipertahankan utuh.
    """
    chunks: List[Chunk] = []

    # Pisahkan body menjadi segmen: teks biasa dan code block
    # Pattern: split di setiap code block, pertahankan code block sebagai token tersendiri
    segments = re.split(r"(```[\s\S]*?```)", body)

    current_text = heading + "\n" if heading else ""

    for segment in segments:
        is_code = segment.startswith("```") and segment.endswith("```")

        if is_code:
            # Code block: cek apakah muat di current chunk
            candidate = (current_text + "\n" + segment).strip()
            if _estimate_tokens(candidate) <= max_tokens:
                current_text = candidate
            else:
                # Flush current chunk dulu, lalu mulai chunk baru dengan code block
                if current_text.strip():
                    chunks.append(Chunk(
                        text=current_text.strip(),
                        source_file=source_file,
                        section_heading=heading,
                        token_count=_estimate_tokens(current_text.strip()),
                        has_code=_has_code_block(current_text),
                    ))
                current_text = (heading + "\n" + segment).strip() if heading else segment.strip()
        else:
            # Teks biasa: split di batas kalimat jika terlalu panjang
            prose_parts = _split_at_sentence_boundary(segment, max_tokens) if segment.strip() else [""]
            for part in prose_parts:
                if not part.strip():
                    continue
                candidate = (current_text + "\n" + part).strip()
                if _estimate_tokens(candidate) <= max_tokens:
                    current_text = candidate
                else:
                    if current_text.strip():
                        chunks.append(Chunk(
                            text=current_text.strip(),
                            source_file=source_file,
                            section_heading=heading,
                            token_count=_estimate_tokens(current_text.strip()),
                            has_code=_has_code_block(current_text),
                        ))
                    current_text = (heading + "\n" + part).strip() if heading else part.strip()

    # Flush sisa
    if current_text.strip():
        chunks.append(Chunk(
            text=current_text.strip(),
            source_file=source_file,
            section_heading=heading,
            token_count=_estimate_tokens(current_text.strip()),
            has_code=_has_code_block(current_text),
        ))

    return chunks


def chunk_markdown(
    filepath: str,
    max_tokens: int = 400,
    min_tokens: int = 50,
) -> List[Chunk]:
    """
    Membaca satu file Markdown dan mengembalikan list of Chunk.

    Strategi:
    - Split berdasarkan heading (#, ##, ###) — setiap heading memulai section baru
    - Code block tidak pernah dipotong
    - Jika section > max_tokens, split di batas kalimat
    """
    path = Path(filepath)
    text = path.read_text(encoding="utf-8")
    source_file = str(path)

    # Split berdasarkan heading
    # Pattern: baris yang dimulai dengan #, ##, atau ###
    heading_pattern = re.compile(r"^(#{1,3} .+)$", re.MULTILINE)
    parts = heading_pattern.split(text)

    # parts: [pre-heading-text, heading1, body1, heading2, body2, ...]
    chunks: List[Chunk] = []

    # Teks sebelum heading pertama (jika ada)
    if parts[0].strip():
        section_chunks = _split_section_preserving_code(
            heading="", body=parts[0], max_tokens=max_tokens, source_file=source_file
        )
        chunks.extend(section_chunks)

    # Iterasi pasangan (heading, body)
    i = 1
    while i < len(parts) - 1:
        heading = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        section_chunks = _split_section_preserving_code(
            heading=heading, body=body, max_tokens=max_tokens, source_file=source_file
        )
        chunks.extend(section_chunks)
        i += 2

    # Filter chunk yang terlalu pendek (< min_tokens) — merge ke chunk sebelumnya
    merged: List[Chunk] = []
    for chunk in chunks:
        if chunk.token_count < min_tokens and merged:
            prev = merged[-1]
            combined_text = prev.text + "\n\n" + chunk.text
            merged[-1] = Chunk(
                text=combined_text,
                source_file=prev.source_file,
                section_heading=prev.section_heading,
                token_count=_estimate_tokens(combined_text),
                has_code=prev.has_code or chunk.has_code,
            )
        else:
            merged.append(chunk)

    return merged


def chunk_all_materials(materi_dir: str) -> List[Chunk]:
    """
    Iterasi semua file .md secara rekursif di materi_dir.
    Mengembalikan flat list of Chunk dari semua file.
    """
    all_chunks: List[Chunk] = []
    materi_path = Path(materi_dir)

    for md_file in sorted(materi_path.rglob("*.md")):
        try:
            file_chunks = chunk_markdown(str(md_file))
            all_chunks.extend(file_chunks)
        except Exception as e:
            print(f"[WARNING] Gagal memproses {md_file}: {e}")

    return all_chunks
