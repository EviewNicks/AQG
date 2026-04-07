"""
Master Concept List: daftar konsep Python per modul.
Digunakan sebagai nilai metadata.concept dalam dataset.
"""
from __future__ import annotations

from typing import Dict, List

# Pemetaan folder modul → daftar konsep yang diuji
CONCEPTS: Dict[str, List[str]] = {
    "01-berkenalan-dengan-python": [
        "Sejarah Python",
        "Ciri Khas Python",
        "Versi Python 2.x",
        "Versi Python 3.x",
        "Python Software Foundation",
        "Zen of Python",
        "Fungsi print()",
        "Menjalankan File Python",
        "Variable dan Assignment",
        "Input Output Python",
        "Komentar dalam Python",
    ],
    "02-berinteraksi-dengan-data": [
        "Abstraksi Data",
        "Data Typing",
        "Tipe Data Integer",
        "Tipe Data Float",
        "Tipe Data String",
        "Tipe Data Boolean",
        "Tipe Data List",
        "Tipe Data Set",
        "Tipe Data Dictionary",
        "Type Conversion",
        "Operasi pada List",
        "Operasi pada String",
    ],
    "03-ekspresi": [
        "Pengertian Ekspresi",
        "Ekspresi Aritmatika",
        "Ekspresi Relasional",
        "Ekspresi Logika",
        "Operator Aritmatika",
        "Operator Perbandingan",
        "Operator Logika",
        "Operator Assignment",
        "Prioritas Operator",
    ],
    "04-aksi-sekuensial": [
        "Aksi Sekuensial",
        "Python Interpreter",
        "One-liner Python",
        "Urutan Eksekusi Program",
    ],
    "05-control-flow": [
        "Percabangan if-else",
        "Ternary Operator",
        "Perulangan for",
        "Perulangan while",
        "Break dan Continue",
        "Error Handling",
        "Try-Except",
        "Exception",
    ],
    "06-array": [
        "Fundamental Array",
        "Implementasi Array dengan Python",
        "Pemrosesan Sekuensial Array",
        "Indeks Array",
        "Traversal Array",
    ],
    "07-matriks": [
        "Fundamental Matriks",
        "Implementasi Matriks Python",
        "Operasi Matriks",
        "Matriks 2D",
        "Traversal Matriks",
    ],
    "08-subprogram": [
        "Definisi Subprogram",
        "Fungsi Python",
        "Parameter Fungsi",
        "Return Value",
        "Prosedur",
        "Scope Variabel",
        "Rekursi",
    ],
    "09-oop": [
        "Duck Typing",
        "Class dan Object",
        "Method",
        "Constructor __init__",
        "Inheritance",
        "Encapsulation",
        "Polymorphism",
    ],
    "10-style-guide": [
        "PEP 8 Style Guide",
        "Format Kode Python",
        "Penamaan Variabel",
        "Indentasi",
        "Style Guide Statement",
        "Prinsip Clean Code",
    ],
    "11-unit-testing": [
        "Pengenalan Unit Test",
        "Penerapan Unit Test",
        "pytest",
        "Test Case",
        "Assertion",
    ],
}


def get_concepts_for_module(module_folder: str) -> List[str]:
    """
    Mengembalikan daftar konsep untuk folder modul tertentu.
    module_folder: nama folder modul (misal '01-berkenalan-dengan-python').
    """
    # Normalisasi: ambil bagian setelah angka prefix jika perlu
    key = module_folder.lower().strip("/").split("/")[-1]
    return CONCEPTS.get(key, [])


def get_all_concepts() -> List[str]:
    """Mengembalikan semua konsep dari semua modul (flat list, tanpa duplikat)."""
    seen = set()
    result = []
    for concepts in CONCEPTS.values():
        for c in concepts:
            if c not in seen:
                seen.add(c)
                result.append(c)
    return result
