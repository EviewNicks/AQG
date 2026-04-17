# Research Report: Automatic Generation of Python Programming Quiz Questions and Distractors Using IndoT5 with LoRA

*Generated: April 17, 2026 | Sources: 15+ | Confidence: High*

## Executive Summary
This report explores the development of an Indonesian Automatic Question Generation (AQG) system for Python programming using **IndoNanoT5** and **LoRA**. The research focuses on the necessity of **Domain Adaptation** to bridge the gap between general Indonesian language and technical Python concepts. Key findings suggest that a two-stage approach—**Domain Adaptation (Pre-training style)** followed by **Task-Specific Fine-tuning**—is optimal. The report also provides 7+ literature references comparing this project with existing studies in Indonesian NLP and educational content generation.

---

## 1. Domain Adaptation & Dataset Strategy
In the context of this project, **Domain Adaptation** is the process of teaching the IndoNanoT5 model about the "world of Python" in Indonesian before training it for the specific task of generating questions.

### Why Domain Adaptation?
IndoNanoT5 is pre-trained on general Indonesian corpora (like CulturaX or common crawl). While it understands Indonesian, it lacks deep semantic understanding of Python-specific terms (e.g., *list, dictionary, loop, function*) and how they are explained in an Indonesian educational context.

### Recommended Dataset Types
| Dataset Type | Purpose | Format | Example Content |
| :--- | :--- | :--- | :--- |
| **Technical Corpus** | Teach technical vocabulary | Raw text / Markdown | Python documentation, Indonesian programming blogs (e.g., Dicoding, Maguru). |
| **Span Corruption** | Learn structure & syntax | Masked text | `Fungsi <extra_id_0> digunakan untuk <extra_id_1> di Python.` |
| **Code-Mixed Text** | Handle bilingual nature | Text with code blocks | Explanations where Python code is embedded within Indonesian sentences. |

---

## 2. Fine-tuning vs. Pre-training: The Strategy
Based on the T5 paper (*Raffel et al., 2019*) and recent studies on IndoT5, the recommended strategy for this project is a **Hybrid Approach**:

### Step 1: Continued Pre-training (Domain Adaptation)
*   **Action**: Perform "Self-Supervised Learning" on unlabelled Python educational content.
*   **Objective**: Use the **Span Corruption** objective (as defined in the original T5 paper) to reconstruct missing parts of technical text.
*   **Benefit**: Adapts the model's internal representations to the Python domain without needing labeled question-answer pairs.

### Step 2: Task-Specific Fine-tuning (AQG & Distractors)
*   **Action**: Use **LoRA (Low-Rank Adaptation)** on the domain-adapted model.
*   **Objective**: Train on a labeled dataset of (Context, Question, Answer, Distractors).
*   **Benefit**: Efficiently teaches the model the specific "format" of a quiz question while maintaining the domain knowledge gained in Step 1.

> **Verdict**: You should do **Domain Adaptation (as a form of continued pre-training)** followed by **Fine-tuning (Task-specific)**. Full pre-training from scratch is unnecessary and computationally expensive.

---

## 3. Literature Review (7+ References)

1. **Towards Automatic Question Generation Using Pre-trained Model in Academic Field for Bahasa Indonesia (2024)**
   *   **Summary**: This paper explores the use of decoder-based models for generating Indonesian academic questions. It highlights the transition from rule-based to generative AI for better flexibility.
   *   **Comparison**: Similar in task (AQG) but differs in domain (general academic vs. specific Python programming).
   *   **Link**: [https://link.springer.com/article/10.1007/s10639-024-12717-9](https://link.springer.com/article/10.1007/s10639-024-12717-9)

2. **High-Performance Indonesian Short Answer Grading via Reasoning-Guided Language Model Fine-Tuning (2025)**
   *   **Summary**: Uses IndoT5 and LoRA for grading short answers in Indonesian. It proves that LoRA is highly effective for specialized educational tasks in low-resource settings.
   *   **Comparison**: Uses the same architecture (IndoT5 + LoRA) but for grading instead of generation.
   *   **Link**: [https://ijecbe.ui.ac.id/go/article/view/148](https://ijecbe.ui.ac.id/go/article/view/148)

3. **IndoT5 (Text-to-Text Transfer Transformer) Algorithm for Paraphrasing Indonesian Language (2025)**
   *   **Summary**: Demonstrates the effectiveness of IndoT5 for text-to-text tasks like paraphrasing, which is a core component of AQG (especially for question variation).
   *   **Comparison**: Focuses on the "Text-to-Text" capability of IndoT5 which is the foundation of your AQG project.
   *   **Link**: [https://ejournal.uinsgd.ac.id/index.php/kjrt/article/download/1093/396](https://ejournal.uinsgd.ac.id/index.php/kjrt/article/download/1093/396)

4. **Comparison of IndoNanoT5 and IndoGPT for Advancing Indonesian Text Formalization (2025)**
   *   **Summary**: A systematic comparison showing that encoder-decoder models like IndoNanoT5 often outperform decoder-only models (GPT) in specific structured text tasks.
   *   **Comparison**: Validates your choice of IndoNanoT5 as a superior architecture for structured output tasks.
   *   **Link**: [https://www.researchgate.net/publication/397568008_Comparison_of_IndoNanoT5_and_IndoGPT_for_Advancing_Indonesian_Text_Formalization_in_Low-Resource_Settings](https://www.researchgate.net/publication/397568008_Comparison_of_IndoNanoT5_and_IndoGPT_for_Advancing_Indonesian_Text_Formalization_in_Low-Resource_Settings)

5. **idT5: Indonesian Version of Multilingual T5 Transformer (2023)**
   *   **Summary**: This is the foundational paper for Indonesian T5 adaptation. It explains how mT5 was specialized for Indonesian, which is the same logic you should apply for Python specialization.
   *   **Comparison**: Provides the methodological framework for your "Domain Adaptation" stage.
   *   **Link**: [https://arxiv.org/abs/2302.00856](https://arxiv.org/abs/2302.00856)

6. **Sequence-to-Sequence Learning for Indonesian Automatic Question Generator (2020)**
   *   **Summary**: An early exploration of Indonesian AQG using Seq2Seq architectures. It provides a baseline for dataset sizes and evaluation metrics (BLEU, ROUGE).
   *   **Comparison**: Represents the "Pre-T5" era of Indonesian AQG, serving as a baseline for your project's improvements.
   *   **Link**: [https://arxiv.org/abs/2009.13889](https://arxiv.org/abs/2009.13889)

7. **Automatic Question Answer Generation using T5 and NLP for Education (2024)**
   *   **Summary**: A review of methodologies for T5-based question generation in educational contexts, emphasizing the importance of context grounding.
   *   **Comparison**: Directly aligns with your project's goal of using T5 for educational content.
   *   **Link**: [https://www.semanticscholar.org/paper/8d07fa3cd59588cd91048e1f7aa407162be325b1](https://www.semanticscholar.org/paper/8d07fa3cd59588cd91048e1f7aa407162be325b1)

---

## Key Takeaways for Your Project
*   **Dataset**: Use Indonesian Python tutorials (Markdown/YAML) for Domain Adaptation.
*   **Technique**: Use **Span Corruption** for domain adaptation and **LoRA** for task-specific fine-tuning.
*   **Architecture**: IndoNanoT5 is an excellent choice due to its balance of size and performance in Indonesian.
*   **Strategy**: You should perform **Domain Adaptation (Continued Pre-training)** followed by **Fine-tuning**.

## Methodology
Searched 15+ queries across academic and technical sources. Analyzed the T5 paper (arXiv:1910.10683) and the user-provided project draft. Cross-referenced findings with current SOTA in Indonesian NLP.
