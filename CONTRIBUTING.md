# Contributing to AI Architect Textbook

Thank you for your interest in contributing. This document explains how to add content, fix issues, and submit changes.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Contribution Types](#contribution-types)
- [Chapter Writing Standards](#chapter-writing-standards)
- [Code Example Requirements](#code-example-requirements)
- [Bilingual Workflow](#bilingual-workflow)
- [Review Process](#review-process)
- [Style Guide](#style-guide)
- [Submitting a PR](#submitting-a-pr)

---

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-topic`
3. Make your changes following the standards below
4. Submit a Pull Request with a clear description

**First-time contributors**: Look for issues labeled `good-first-issue` or `help-wanted`.

---

## Contribution Types

| Type | Description | Example |
|------|-------------|---------|
| **New chapter section** | Add content to an existing chapter | Add a subsection on GPU memory optimization to Ch 7 |
| **New case study** | Add a real-world project example | Add a case study based on your production system |
| **Code example** | Add or improve runnable code | Add a Python example for LoRA fine-tuning |
| **Translation** | Translate content between CN/EN | Translate an English section to Chinese |
| **Correction** | Fix factual errors, typos, broken links | Fix an outdated tool version reference |
| **Architecture diagram** | Add or improve visual diagrams | Add a system architecture diagram for Ch 8 |

---

## Chapter Writing Standards

### File Naming

- Chapters: `chapter-XX.md` (e.g., `chapter-03.md`)
- Sections: `chXX_sY.md` (e.g., `ch03_s1.md`)
- Use English filenames for both CN and EN content

### Content Structure

Each chapter must include:

```markdown
# Chapter N: Title

> 🟢/🟡/🔴/⚫ Level marker

## Introduction
- Brief overview of what this chapter covers
- Why this topic matters for AI architects

## N.1 Section Title
### Key concepts
### Practical guidance
### Code examples
### 📌 Key Concepts (boxed summaries)

## N.2 Section Title
...

## Summary
- Key takeaways (3-5 bullet points)

## Further Reading
- Links to papers, docs, open-source projects

## Exercises
- 📝 Practical exercises with clear objectives
```

### Required Elements

- **Level marker** at the top of each chapter
- **At least one code example** per section (where applicable)
- **At least one 📌 Key Concept** per section
- **At least one 💡 Case Study** per chapter (except Ch 1, 2, 23, 25)
- **Summary section** at the end
- **Exercises section** where appropriate

### Forbidden Elements

- ❌ Vendor-specific marketing language
- ❌ Unverifiable claims or statistics
- ❌ Code without comments or explanations
- ❌ Dead links or outdated URLs

---

## Code Example Requirements

### Standards

- All code must be **syntactically correct** and **runnable**
- Include a brief comment at the top explaining what the code does
- Specify required dependencies in a comment or code block
- Use Python 3.10+ syntax unless the chapter specifically covers an older version

### Format

```python
# Example: Feature store setup with Feast
# Dependencies: feast==0.38.0, pandas
# Python: 3.10+

from feast import FeatureStore

store = FeatureStore(repo_path="./feature_repo")
features = store.get_historical_features(
    entity_df=entity_df,
    features=["driver_features:avg_rating"]
)
```

### Testing

Before submitting, verify your code:

```bash
# For Python examples
python -m py_compile your_example.py

# For shell scripts
bash -n your_script.sh
```

---

## Bilingual Workflow

This book is bilingual (Chinese + English). When contributing:

### Adding New Content

1. Write in **your preferred language** first
2. Flag the PR as needing translation: add label `needs-translation`
3. A translator will create the complementary version

### Translation Guidelines

- Translate meaning, not word-for-word
- Keep technical terms in English where standard (e.g., "Kubernetes", "RAG", "LoRA")
- Use the bilingual terminology in Appendix A as reference
- Preserve all code examples, diagrams, and formatting markers

### Section Mapping

| Source | Target |
|--------|--------|
| `src/zh/chapters/chapter-XX.md` | `src/en/chapters/chapter-XX.md` |
| `sections/zh/chXX_sY.md` | `sections/en/chXX_sY.md` |

---

## Review Process

### For Authors

1. Self-review against the checklist below
2. Run `python scripts/combine.py` to verify sections combine correctly
3. Submit PR with `ready-for-review` label

### Review Checklist

```
□ Content is factually accurate
□ Code examples are runnable and tested
□ Level marker (🟢🟡🔴⚫) is correct
□ Key concepts (📌) are present
□ Case studies (💡) are real and verifiable
□ No vendor-specific marketing language
□ Links are valid
□ Both CN and EN versions are consistent (or flagged for translation)
□ File naming follows conventions
□ No secrets, API keys, or credentials in code
```

### Reviewer Guidelines

- Focus on **accuracy**, **clarity**, and **practical value**
- Check that code examples actually run
- Verify case studies reference real projects
- Ensure level markers match content difficulty
- Flag content that violates the "no marketing" rule

---

## Style Guide

### Tone

- Professional but accessible
- Direct and actionable
- Avoid unnecessary jargon
- Prefer concrete examples over abstract explanations

### Formatting

- Use `##` for section headers (not `#` — reserved for chapter titles)
- Use `###` for subsections
- Use bold for **key terms** on first introduction
- Use inline code for `commands`, `functions`, `filenames`, and `parameters`
- Use code blocks for all code examples

### Terminology

- Prefer English technical terms with Chinese explanation in parentheses
- Use consistent terminology throughout (reference Appendix A)
- Abbreviate on first use: "Large Language Model (LLM)"

### Citations

- Reference open-source projects with GitHub URLs
- Reference papers with arxiv links where available
- Format: `[Tool Name](https://github.com/org/repo)` or `[Paper Title](https://arxiv.org/abs/XXXX.XXXXX)`

---

## Submitting a PR

### PR Title Format

```
[Chapter XX] Brief description

Examples:
[Ch 7] Add distributed training section with DeepSpeed examples
[Ch 12] Fix RAG evaluation framework code
[Translation] Add English version of Ch 21
```

### PR Description Template

```markdown
## What this PR does

## Changes made

## Checklist

- [ ] Code examples tested
- [ ] Level marker is correct
- [ ] Both CN/EN versions updated (or translation flagged)
- [ ] No secrets or credentials included
```

### Merging

- PRs require at least 1 review approval
- All CI checks must pass
- Authors merge their own PRs after approval

---

## Code of Conduct

- Be respectful and constructive
- Focus on the content, not the contributor
- Welcome newcomers and first-time contributors
- Disagreements should be resolved through evidence and examples

---

## Questions?

Open a GitHub Discussion or file an issue. We typically respond within 48 hours.
