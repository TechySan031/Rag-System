# 🔬 RAG System Stress Test Results

**Date:** 2026-06-02 12:29 UTC  
**Endpoint:** `https://saniyamihani-rag-system.hf.space`  
**Total Queries:** 20  
**Successful:** 20 | **Failed:** 0

---

## 1️⃣ Latency Test

### Summary Statistics

| Metric | Value |
|--------|-------|
| Total Queries | 20 |
| ⬇️ Min Latency | 3419.83 ms |
| ⬆️ Max Latency | 7159.99 ms |
| 📊 Avg Latency | 3999.47 ms |
| 📐 Median Latency | 3833.09 ms |
| 📏 Std Dev | 756.5 ms |
| 🎯 P95 Latency | 7159.99 ms |
| 🏁 P99 Latency | 7159.99 ms |

### Per-Query Breakdown

| # | Query (truncated) | Latency (ms) | Status | Confidence |
|---|-------------------|-------------|--------|------------|
| 1 | What is the main topic of the document? | 7160 | ✅ success | low (0.00) |
| 2 | Summarize the key points. | 3818 | ✅ success | low (0.00) |
| 3 | What are the prerequisites mentioned? | 3742 | ✅ success | low (0.00) |
| 4 | How does the system handle errors and edge cases? | 3997 | ✅ success | low (0.00) |
| 5 | What is the relationship between the components de… | 4000 | ✅ success | low (0.00) |
| 6 | Compare the advantages and disadvantages mentioned… | 3822 | ✅ success | low (0.00) |
| 7 | Can you explain in detail the process described in… | 3884 | ✅ success | low (0.00) |
| 8 | What are all the technical specifications, configu… | 3907 | ✅ success | low (0.00) |
| 9 | Why? | 3917 | ✅ success | low (0.00) |
| 10 | Explain. | 3833 | ✅ success | low (0.00) |
| 11 | What algorithms or methods are used? | 3736 | ✅ success | low (0.00) |
| 12 | What data formats are supported? | 3827 | ✅ success | low (0.00) |
| 13 | How is performance measured? | 3646 | ✅ success | low (0.00) |
| 14 | What security considerations are mentioned? | 3420 | ✅ success | low (0.00) |
| 15 | What are the limitations of the approach described… | 3833 | ✅ success | low (0.00) |
| 16 | How does this compare to alternative approaches? | 3734 | ✅ success | low (0.00) |
| 17 | What future improvements are suggested? | 3990 | ✅ success | low (0.00) |
| 18 | List all the tools, libraries, or frameworks menti… | 3899 | ✅ success | low (0.00) |
| 19 | What are the exact configuration values and their … | 3829 | ✅ success | low (0.00) |
| 20 | What is not covered in this document? | 3996 | ✅ success | low (0.00) |

### Pipeline Stage Breakdown (avg across successful queries)

| Stage | Avg (ms) | Min (ms) | Max (ms) |
|-------|----------|----------|----------|
| classification | 0 | 0 | 0 |
| confidence | 0 | 0 | 0 |
| context_assembly | 0 | 0 | 0 |
| generation | 1355 | 1187 | 1479 |
| reranking | 0 | 0 | 0 |
| retrieval | 1622 | 1261 | 4623 |
