# So sánh ba phiên bản

Điểm component dùng thang 0-10 và chỉ phản ánh bằng chứng có trong archive, không cộng metric chưa truy vết.

| Component | A | B | C | Winner | Evidence |
|---|---:|---:|---:|---|---|
| Scientific design | 6 | 9 | 8 | B | B defines one fixed-CNN, paired 49-condition protocol. |
| Raw data | 5 | 8 | 7 | Final | Final reruns PIL verify/full decode/SHA256 on supplied raw data. |
| EDA | 6 | 8 | 8 | B+C | B/C share the stronger multi-feature EDA; final regenerates counts. |
| Split | 5 | 9 | 8 | B | Grouped stratified split and hash leakage checks. |
| Leakage protection | 5 | 9 | 8 | B | Path and SHA overlap gates; final adds dataset guard. |
| Preprocessing | 6 | 9 | 8 | B | Canonical EXIF/RGB/letterbox implementation and parity tests. |
| Degradation | 7 | 9 | 8 | B | Deterministic per-image seeds and locked 5x3 matrix. |
| Enhancement | 6 | 8 | 8 | B+Final | Final adds Macro F1, SSIM, latency tie-break and metadata. |
| Image metrics | 7 | 8 | 8 | B | PSNR, SSIM, Delta E, brightness, contrast, edge and histogram. |
| MobileNetV2 | 0 | 5 | 4 | B+Final | A is surrogate-only; B supplies correct architecture; final runs real Keras. |
| Training | 0 | 6 | 5 | B+Final | Final fixes executable loader and captures two-stage epoch evidence. |
| Tuning | 2 | 7 | 6 | B+Final | Validation-only grid with deterministic three-level tie-break. |
| Evaluation | 2 | 8 | 5 | B+Final | B has complete 49-condition evaluator; final executes it. |
| Statistics | 1 | 8 | 3 | B | Paired bootstrap, exact McNemar and Holm correction. |
| Error analysis | 6 | 8 | 7 | A+B+Final | A taxonomy retained; B traceability extended to five required groups. |
| Notebook | 5 | 7 | 6 | B+Final | B has the best linear notebook; final replaces stale/duplicate execution path. |
| Code architecture | 4 | 9 | 7 | B | Modular src and standalone Streamlit are the canonical architecture. |
| Tests | 2 | 8 | 5 | B+Final | B has broad suite; final adds leakage, metadata and semantic gates. |
| CI | 1 | 8 | 6 | B | B has the strongest single-service CI baseline. |
| Streamlit | 3 | 9 | 6 | B | Standalone direct src calls; no internal HTTP. |
| Docker/deploy | 1 | 8 | 8 | B+C | B canonical single service; C contributes documentation/static checks. |
| README | 4 | 9 | 9 | B+C+Final | C narrative plus B truthfulness, rewritten from final facts. |
| Word content | 7 | 9 | 9 | B+C+Final | C/B academic depth; final metrics are generated from CSV. |
| Word presentation | 7 | 8 | 9 | C | C provides the best polished report layout. |
| PPT content | 6 | 8 | 8 | B+C+Final | Final removes placeholders and uses canonical sources. |
| PPT presentation | 7 | 8 | 9 | C | C provides the strongest visual template. |
| Excel | 3 | 7 | 8 | C+Final | C assignment structure; final rebuilds correct semantic formulas. |
| Data Card | 0 | 9 | 6 | B+Final | B has explicit provenance and limitations; final rerun counts. |
| Model Card | 0 | 8 | 6 | B+Final | B honest pending card; final populates only real checkpoint/results. |
| Assignment docs | 2 | 8 | 9 | C+Final | C structure plus canonical three-member metadata. |
| Validator | 1 | 9 | 5 | B+Final | B strict validator is extended to visible Office text and semantics. |
| Repository hygiene | 3 | 9 | 6 | B | B clean single-service package; C legacy services are excluded. |
| Coherence | 3 | 8 | 5 | Final | Final registry and visible-text audit remove stale administrative facts. |
