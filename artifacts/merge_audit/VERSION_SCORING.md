# Chấm điểm độc lập ba phiên bản

Không cộng điểm thực nghiệm cho source chưa chạy; A bị loại toàn bộ điểm CNN do surrogate.

| Hạng mục | Max | A | B | C |
|---|---:|---:|---:|---:|
| Scientific design | 10 | 6 | 9 | 8 |
| Data/EDA/split | 12 | 7 | 11 | 10 |
| Degradation/enhancement/image metrics | 10 | 7 | 9 | 8 |
| CNN/model evidence | 15 | 0 | 4 | 3 |
| Results/statistics | 10 | 2 | 4 | 3 |
| Notebook/reproducibility | 10 | 5 | 7 | 6 |
| Code/test/CI | 10 | 4 | 9 | 7 |
| App/deploy | 7 | 2 | 6 | 5 |
| Word/PDF | 8 | 6 | 7 | 8 |
| PowerPoint | 5 | 4 | 4 | 5 |
| Repository hygiene | 3 | 1 | 3 | 2 |
| **Tổng** | **100** | **44** | **73** | **65** |

- Version A: presentation tốt nhưng CNN/result là surrogate, không đủ submission evidence.
- Version B: winner kỹ thuật, test, validator và Streamlit; thiếu FULL_RUN trong archive.
- Version C: winner trình bày Word/PPT/Excel; kiến trúc FastAPI/frontend là legacy đối với final.
