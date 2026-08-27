# Đưa dự án lên GitHub và Streamlit Community Cloud

## Trước khi push

```bash
make test
make lint
make validate
git status --short
```

Kiểm tra không có dataset, archive, secret, cache, model tạm hoặc output render QA. Checkpoint chính chỉ
được phân phối bằng phương án phù hợp giới hạn GitHub/Streamlit, có version và SHA-256.

## Tạo repository

```bash
git init -b main
git add .
git commit -m "Complete standalone Streamlit flower restoration study"
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

Không điền một URL mẫu vào README như thể đó là repository thật. Topics gợi ý:
`computer-vision`, `image-processing`, `cnn`, `mobilenetv2`, `tensorflow`, `streamlit`,
`flower-classification`, `image-restoration`.

## Nhánh và review

- `main`: chỉ chứa bản đã qua cổng lõi.
- Nhánh công việc: `data-eda`, `cnn-experiment`, `streamlit-docs`.
- Pull request phải ghi tệp thay đổi, lệnh đã chạy, bằng chứng và rủi ro.
- Không force-push `main`; không commit token hoặc `.streamlit/secrets.toml`.

## Deploy

Sau khi `make validate-final` đạt, làm theo `docs/STREAMLIT_DEPLOYMENT.md`. Chỉ thêm URL production,
ảnh chụp và `artifacts/deployment_verification.json` sau khi kiểm tra thật ở phiên ẩn danh.
