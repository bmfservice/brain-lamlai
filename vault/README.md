# Obsidian Vault — cấu trúc tối ưu cho AI

Vault này được thiết kế để Braind ingest dễ dàng: mỗi note có **YAML frontmatter** gọn, **tên file dạng kebab-case**, **tags có cấu trúc**, và **liên kết có nghĩa**.

## Quy ước frontmatter
```yaml
---
title: Tên hiển thị
type: knowledge | project | reference | dasboard | daily | template
tags: [tag-a, tag-b]          # tags chia category: ngôn-ngữ, chủ-đề, hot, marketing
topic: chủ đề chính          # giúp RAG tìm theo chủ đề
source: link gốc (nếu có)
status: draft | done
date: 2026-08-13
---
```

## Thư mục
- `daily/` — ghi chú ngắn hằng ngày, cảm xúc, phân tích
- `knowledge/` — kiến thức bền vững (học từ sách, khóa học, kinh nghiệm)
- `projects/` — hồ sơ dự án đang triển khai
- `references/` — tài liệu tham khảo, link, nguồn khác
- `dashboards/` — bảng điều khiển (markdown + dataview)
- `templates/` — mẫu note (không bị ingest)

> Lưu ý: Note trong thư mục có `type:` tương ứng trong frontmatter sẽ ưu tiên được
> Braind ingest với tags + metadata đúng, giúp RAG hiểu ngữ cảnh tốt hơn.