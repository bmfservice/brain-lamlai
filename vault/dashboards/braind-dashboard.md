---
title: Bảng điều khiển Braind
type: dashboard
tags: [braind, dashboard]
date: 2026-08-13
---

# 📊 Braind Dashboard

## Kiến thức mới học
```dataview
TABLE date as "Ngày", topic as "Chủ đề"
FROM "knowledge"
WHERE date
SORT date DESC
LIMIT 10
```

## Dự án đang chạy
```dataview
TABLE status as "Trạng thái", date as "Ngày"
FROM "projects"
WHERE status != "done"
SORT date ASC
```

## Notes gần đây
```dataview
LIST FROM "" 
WHERE date
SORT date DESC
LIMIT 15
```

---
*Dashboard này là gợi ý — Braind tự ingest mọi note trong vault bất kể nằm ở đâu.*