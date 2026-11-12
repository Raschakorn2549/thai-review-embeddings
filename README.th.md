# คู่มือ repo (ฉบับย่อ ภาษาไทย)

ไฟล์นี้ไว้เปิดดูเร็วๆ ว่าอะไรอยู่ตรงไหนและสั่งอะไรได้บ้าง
**รายละเอียดผลลัพธ์พร้อมคำอธิบายอยู่ใน [`README.md`](README.md)** ส่วนตัวเลขดิบทุกตัวอยู่ใน
[`results/metrics.csv`](results/metrics.csv) — สองที่นั้นคือต้นทาง ไฟล์นี้ไม่เก็บตัวเลขซ้ำ

## ทำอะไร

รีวิวร้านอาหารภาษาไทยจาก Wongnai 39,993 รีวิว แปลงดาวเป็น 3 คลาส (ลบ / กลาง / บวก)
แล้วตอบ 2 คำถาม

1. embedding ภาษาไทยทำนาย sentiment ได้ดีแค่ไหน เทียบกับ TF-IDF ธรรมดา
2. PCA บีบ embedding ลงได้แค่ไหนก่อนผลจะตก

**ผลสั้นๆ:** TF-IDF + Linear SVM ชนะที่ macro-F1 0.641 · ฝั่ง embedding ดีสุด 0.628 ·
PCA ลด 1024 → 64 มิติแล้ว LightGBM ดีขึ้นนิดหน่อยด้วยซ้ำ

คลาสไม่สมดุลหนัก (บวก 63.9%) ตอบ "บวก" อย่างเดียวได้ accuracy 63.9% แล้ว
**เลยใช้ macro-F1 เป็นตัวหลัก ไม่ใช่ accuracy**

## ไฟล์ไหนทำอะไร

| ไฟล์ | หน้าที่ |
|---|---|
| `src/config.py` | ค่าตั้งทั้งหมด seed, path, มิติ PCA, รายชื่อโมเดล, grid — **แก้ที่นี่ที่เดียว** |
| `src/data.py` | โหลด ทำความสะอาด แบ่ง train/val/test |
| `src/features.py` | ตัดคำ newmm, TF-IDF, embedding bge-m3 |
| `src/train.py` | วนเทรน โมเดล × มิติ แล้วเขียนลง metrics.csv |
| `src/evaluate.py` | สร้างโมเดล, คิด metric, confusion matrix |
| `src/plotting.py` | ตั้งฟอนต์ไทย เซฟรูปลง 2 ที่พร้อมกัน |
| `notebooks/01_eda.ipynb` | สำรวจข้อมูล + รูป EDA |
| `notebooks/02_results.ipynb` | เทียบโมเดล, กราฟ PCA, error analysis |
| `notebooks/embed_colab.ipynb` | ตัวสำรอง รัน embedding บน Colab ถ้าการ์ดจอมีปัญหา |
| `tools/fetch_bge_m3.py` | ดึงน้ำหนักโมเดลจาก ModelScope ตอน huggingface เข้าไม่ได้ |

## สั่งงาน

สั่งจาก root ของ repo เสมอ

```powershell
.venv\Scripts\Activate.ps1

python -m src.data        # โหลด + ทำความสะอาด + แบ่ง -> data/processed/*.parquet
python -m src.features    # ตัดคำ + embedding        -> data/embeddings/*.npy
python -m src.train       # เทรนทุกคู่               -> results/metrics.csv
```

ทั้งสามคำสั่ง **ข้ามงานที่ทำไปแล้วเอง** ถ้าไฟล์ปลายทางมีอยู่ รันซ้ำได้ไม่เสียเวลา
ถ้าอยากรันใหม่จริงๆ ต้องลบไฟล์ปลายทางก่อน

รันเฉพาะบางส่วนได้ด้วย

```powershell
python -m src.train --part tfidf           # เฉพาะ baseline
python -m src.train --part embeddings      # เฉพาะ sweep PCA
python -m src.train --models lightgbm,knn  # เฉพาะบางโมเดล
python -m src.train --dims 64,16           # เฉพาะบางมิติ
```

`--models` กับ `--dims` คั่นด้วยจุลภาค **ห้ามเว้นวรรค**

กราฟทั้งหมดสร้างจาก `notebooks/02_results.ipynb` เซฟลง `results/figures/` และ
`report/figures/` พร้อมกัน ไม่ต้องก๊อปเอง

## เวลาที่ใช้

- embedding 39,993 รีวิว ~4.5 นาที บน RTX 3060 (ขั้นตอนเดียวที่ต้องใช้การ์ดจอ)
- sweep 28 run ~15 นาที บน CPU

## จุดที่เคยพลาด อย่าทำซ้ำ

**ห้ามแก้ label ให้ผลดูดีขึ้น** F1 ของคลาสกลางต่ำ (~0.52) เป็นเรื่องปกติ รีวิว 3 ดาว
กำกวมจริงๆ คนอ่านเองยังเดาดาวไม่ถูก — เขียนอธิบายในบทที่ 7 ไม่ใช่ไปยุ่งกับข้อมูล

**EDA เดาผิด** ตอนแรกคิดว่า embedding จะชนะเพราะ newmm ตัด `ไม่อร่อย` เป็น `ไม่` + `อร่อย`
แล้ว `ไม่` เป็น stopword แต่ baseline เก็บ stopword ไว้และใช้ bigram อยู่แล้ว `ไม่ อร่อย`
เลยรอด **อย่าเขียนในรายงานว่าคำทำนายนี้ถูก**

**bge-m3 เป็นโมเดล retrieval ไม่ใช่ sentiment** เพื่อนบ้านที่ใกล้ที่สุดของมันคือรีวิวที่พูดถึง
*เรื่องเดียวกัน* ไม่ใช่*อารมณ์เดียวกัน* (ร้านขนมอยู่ข้างร้านขนม ไม่ว่าจะชมหรือด่า)
นี่คือเหตุผลที่ kNN แย่ที่สุด

**test set แตะครั้งเดียว** จูนด้วย val เท่านั้น

## ของที่ไม่ได้อยู่ใน git

`data/` (รวมโมเดล 2.2 GB กับ embedding), `.venv/`, `tools/` ยกเว้น `fetch_bge_m3.py`,
และโน้ตส่วนตัว `CLAUDE.md` / `PLAN.md`

ถ้าสร้าง venv ใหม่ **ต้องทำ CA bundle ใหม่ก่อนลงอะไรทั้งนั้น** ไม่งั้น pip พังหมด
ขั้นตอนอยู่ใน `CLAUDE.md`
