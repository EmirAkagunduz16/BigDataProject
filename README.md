# Akademik Makalelerin Kümelenmesi

ArXiv'den akademik makale toplama ve PySpark ile kümeleme projesi.

## Kurulum

```bash
pip install -r requirements.txt
mkdir -p data visualizations logs
```

## Kullanım

```bash
python main.py --full-pipeline --max-results 2000
python main.py --collect-data --max-results 1000
python main.py --cluster --data-file data/arxiv_papers.csv
```

## Kategoriler (30 adet)

### Bilgisayar Bilimleri - AI & ML (8)
- cs.AI, cs.ML, cs.LG, cs.CV, cs.CL, cs.NE, cs.IR, cs.RO

### Bilgisayar Bilimleri - Sistemler (5)
- cs.CR, cs.DB, cs.SE, cs.DS, cs.DC

### Matematik & İstatistik (6)
- math.ST, math.PR, math.OC, math.NA, stat.ML, stat.ME

### Fizik & Mühendislik (4)
- physics.data-an, physics.comp-ph, cond-mat.stat-mech, cond-mat.soft

### Biyoloji & Tıp (4)
- q-bio.QM, q-bio.MN, q-bio.CB, q-bio.BM

### Ekonomi & Sosyal Bilimler (3)
- econ.EM, econ.TH, econ.GN

## Teknolojiler

- PySpark
- ArXiv API
- React (frontend)
- Flask (API)
- Pandas
- NLTK

## Web Interface

```bash
cd frontend && npm install && npm start
cd api && python app.py
```

Tarayıcıda: http://localhost:3000
