# مشروع التنبؤ بترك العملاء 

## ١. نظرة عامة

هذا المشروع يقدّم نظامًا متكاملاً لمعالجة بيانات الاستخدام، واستخراج السمات، وتدريب نموذج للتنبؤ باحتمالية مغادرة المستخدمين، إضافةً إلى نظام إعادة تدريب، وواجهة API، وأدوات مراقبة وانحراف بيانات، وتتبّع التجارب باستخدام MLflow

---

## ٢. بنية المشروع

```
ml-churn-project/
│
├─ data/
│   ├─ user_labels.csv
│   ├─ user_features.csv
│   ├─ training_data.csv
│   ├─ model_predictions.csv
│   ├─ false_negative.csv
│   └─ false_positive.csv
│
├─ models/
│   ├─ churn_model.joblib
│   ├─ feature_columns.json
│   └─ last_metrics.json
│
├─ scripts/
│   ├─ create_labels.py
│   ├─ create_features.py
│   ├─ training_data.py
│   ├─ train_model.py
│   ├─ retrain.py
│   ├─ create_baseline_stats.py
│   └─ monitor_drift.py
│
├─ api/
│   └─ main.py
│
├─ mlruns/  ← سجلات MLflow
├─ retrain_history/
├─ monitoring_reports/
├─ notebooks/
├─ Dockerfile
├─ Makefile
├─ requirements.txt
├─ .gitignore
└─ README.md
```

---

## ٣. التثبيت والتشغيل

### إنشاء البيئة

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### تثبيت المتطلبات

```bash
pip install -r requirements.txt
```

---

## ٤. خط معالجة البيانات (Data Pipeline)

يتم تنفيذ خط معالجة البيانات عبر ٣ سكربتات رئيسية:

### ١) إنشاء الملصقات (Labels)

```bash
python scripts/create_labels.py
```

يُصنَّف المستخدم كـ **منسحب (churn)** إذا ظهر لديه حدث:

```
Cancellation Confirmation
```

الناتج: `data/user_labels.csv`

### ٢) إنشاء السمات (Features)

```bash
python scripts/create_features.py
```

يتم حساب مجموعة كبيرة من السمات:

* عدد الأغاني
* عدد الإعلانات
* عدد الجلسات
* عدد الأخطاء
* عدد الفنانين المميزين
* نسبة التفاعل (thumbs_up_ratio)
* مدة الاستخدام وغيرها

الناتج: `data/user_features.csv`

### ٣) بناء بيانات التدريب

```bash
python scripts/training_data.py
```

دمج السمات والملصقات.
الناتج: `data/training_data.csv`

---

## ٥. تدريب النموذج

```bash
python scripts/train_model.py
```

### يقوم السكربت بـ:

* تقسيم البيانات (Train/Test) بطريقة stratified
* تدريب نموذج Random Forest
* حفظ النموذج
* حفظ أعمدة الميزات للمزامنة مع الـAPI
* استخراج ملفات الأخطاء (False Positive / False Negative)
* تسجيل التجربة باستخدام MLflow

النواتج:

* `models/churn_model.joblib`
* `models/feature_columns.json`
* `models/last_metrics.json`
* ملفات الأخطاء داخل data/

---

## ٦. إعادة التدريب الدوري

لإعادة بناء النموذج مع تغير البيانات:

```bash
python scripts/retrain.py
```

### يقوم بـ:

1. إعادة إنشاء labels
2. إعادة إنشاء features
3. إعادة بناء training_data
4. تدريب نموذج جديد
5. حفظ النتائج داخل `retrain_history/`
6. تسجيل التجربة عبر MLflow

يمكن تشغيله أسبوعيًا عبر Task Scheduler.

---

## ٧. واجهة البرمجة (API) باستخدام FastAPI

تشغيل الواجهة:

```bash
uvicorn api.main:app --reload
```

### المسارات:

* **GET /** → فحص عمل الخادم
* **POST /predict** → استقبال سمات المستخدم وإرجاع:

  * churn_prediction
  * churn_probability

واجهة التوثيق:

```
http://127.0.0.1:8000/docs
```

---

## ٨. التغليف باستخدام Docker

### بناء الحاوية:

```bash
docker build -t churn-api .
```

### تشغيلها:

```bash
docker run -p 8000:8000 churn-api
```

---

## ٩. Makefile

يُسهّل تنفيذ الأوامر:

```bash
make train
make retrain
make api
make docker-build
make docker-run
```

---

## ١٠. أدوات ضبط الجودة

تم استخدام:

* **ruff** لمسح جودة الكود
* **black** لتنسيق الكود
* **pre-commit** لتشغيل الفحوص تلقائيًا قبل أي commit

تفعيل:

```bash
pre-commit install
```

---

## ١١. تتبّع التجارب باستخدام MLflow

تم دمج MLflow داخل `train_model.py`:

* تسجيل معاملات النموذج (Params)
* تسجيل المقاييس (Metrics)
* حفظ النموذج كـ Artifact

تشغيل الواجهة:

```bash
mlflow ui --backend-store-uri mlruns
```

زيارة:

```
http://127.0.0.1:5000
```

---

## ١٢. نظام مراقبة انحراف البيانات وانحراف المفهوم

### baseline_stats.py

إنشاء إحصائيات baseline لبيانات التدريب الأصلية.

### monitor_drift.py

يكتشف:

* **انحراف البيانات** (Data Drift)
* **انحراف المفهوم** (Concept Drift)
* مقارنة الأداء عبر retrain_history

النواتج داخل `monitoring_reports/`.

---

## ١٣. التحديات والاقتراحات

### التحديات:

* عدم توازن الفئات (churn قليل جدًا)
* الحاجة لمنع تسرب البيانات
* صعوبة تحديد churn بدقة
* اختلاف سلوك المستخدمين عبر الزمن

### الاقتراحات:

* تجربة نماذج مثل XGBoost
* استخدام Time-based Split بدلاً من Random Split
* إضافة تنبيهات Slack عند حدوث Drift
* نشر النموذج على خادم كامل مع مراقبة (Prometheus + Grafana)

---

## ١٤. خاتمة

هذا المشروع يقدّم نظامًا متكاملًا يشبه ما يحدث في بيئات العمل الاحترافية:
معالجة بيانات → بناء نموذج → API → إعادة تدريب → مراقبة → تتبع تجارب → جودة كود → Docker.

وهو جاهز بالكامل للتسليم ضمن تكليف مهندس تعلم آلة في ثمانية.
