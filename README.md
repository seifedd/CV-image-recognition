# 🔬 Vision Intelligence — FullStack Image Classifier MVP

> **Live Demo:** [vision-intelligence-classifier.netlify.app](https://vision-intelligence-classifier.netlify.app)  
> **API Docs:** [vision-intelligence-api.onrender.com/docs](https://vision-intelligence-api.onrender.com/docs)

A full-stack MVP showcasing advanced ML integration with a modern web application. Upload any image and have it classified by a K-Nearest Neighbors (k-NN) model exposed through a REST API.

---

## 🏗️ Architecture

```
[React / Vite Frontend]  ──HTTP──►  [FastAPI Backend]  ──►  [k-NN Model (sklearn + joblib)]
     Deployed on Netlify                Deployed on Render
```

## 🚀 Tech Stack

| Layer     | Technology                        |
|-----------|-----------------------------------|
| Frontend  | React, Vite, Vanilla CSS          |
| Backend   | Python, FastAPI, Uvicorn          |
| ML        | scikit-learn k-NN, OpenCV, joblib |
| Deploy    | Netlify (frontend) + Render (API) |

## 📦 Local Development

**Backend:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_and_save_model.py   # generates the model files
uvicorn main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

The app will be available at `http://localhost:5173` and talks to the backend at `http://localhost:8000`.

## 🔐 Security Notes
- CORS restricted to known frontend origins (env-configurable via `ALLOWED_ORIGINS`)
- File uploads are validated server-side (max 5 MB, must be a decodable image)
- No user data is stored — classification is stateless

## 📊 Background

The original classifier script (`chapter07-first_image_classifier/`) uses raw pixel intensities (32×32×3 = 3072 features) fed into a k-NN classifier trained on the Animals dataset (dogs, cats, pandas). It achieves ~52% accuracy — a useful baseline that demonstrates why CNNs outperform traditional ML on image tasks.

---

**By Seif** · Built with FastAPI · React · k-NN
