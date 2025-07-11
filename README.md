# 🌍 Global Food Waste Analysis - Machine Learning Project

**Submission Akhir Belajar Machine Learning untuk Pemula (BMLP)**  
**Nama:** Dewi Safira Permata Sari  
**Program:** Dicoding Machine Learning Path

## 📋 Deskripsi Project

Project ini mengimplementasikan analisis machine learning terhadap dataset **Global Food Waste** menggunakan dua pendekatan utama:

1. **Unsupervised Learning (Clustering)** - Mengidentifikasi pola dan pengelompokan limbah makanan global
2. **Supervised Learning (Classification)** - Memprediksi cluster berdasarkan karakteristik negara dan kategori makanan

## 📊 Dataset

### Sumber Data

- **Dataset:** Global Food Wastage Dataset 2018-2024
- **Sumber:** [Kaggle](https://www.kaggle.com/datasets/atharvasoundankar/global-food-wastage-dataset-2018-2024)
- **Ukuran:** 5.000 baris × 8 kolom
- **Periode:** 2018-2024 (7 tahun)
- **Cakupan:** 20 negara, 8 kategori makanan

### Struktur Data

#### Fitur Numerik

- `Year` - Tahun data (2018-2024)
- `Total Waste (Tons)` - Total limbah makanan dalam ton
- `Economic Loss (Million $)` - Kerugian ekonomi dalam juta dolar
- `Avg Waste per Capita (Kg)` - Rata-rata limbah per kapita dalam kg
- `Population (Million)` - Populasi dalam juta jiwa
- `Household Waste (%)` - Persentase limbah rumah tangga

#### Fitur Kategorikal

- `Country` - Negara (20 negara)
- `Food Category` - Kategori makanan (8 kategori)

## 🗂️ Struktur Project

```
BMLP_Project/
├── [Clustering]_Submission_Akhir_BMLP_Dewi_Safira_Permata_Sari_(Updated).ipynb
├── [Klasifikasi]_Submission_Akhir_BMLP_Dewi_Safira_Permata_Sari.ipynb
├── Dataset_inisiasi.csv
├── Dataset_clustering.csv
└── README.md
```

## 📁 File Description

| File                                          | Deskripsi                                                  |
| --------------------------------------------- | ---------------------------------------------------------- |
| `[Clustering]_Submission_Akhir_BMLP_*.ipynb`  | Notebook untuk analisis clustering (unsupervised learning) |
| `[Klasifikasi]_Submission_Akhir_BMLP_*.ipynb` | Notebook untuk klasifikasi (supervised learning)           |
| `Dataset_inisiasi.csv`                        | Dataset asli tanpa label cluster                           |
| `Dataset_clustering.csv`                      | Dataset dengan hasil clustering untuk klasifikasi          |

## 🔬 Metodologi

### 1. Clustering Analysis (Unsupervised Learning)

#### Tahapan:

1. **Data Loading & EDA**

   - Exploratory Data Analysis
   - Visualisasi distribusi data
   - Analisis korelasi antar fitur

2. **Data Preprocessing**

   - Handling missing values
   - Feature scaling menggunakan StandardScaler
   - Encoding kategorikal menggunakan LabelEncoder

3. **Clustering Model**

   - **Algoritma:** K-Means Clustering
   - **Optimisasi:** Elbow Method untuk menentukan jumlah cluster optimal
   - **Evaluasi:** Silhouette Score

4. **Dimensionality Reduction**
   - Principal Component Analysis (PCA)
   - Visualisasi cluster dalam 2D

#### Libraries yang Digunakan:

```python
- pandas, numpy (data manipulation)
- matplotlib, seaborn (visualisasi)
- sklearn.cluster.KMeans (clustering)
- sklearn.decomposition.PCA (dimensionality reduction)
- sklearn.metrics.silhouette_score (evaluasi)
- yellowbrick.cluster.KElbowVisualizer (optimisasi cluster)
```

### 2. Classification Analysis (Supervised Learning)

#### Tahapan:

1. **Data Loading**

   - Menggunakan dataset hasil clustering sebagai target

2. **Data Preprocessing**

   - Label encoding untuk fitur kategorikal
   - Feature scaling menggunakan StandardScaler
   - Train-test split (80:20)

3. **Model Training & Evaluation**
   - **Algoritma yang Diuji:**
     - Logistic Regression
     - Decision Tree Classifier
     - Random Forest Classifier
     - K-Nearest Neighbors (KNN)
4. **Hyperparameter Tuning**

   - GridSearchCV untuk optimisasi parameter

5. **Model Evaluation**
   - Accuracy Score
   - Precision, Recall, F1-Score
   - Confusion Matrix
   - Classification Report

#### Libraries yang Digunakan:

```python
- sklearn.linear_model.LogisticRegression
- sklearn.tree.DecisionTreeClassifier
- sklearn.ensemble.RandomForestClassifier
- sklearn.neighbors.KNeighborsClassifier
- sklearn.model_selection.GridSearchCV
- sklearn.metrics (evaluasi model)
```

## 🎯 Objektif

### Clustering Analysis

- Mengidentifikasi pola limbah makanan global
- Mengelompokkan negara berdasarkan karakteristik limbah makanan
- Menemukan insight untuk strategi pengurangan limbah makanan

### Classification Analysis

- Memprediksi cluster negara berdasarkan karakteristik limbah makanan
- Membandingkan performa berbagai algoritma klasifikasi
- Validasi hasil clustering melalui supervised learning

## 🔧 Requirements

### Python Libraries

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
yellowbrick>=1.4.0
```

### Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn yellowbrick
```

## 🚀 Cara Menjalankan

1. **Clone atau download project ini**
2. **Install dependencies** sesuai requirements
3. **Jalankan notebook secara berurutan:**
   - Jalankan `[Clustering]_Submission_Akhir_BMLP_*.ipynb` terlebih dahulu
   - Kemudian jalankan `[Klasifikasi]_Submission_Akhir_BMLP_*.ipynb`

### Environment

- **Recommended:** Google Colab, Jupyter Notebook, atau VS Code dengan Python extension
- **Python Version:** 3.7+

## 📈 Expected Results

### Clustering Analysis

- Identifikasi optimal jumlah cluster menggunakan Elbow Method
- Visualisasi cluster dalam ruang 2D menggunakan PCA
- Analisis karakteristik setiap cluster
- Insight tentang pola limbah makanan global

### Classification Analysis

- Perbandingan performa multiple algoritma klasifikasi
- Model terbaik untuk prediksi cluster
- Validation accuracy dari hasil clustering
- Feature importance analysis

## 💡 Key Insights

Project ini bertujuan untuk:

- Memahami pola global limbah makanan di berbagai negara
- Mengidentifikasi negara/kategori yang memerlukan perhatian khusus
- Memberikan foundation untuk strategi pengurangan limbah makanan
- Demonstrasi implementasi unsupervised dan supervised learning

## 🏆 Kriteria Penilaian

Project ini memenuhi kriteria submission Dicoding BMLP:

- ✅ Dataset tanpa label (5000+ rows, 8 columns)
- ✅ Mengandung data kategorikal dan numerikal
- ✅ Implementasi clustering dengan evaluasi yang tepat
- ✅ Implementasi klasifikasi dengan multiple algoritma
- ✅ EDA yang komprehensif
- ✅ Visualisasi yang informatif
- ✅ Code documentation yang baik

## 👨‍💻 Author

**Dewi Safira Permata Sari**  
Dicoding Machine Learning Path Student

---

_Project ini dibuat sebagai submission akhir untuk kelas "Belajar Machine Learning untuk Pemula" di Dicoding Academy._
