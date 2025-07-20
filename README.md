# 🧠 AI-for-Sentimental-Analysis-for-Social-Media-Text

This project provides a **Python-based sentiment analysis system** using the VADER sentiment analyzer from NLTK. It processes textual data, analyzes sentiment, and creates insightful interactive visualizations using Plotly and Cufflinks.

## 🚀 Features

- 🔍 Tokenizes and cleans text using NLTK
- 😊 Performs sentiment analysis (Positive, Negative, Neutral)
- 📉 Generates sentiment scores using VADER
- 📊 Visualizes results with:
  - Pie charts
  - Bar plots
  - Time-series sentiment trends
- 🧼 Removes stopwords and unwanted characters
- 💡 Useful for product reviews, social media analysis, and customer feedback

## 🛠️ Tech Stack

| Component     | Technology        |
|---------------|-------------------|
| Language      | Python 3.x        |
| NLP           | NLTK (VADER, Tokenizer, Stopwords) |
| Visualization | Plotly, Cufflinks, Seaborn, Matplotlib |
| Data Handling | Pandas, NumPy     |

## 📁 Project Structure

```
📦 sentiment-analysis-tool
├── major.py              # Main script for sentiment analysis
├── data/                 # (Optional) Directory to store input data
├── visualizations/       # (Optional) Save generated plots here
├── requirements.txt      # List of Python dependencies
└── README.md             # Project documentation
```

## ⚙️ Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/Vure-Maneesh/AI-for-Sentimental-Analysis-for-Social-Media-Text.git
cd AI-for-Sentimental-Analysis-for-Social-Media-Text
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

4. **Download NLTK datasets**
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
```

5. **Run the script**
```bash
python major.py
```

## 📄 Requirements

Your `requirements.txt` should include:

```
nltk
numpy
pandas
seaborn
matplotlib
plotly
cufflinks
```

## 📤 Input Format

Provide text data either as:

- A list of strings in code
- A CSV file containing a column of text entries (requires slight modification in `major.py`)

## 📈 Sample Output

- **Sentiment Pie Chart**
- **Bar chart of sentiment distribution**
- **Interactive plots showing sentiment score trends**
- **Tables of cleaned and tokenized data**

## 🔧 Customization Tips

- Change the input to load from CSV
- Save visualizations to files
- Integrate with a web app using Flask or Streamlit

## 📌 TODO

- [ ] Add support for CSV input
- [ ] Export results to Excel
- [ ] Add word cloud visualizations
- [ ] Web UI using Streamlit

## 👤 Author

**Vure Maneesh**  
Computer Science and Engineering  
JNTUH University College of Engineering, Rajanna Sircilla
