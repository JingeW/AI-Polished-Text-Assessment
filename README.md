
# Article AI Detection and Readability Assessment
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

This project analyzes scientific articles to detect AI-generated content and assess readability improvements after AI polishing. It provides workflows for:
1. **Text Polishing**: Enhancing text clarity and readability using AI.
2. **AI Detection**: Identifying the probability of AI-generated text.
3. **Readability Assessment**: Calculating traditional readability metrics and scientific clarity metrics.

---

## **Project Structure**

```
ARTICLE_AI_DETECTION/
│-- data/                          # Original article text files (input)
│-- outputs/
│   └── polished_articles/         # AI-polished articles for rep1, rep2, rep3
│-- plots/                         # Generated plots
│-- results/                       # Analysis results (JSON, CSV)
│-- service/                       # Core services for analysis and preprocessing
│   │-- analysis_service.py        # Readability and clarity metrics
│   │-- data_prep_service.py       # Text preprocessing (clean-up)
│   │-- polish_service.py          # AI polishing service
│   │-- prompt_service.py          # AI prompt management
│-- main_ai_detection.py           # AI detection workflow
│-- main_article_polish.py         # Text polishing workflow
│-- main_readability_assessment.py # Readability assessment workflow
│-- plot_results.py                # Plotting results
│-- utils.py                       # Utility functions (file I/O, plotting)
│-- .env                           # Environment variables (API keys)
│-- README.md                      # Project documentation
```

---

## **Requirements**

Ensure you have the following tools and libraries installed:

- Python 3.8+
- Required libraries (installed via `requirements.txt`)

### **Dependencies**
Install the required Python libraries:

```bash
pip install -r requirements.txt
```

---

## **How to Use**

### **1. Prepare the Data**

- Place the original Excel file with articles into the `data/` folder.
- Run the preprocessing and extraction script:

   ```bash
   python main_article_polish.py
   ```

   This script will:
   - Extract articles from the Excel file.
   - Preprocess and save the original text in `data/`.

---

### **2. AI Text Polishing**

Run the polishing workflow to generate AI-enhanced versions of the text:

```bash
python main_article_polish.py --repetitions 3
```

- Polished text files will be saved in `outputs/polished_articles/`.

---

### **3. Readability Assessment**

Analyze readability and clarity of both original and polished texts:

```bash
python main_readability_assessment.py
```

- Results will be saved in `results/readability_results.csv`

---

### **4. AI Detection**

Run the AI detection workflow to evaluate articles:

```bash
python main_ai_detection.py
```

- Results will be saved in `results/ai_detection_results.csv`.

---

### **5. Plot Results**

Visualize results using the plotting script:

```bash
python plot_results.py
```

Plots will be saved in the `plots/` folder.

---

## **Environment Variables**

Add your API keys to the `.env` file in the project root:

```plaintext
GPTZERO_API_KEY=your_gptzero_api_key
API_KEY=your_openai_api_key
```

---

## **Outputs**

- **Results**: Stored in `results/` (CSV files).
- **Polished Text**: Stored in `outputs/polished_articles/`.
- **GPTZERO Responses**: Stored in `outputs/gptzero_responses/`.
- **Plots**: Visualizations saved in `plots/`.

---

## **License**

This project is licensed under the MIT License.

---
