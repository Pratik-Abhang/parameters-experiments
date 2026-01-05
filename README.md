# Parameters Experiments for RAG Systems

This repository contains comprehensive experiments testing various parameters in Retrieval-Augmented Generation (RAG) systems using OpenAI's API and the Gemini research paper as the knowledge base.

## 📁 Project Structure

```
parameters-experiments/
├── notebooks/
│   ├── 00_pdf_embeddings.ipynb      # PDF processing & embeddings setup
│   ├── 01_temperature.ipynb         # Temperature parameter testing
│   ├── 02_top_p_top_k.ipynb        # Top-p and retrieval-k experiments
│   └── 03_max_tokens.ipynb         # Max tokens parameter testing
├── data/
│   ├── Gemini_FamilyOfMultimodelModels.pdf
│   └── rag_embeddings.pkl          # Pre-computed embeddings
├── requirements.txt
└── README.md
```

## 🚀 Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Pratik-Abhang/parameters-experiments.git
   cd parameters-experiments
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   # Create .env file in project root
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run notebooks in order:**
   - Start with `00_pdf_embeddings.ipynb` to generate embeddings
   - Then run parameter experiments: `01_temperature.ipynb`, `02_top_p_top_k.ipynb`, `03_max_tokens.ipynb`

## 📊 Current Experiments

### ✅ Completed

| Notebook | Parameter | Values Tested | Metrics |
|----------|-----------|---------------|---------|
| 01_temperature | Temperature | 0.0, 0.2, 0.5, 0.8, 1.2 | Accuracy, Completeness, Clarity, Latency |
| 02_top_p_top_k | Top-p | 0.2, 0.5, 0.9, 1.0 | Response quality, Creativity vs Consistency |
| 02_top_p_top_k | Retrieval-k | 1, 3, 5, 10, 15 | Information completeness vs noise |
| 03_max_tokens | Max Tokens | 50, 100, 200, 400, 800 | Token utilization, Response completeness |

### 🔄 In Progress

- [ ] **04_chunk_size.ipynb** - Testing different text chunk sizes (250, 500, 750, 1000, 1500 words)
- [ ] **05_chunk_overlap.ipynb** - Overlap percentage experiments (0%, 10%, 25%, 50%)
- [ ] **06_prompt_engineering.ipynb** - Different prompt templates and structures
- [ ] **07_cost_analysis.ipynb** - Token usage and cost optimization across parameters
- [ ] **08_evaluation_methods.ipynb** - Comparing LLM judge vs human evaluation vs automated metrics

## 🎯 Key Research Questions

1. **Parameter Optimization**: What are the optimal parameter combinations for different use cases?
2. **Cost vs Quality**: How to balance response quality with API costs?
3. **Chunking Strategy**: What chunking approach works best for technical documents?
4. **Evaluation Methods**: Which evaluation approach most accurately reflects real-world performance?
5. **Scalability**: How do parameters perform with larger knowledge bases?

## 📈 Metrics Tracked

- **Quality Metrics**: Accuracy, Completeness, Clarity (via LLM judge)
- **Performance Metrics**: Latency, Token usage, Cost per query
- **Retrieval Metrics**: Similarity scores, Context relevance
- **User Experience**: Response length, Readability

## 🔧 Tools & Technologies

- **Language Model**: OpenAI GPT-4o-mini
- **Embeddings**: OpenAI text-embedding-3-small
- **Document Processing**: PyMuPDF
- **Data Analysis**: Pandas, NumPy, Scikit-learn
- **Evaluation**: Custom LLM judge + automated metrics

## 📝 Contributing

1. Fork the repository
2. Create a feature branch for new experiments
3. Follow the notebook naming convention: `XX_experiment_name.ipynb`
4. Include proper documentation and results analysis
5. Submit a pull request


## 🤝 Acknowledgments

- Gemini research paper for providing the knowledge base
- OpenAI for API access
- Community contributions and feedback

---

**Next Steps**: Run `git pull` before pushing to sync with remote changes, then continue with the planned experiments!
