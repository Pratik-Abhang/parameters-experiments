# RAG Parameters Experiments

This project provides comprehensive experiments testing various parameters in Retrieval-Augmented Generation (RAG) systems using OpenAI's API and the Gemini research paper as the knowledge base.

## 📁 Project Structure

```
parameters-experiments/
├── notebooks/
│   ├── 01_embedding_models_comparison.ipynb      # Compare embedding models
│   ├── 02A_similarity_methods_comparison.ipynb   # Compare similarity search methods  
│   ├── 02B_similarity_methods_llm_judge.ipynb    # LLM-as-Judge evaluation
│   ├── 03_temperature.ipynb                     # Temperature parameter testing
│   ├── 04_top_p_top_k.ipynb                    # Top-p and retrieval-k experiments
│   └── 05_max_tokens.ipynb                     # Max tokens parameter testing ⭐ ENHANCED
├── data/
│   ├── Gemini_FamilyOfMultimodelModels.pdf
│   ├── rag_embeddings.pkl                      # Pre-computed embeddings
│   ├── temperature_rag_detailed.pkl            # Temperature experiment results
│   ├── similarity_methods_comparison.pkl       # Similarity methods results
│   ├── similarity_methods_llm_judge.pkl        # LLM judge results
│   └── max_tokens_results.pkl                  # Max tokens results (NEW)
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
   - **Foundation**: `01_embedding_models_comparison.ipynb`, `02A_similarity_methods_comparison.ipynb`, `02B_similarity_methods_llm_judge.ipynb`
   - **Parameters**: `03_temperature.ipynb`, `04_top_p_top_k.ipynb`, `05_max_tokens.ipynb`

## 🎯 Experiment Overview

### Phase 1: Foundation (Embedding & Search)
**01 - Embedding Models Comparison**
- Compares text-embedding-3-small, 3-large, and ada-002
- Evaluates: accuracy, speed, cost, storage requirements
- Provides cost/performance analysis for production decisions

**02A - Similarity Search Methods**
- Tests cosine, euclidean, manhattan, dot product similarities
- Analyzes retrieval accuracy and computational speed
- Recommends optimal methods for different use cases

**02B - LLM-as-Judge Evaluation**
- Enhanced evaluation using GPT-4o-mini as judge
- Measures actual relevance vs similarity scores
- More meaningful assessment than mathematical similarity alone

### Phase 2: Generation Parameters
**03 - Temperature Effects**
- Tests temperature values: 0.0, 0.2, 0.5, 0.7, 1.2, 1.5, 2.0
- Enhanced evaluation with 6 metrics including creativity, diversity, consistency
- Comprehensive analysis with statistical summaries and visualizations
- Uses LLM judge with temperature-specific criteria

**04 - Top-p and Retrieval-k Analysis**
- Explores nucleus sampling (top-p) from 0.1 to 1.0
- Tests retrieval-k values from 1 to 15 chunks
- Two-phase approach: generation parameters vs retrieval parameters

**05 - Max Tokens Impact** ⭐ ENHANCED
- Tests response length limits: 50, 100, 200, 400, 800, 1200 tokens
- **Comprehensive evaluation**: 6 quality dimensions + efficiency metrics
- **Truncation analysis**: Automatic detection of cut-off responses
- **Token efficiency**: Quality per token ratio analysis
- **Cost-effectiveness**: Optimal token ranges for different use cases

## 📊 Experiments Status

### ✅ Completed & Enhanced

| Notebook | Parameter | Values Tested | Key Metrics | Status |
|----------|-----------|---------------|-------------|---------|
| 01 | Embedding Models | 3-small, 3-large, ada-002 | Accuracy, Speed, Cost | ✅ Complete |
| 02A | Similarity Methods | Cosine, Euclidean, Manhattan, Dot Product | Retrieval Accuracy, Speed | ✅ Complete |
| 02B | LLM Judge | Same methods with relevance scoring | Actual Relevance, Judge Accuracy | ✅ Complete |
| 03 | Temperature | 0.0, 0.2, 0.5, 0.7, 1.2, 1.5, 2.0 | Accuracy, Creativity, Diversity, Consistency | ✅ Complete |
| 04 | Top-p & Retrieval-k | Top-p: 0.1-1.0, k: 1-15 | Quality, Completeness, Clarity | ✅ Complete |
| 05 | Max Tokens | 50, 100, 200, 400, 800, 1200 | Quality, Efficiency, Truncation, Utilization | ⭐ **ENHANCED** |

### 🔄 Future Experiments

#### Advanced Parameters
- [ ] **06_frequency_penalty.ipynb** - Frequency and presence penalties
- [ ] **07_model_comparison.ipynb** - GPT-4o-mini vs GPT-4o vs GPT-3.5-turbo
- [ ] **08_system_prompts.ipynb** - Different system prompt strategies

#### Retrieval Optimization
- [ ] **09_chunk_size.ipynb** - Different text chunk sizes (250-1500 words)
- [ ] **10_chunk_overlap.ipynb** - Overlap percentage experiments (0%-50%)
- [ ] **11_similarity_thresholds.ipynb** - Retrieval similarity score thresholds

## 🎯 Key Research Questions

1. **Parameter Optimization**: What are the optimal parameter combinations for different use cases?
2. **Cost vs Quality**: How to balance response quality with API costs?
3. **Token Efficiency**: What's the sweet spot for max tokens in different scenarios?
4. **Evaluation Methods**: Which evaluation approaches most accurately reflect real-world performance?

## 📈 Max Tokens Experiment Highlights ⭐

The enhanced max tokens notebook provides:

### **Comprehensive Evaluation Framework**
- **6 quality dimensions**: Accuracy, Completeness, Clarity, Coherence, Efficiency, Overall
- **LLM-as-Judge**: Enhanced judge function specifically for token limit evaluation
- **Truncation detection**: Automatic identification of cut-off responses

### **Advanced Metrics**
- **Token Utilization Rate**: Percentage of allocated tokens actually used
- **Token Efficiency**: Quality per token ratio (cost-effectiveness)
- **Truncation Analysis**: Percentage of responses cut off by token limits
- **Quality Trade-offs**: Comprehensive analysis of quality vs efficiency

### **Key Findings**
- **Quality Plateau**: Performance peaks around 400 tokens
- **Efficiency Decay**: Exponential decrease in quality per token
- **Truncation Threshold**: 400+ tokens needed to eliminate cut-offs
- **Sweet Spot**: 200-400 tokens for most applications
- **Avoid**: 800+ tokens due to diminishing returns

### **Production Recommendations**
- **Cost-Conscious**: 50-100 tokens for simple questions
- **Balanced**: 200 tokens for general use cases
- **Quality-Focused**: 400 tokens for comprehensive answers
- **Avoid**: 800+ tokens (high cost, minimal benefit)

## 🔧 Enhanced Evaluation Framework

### **LLM-as-Judge System**
- **Model**: GPT-4o-mini for consistent evaluation
- **Temperature**: 0.1 for reliable scoring
- **Criteria**: Parameter-specific evaluation dimensions
- **Validation**: Score normalization and error handling

### **Quality Metrics**
- **Accuracy**: Factual correctness based on context
- **Completeness**: Addresses all parts of the question
- **Clarity**: Well-structured and understandable
- **Coherence**: Proper endings, not abruptly cut off
- **Efficiency**: Good use of available resources
- **Overall**: Combined quality assessment

### **Performance Metrics**
- **Response Time**: Generation latency
- **Token Usage**: Actual vs allocated tokens
- **Utilization Rate**: Efficiency of token allocation
- **Cost Analysis**: Quality per dollar spent

## 🔧 Tools & Technologies

- **Language Model**: OpenAI GPT-4o-mini
- **Embeddings**: OpenAI text-embedding-3-small (optimized choice)
- **Document Processing**: PyMuPDF for PDF extraction
- **Data Analysis**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn with professional styling
- **Evaluation**: Enhanced LLM judge with parameter-specific criteria

## 📊 Visualization Framework

Each notebook includes comprehensive visualizations:
- **Performance curves**: Parameter vs quality relationships
- **Efficiency analysis**: Cost-benefit trade-offs
- **Distribution plots**: Score distributions and outliers
- **Correlation heatmaps**: Parameter interaction effects
- **Trade-off scatter plots**: Multi-dimensional analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch for new experiments
3. Follow the notebook naming convention: `XX_experiment_name.ipynb`
4. Include comprehensive analysis and visualizations
5. Use the established evaluation framework
6. Submit a pull request with detailed documentation

## 📄 License

This project is open source and available under the MIT License.

---

**Latest Update**: Max tokens notebook enhanced with comprehensive evaluation framework, advanced metrics, truncation analysis, and production-ready recommendations for optimal token allocation strategies.
