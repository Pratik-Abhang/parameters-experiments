# RAG Parameters Experiments

This project provides comprehensive experiments testing various parameters in Retrieval-Augmented Generation (RAG) systems using OpenAI's API and the Gemini research paper as the knowledge base.

## 📁 Project Structure

```
parameters-experiments/
├── notebooks/
│   ├── 01_embedding_models_comparison.ipynb    # Compare embedding models
│   ├── 02_similarity_methods_comparison.ipynb  # Compare similarity search methods  
│   ├── 03_temperature.ipynb                   # Temperature parameter testing (UPDATED)
│   ├── 04_top_p_top_k.ipynb                  # Top-p and retrieval-k experiments
│   └── 05_max_tokens.ipynb                   # Max tokens parameter testing
├── data/
│   ├── Gemini_FamilyOfMultimodelModels.pdf
│   ├── rag_embeddings.pkl                    # Pre-computed embeddings
│   ├── temperature_rag_results.csv           # Temperature experiment results (NEW)
│   └── temperature_rag_detailed.pkl          # Detailed temperature results (NEW)
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
   - **Foundation**: `01_embedding_models_comparison.ipynb`, `02_similarity_methods_comparison.ipynb`
   - **Parameters**: `03_temperature.ipynb`, `04_top_p_top_k.ipynb`, `05_max_tokens.ipynb`

## 🎯 Experiment Overview

### Phase 1: Foundation (Embedding & Search)
**01 - Embedding Models Comparison**
- Compares text-embedding-3-small, 3-large, and ada-002
- Evaluates: accuracy, speed, cost, storage requirements
- Provides cost/performance analysis for production decisions

**02 - Similarity Search Methods**
- Tests cosine, euclidean, manhattan, dot product similarities
- Analyzes retrieval accuracy and computational speed
- Recommends optimal methods for different use cases

### Phase 2: Generation Parameters
**03 - Temperature Effects** ⭐ UPDATED
- Tests temperature values: 0.0, 0.2, 0.5, 0.8, 1.2, 1.5, 2.0
- **RAG-only experiments** (removed non-RAG comparison)
- **Enhanced evaluation**: 6 metrics including creativity, diversity, consistency
- **Comprehensive analysis**: Statistical summaries, visualizations, recommendations
- Uses LLM judge with temperature-specific criteria

**04 - Top-p and Retrieval-k Analysis**
- Explores nucleus sampling (top-p) from 0.2 to 1.0
- Tests retrieval-k values from 1 to 15 chunks
- Finds optimal balance at k=3-5 chunks

**05 - Max Tokens Impact**
- Tests response length limits from 50 to 800 tokens
- Analyzes information density vs verbosity trade-offs

## 📊 Current Experiments Status

### ✅ Completed

| Notebook | Parameter | Values Tested | Key Metrics |
|----------|-----------|---------------|-------------|
| 01 | Embedding Models | 3-small, 3-large, ada-002 | Accuracy, Speed, Cost |
| 02 | Similarity Methods | Cosine, Euclidean, Manhattan, Dot Product | Retrieval Accuracy, Speed |
| 03 | Temperature | 0.0, 0.2, 0.5, 0.8, 1.2, 1.5, 2.0 | Accuracy, Creativity, Diversity, Consistency |
| 04 | Top-p & Retrieval-k | Top-p: 0.2-1.0, k: 1-15 | Quality, Completeness |
| 05 | Max Tokens | 50, 100, 200, 400, 800 | Token Utilization, Completeness |

### 🔄 Planned Experiments

#### Advanced Parameters
- [ ] **06_frequency_penalty.ipynb** - Frequency and presence penalties
- [ ] **07_model_comparison.ipynb** - GPT-4o-mini vs GPT-4o vs GPT-3.5-turbo
- [ ] **08_system_prompts.ipynb** - Different system prompt strategies

#### Retrieval Optimization
- [ ] **09_chunk_size.ipynb** - Different text chunk sizes (250-1500 words)
- [ ] **10_chunk_overlap.ipynb** - Overlap percentage experiments (0%-50%)
- [ ] **11_similarity_thresholds.ipynb** - Retrieval similarity score thresholds

#### Evaluation & Analysis
- [ ] **12_cost_analysis.ipynb** - Token usage and cost optimization
- [ ] **13_evaluation_methods.ipynb** - LLM judge vs automated metrics
- [ ] **14_error_analysis.ipynb** - Failure modes and mitigation

## 🎯 Key Research Questions

1. **Temperature Optimization**: What temperature values work best for different use cases?
2. **Parameter Interactions**: How do different parameters interact with each other?
3. **Cost vs Quality**: Optimal balance between response quality and API costs?
4. **Evaluation Reliability**: Which evaluation methods most accurately reflect performance?

## 📈 Enhanced Metrics (Temperature Notebook)

### Quality Metrics
- **Accuracy**: Factual correctness based on context
- **Completeness**: Addresses all parts of the question  
- **Clarity**: Well-structured and understandable

### Temperature-Specific Metrics
- **Creativity**: Novel insights or creative explanations
- **Diversity**: Varied vocabulary and expression
- **Consistency**: Logical flow and coherence

### Performance Metrics
- **Latency**: Response generation time
- **Token Usage**: Total tokens consumed
- **Lexical Diversity**: Unique words / total words ratio

## 🔧 Tools & Technologies

- **Language Model**: OpenAI GPT-4o-mini
- **Embeddings**: OpenAI text-embedding-3-small (optimized choice)
- **Document Processing**: PyMuPDF
- **Data Analysis**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Evaluation**: Enhanced LLM judge with temperature-specific criteria

## 📊 Temperature Experiment Highlights

The updated temperature notebook provides:
- **7 temperature values** tested across 8 queries (56 total experiments)
- **Statistical analysis** with mean/std for all metrics
- **Multiple visualizations**: Box plots, line plots, correlation heatmaps
- **Best performer identification** for each metric
- **Use case recommendations** based on results
- **Production-ready insights** for different scenarios

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch for new experiments
3. Follow the notebook naming convention: `XX_experiment_name.ipynb`
4. Include comprehensive analysis and visualizations
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

---

**Latest Update**: Temperature notebook enhanced with comprehensive analysis, temperature-specific metrics, and production recommendations.
