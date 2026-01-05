<<<<<<< HEAD
# RAG Mastery: Complete Parameter Experiments

This project provides a comprehensive exploration of RAG (Retrieval-Augmented Generation) systems, from foundational embedding choices to advanced parameter optimization using the Gemini research paper as a knowledge base.
=======
# Parameters Experiments for RAG Systems

This repository contains comprehensive experiments testing various parameters in Retrieval-Augmented Generation (RAG) systems using OpenAI's API and the Gemini research paper as the knowledge base.
>>>>>>> 280b73cdbd24f24be0828750a4a3a12cd7e8dc70

## 📁 Project Structure

```
parameters-experiments/
├── notebooks/
<<<<<<< HEAD
│   ├── 00_embedding_models_comparison.ipynb    # Compare embedding models (NEW)
│   ├── 01_similarity_methods_comparison.ipynb  # Compare similarity search methods (NEW)
│   ├── 02_pdf_embeddings.ipynb                # PDF processing & embeddings setup (RENAMED)
│   ├── 03_temperature.ipynb                   # Temperature parameter testing (RENAMED)
│   ├── 04_top_p_top_k.ipynb                  # Top-p and retrieval-k experiments (RENAMED)
│   └── 05_max_tokens.ipynb                   # Max tokens parameter testing (RENAMED)
├── data/
│   ├── Gemini_FamilyOfMultimodelModels.pdf
│   ├── rag_embeddings.pkl                    # Pre-computed embeddings
│   ├── embedding_models_comparison.pkl       # Embedding comparison results (NEW)
│   └── similarity_methods_comparison.pkl     # Similarity methods results (NEW)
=======
│   ├── 00_pdf_embeddings.ipynb      # PDF processing & embeddings setup
│   ├── 01_temperature.ipynb         # Temperature parameter testing
│   ├── 02_top_p_top_k.ipynb        # Top-p and retrieval-k experiments
│   └── 03_max_tokens.ipynb         # Max tokens parameter testing
├── data/
│   ├── Gemini_FamilyOfMultimodelModels.pdf
│   └── rag_embeddings.pkl          # Pre-computed embeddings
>>>>>>> 280b73cdbd24f24be0828750a4a3a12cd7e8dc70
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
<<<<<<< HEAD
   - **Phase 1 (Foundation)**: `00_embedding_models_comparison.ipynb`, `01_similarity_methods_comparison.ipynb`
   - **Phase 2 (Setup)**: `02_pdf_embeddings.ipynb` to generate embeddings
   - **Phase 3 (Parameters)**: `03_temperature.ipynb`, `04_top_p_top_k.ipynb`, `05_max_tokens.ipynb`

## 🎯 Complete Learning Path

### Phase 1: Foundation (Embedding & Search)
**00 - Embedding Models Comparison** ⭐ NEW
- Compares text-embedding-3-small, 3-large, and ada-002
- Evaluates: accuracy, speed, cost, storage requirements
- Provides cost/performance analysis for production decisions

**01 - Similarity Search Methods** ⭐ NEW
- Tests cosine, euclidean, manhattan, dot product, minkowski
- Analyzes retrieval accuracy and computational speed
- Recommends optimal methods for different use cases

### Phase 2: RAG Pipeline Setup
**02 - PDF Embeddings & RAG Pipeline** (formerly 00)
- Processes the Gemini PDF and creates embeddings
- Sets up the foundation for parameter experiments
- Creates 82 semantic chunks with optimal embedding model

### Phase 3: Generation Parameters
**03 - Temperature Effects** (formerly 01)
- Tests temperature values from 0.0 to 1.2
- Compares Non-RAG vs RAG responses
- Uses LLM judge for evaluation
- Key finding: RAG consistently outperforms non-RAG

**04 - Top-p and Retrieval-k Analysis** (formerly 02)
- Explores nucleus sampling (top-p) from 0.2 to 1.0
- Tests retrieval-k values from 1 to 15 chunks
- Finds optimal balance at k=3-5 chunks
- Shows RAG robustness to parameter variations

**05 - Max Tokens Impact** (formerly 03)
- Tests response length limits from 50 to 800 tokens
- Analyzes information density vs verbosity trade-offs
- Determines optimal token limits for different use cases
=======
   - Start with `00_pdf_embeddings.ipynb` to generate embeddings
   - Then run parameter experiments: `01_temperature.ipynb`, `02_top_p_top_k.ipynb`, `03_max_tokens.ipynb`
>>>>>>> 280b73cdbd24f24be0828750a4a3a12cd7e8dc70

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

<<<<<<< HEAD
### 📋 Planned Experiments

#### Retrieval & Chunking
- [ ] **09_embedding_models.ipynb** - Compare different embedding models (text-embedding-3-small vs large vs ada-002)
- [ ] **10_similarity_thresholds.ipynb** - Test retrieval similarity score thresholds
- [ ] **11_chunk_strategies.ipynb** - Sentence-based vs paragraph-based vs semantic chunking
- [ ] **12_context_window.ipynb** - Optimal context length for different query types

#### Model Parameters
- [ ] **13_model_comparison.ipynb** - GPT-4o-mini vs GPT-4o vs GPT-3.5-turbo
- [ ] **14_frequency_penalty.ipynb** - Test frequency and presence penalties
- [ ] **15_system_prompts.ipynb** - Different system prompt strategies
- [ ] **16_few_shot_examples.ipynb** - Impact of few-shot examples in prompts

#### Advanced Techniques
- [ ] **17_query_expansion.ipynb** - Query rewriting and expansion techniques
- [ ] **18_reranking.ipynb** - Post-retrieval reranking strategies
- [ ] **19_hybrid_search.ipynb** - Combining semantic + keyword search
- [ ] **20_multi_query.ipynb** - Multiple query generation and aggregation

#### Evaluation & Optimization
- [ ] **21_response_quality.ipynb** - Comprehensive quality metrics (BLEU, ROUGE, BERTScore)
- [ ] **22_latency_optimization.ipynb** - Speed vs quality trade-offs
- [ ] **23_cost_efficiency.ipynb** - Cost per query optimization strategies
- [ ] **24_error_analysis.ipynb** - Common failure modes and mitigation

#### Domain-Specific
- [ ] **25_query_types.ipynb** - Performance across different question types (factual, analytical, comparative)
- [ ] **26_document_types.ipynb** - Testing with different document formats (academic papers, manuals, web content)
- [ ] **27_multilingual.ipynb** - Cross-language retrieval and generation

=======
>>>>>>> 280b73cdbd24f24be0828750a4a3a12cd7e8dc70
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

<<<<<<< HEAD
## 📄 License

This project is open source and available under the MIT License.
=======
>>>>>>> 280b73cdbd24f24be0828750a4a3a12cd7e8dc70

## 🤝 Acknowledgments

- Gemini research paper for providing the knowledge base
- OpenAI for API access
- Community contributions and feedback

---

**Next Steps**: Run `git pull` before pushing to sync with remote changes, then continue with the planned experiments!
