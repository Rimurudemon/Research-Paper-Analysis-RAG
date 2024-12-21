# Research Paper RAG Analysis Model

## Overview
This project implements a Retrieval-Augmented Generation (RAG) model specifically designed for research paper analysis. It leverages state-of-the-art tools and frameworks to provide comprehensive analysis of academic papers, including text, tables, figures, and mathematical equations.

## Technology Stack
### Core Components
- **LangChain**: Orchestrates the RAG pipeline and provides:
  - Document loading and processing
  - Text splitting with academic-aware chunking
  - Vector store integration
  - Prompt management and chain orchestration
  - Memory management for contextual conversations

- **Unstructured**: Powers robust document processing with:
  - PDF parsing with layout preservation
  - Table structure detection
  - Figure extraction
  - Mathematical equation recognition
  - Metadata extraction

- **Google Gemini API**: Provides the large language model capabilities for:
  - Natural language understanding
  - Multi-modal analysis
  - Context-aware responses
  - Technical content comprehension

### Specialized Tools
- **Camelot**: Extracts tables from PDFs while maintaining:
  - Structure preservation
  - Cell relationships
  - Header detection
  - Data type recognition

- **PyMuPDF (fitz)**: Handles document layout analysis:
  - Image extraction
  - Vector graphics processing
  - Page segmentation
  - Document structure analysis

- **matplotlib**: Visualizes extracted data:
  - Figure recreation
  - Data plotting
  - Statistical visualizations

### Vector Store Options
- **Chroma**: Default vector store for:
  - Fast similarity search
  - Metadata filtering
  - Efficient indexing

- **FAISS**: Alternative for large-scale deployments:
  - Scalable similarity search
  - Optimized for high-dimensional vectors
  - Efficient memory usage

## Features

### Enhanced Document Processing
- Automatic structure detection and preservation
- Smart chunking based on document sections
- Reference and citation linking
- Mathematical formula extraction and rendering

### Table Analysis
- Automatic table detection and extraction
- Structure preservation and formatting
- Data type recognition
- Statistical analysis of tabular data
- Conversion to pandas DataFrames for analysis
- Table comparison across papers

### Image and Figure Processing
- Automatic figure extraction
- Caption and reference linking
- Graph and chart data extraction
- Visual element classification
- Multi-modal analysis with Gemini Vision
- Figure-text relationship analysis

### Content Analysis
- Section-wise summarization
- Methodology comparison
- Results analysis
- Literature review synthesis
- Citation network analysis
- Research gap identification

## Prerequisites
- Python 3.8+
- Jupyter Notebook
- Google Cloud account with Gemini API access
- Required Python packages:
```
langchain>=0.1.0
unstructured>=0.10.0
camelot-py>=0.11.0
pymupdf>=1.23.0
chromadb>=0.4.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
python-dotenv>=1.0.0
```

## Quick Start
1. Clone this repository
```bash
git clone https://github.com/Rimurudemon/Research-Paper-Analysis-RAG.git
cd research-rag-analysis
```

2. Install dependencies

3. Set up your Gemini API key
- Create a file named `.env` in the project root
- Add your API key: `GEMINI_API_KEY=your_api_key_here`

4. Launch the Jupyter notebook
```bash
jupyter notebook Research_Paper_Analysis.ipynb
```

## Usage Examples
```python
# Initialize the paper analyzer
from paper_analyzer import PaperAnalyzer

# Load and process a paper
paper = PaperAnalyzer("path/to/your/paper.pdf")

# Extract and analyze tables
tables = paper.extract_tables()
table_analysis = paper.analyze_table_data(tables[0])

# Process figures
figures = paper.extract_figures()
figure_analysis = paper.analyze_figure(figures[0])

# Comprehensive analysis
full_analysis = paper.analyze_paper({
    'include_tables': True,
    'include_figures': True,
    'analyze_citations': True
})
```

## Advanced Configuration
The system can be configured through `config.yaml`:
```yaml
processing:
  chunk_size: 1000
  chunk_overlap: 200
  table_extraction:
    engine: "camelot"
    flavor: "lattice"
  image_extraction:
    min_size: 100
    formats: ["png", "jpg", "svg"]

embedding:
  model: "sentence-transformers/all-mpnet-base-v2"
  dimension: 768

vector_store:
  backend: "chroma"
  similarity_metric: "cosine"
  k_neighbors: 4
```

## Contributing
We welcome contributions! 



## Support
For issues and feature requests, please use the GitHub issue tracker. 
