# Restaurant Recommendation System

A modern, multi-model restaurant recommendation platform powered by machine learning, vector search, and LLMs. This project integrates data science notebooks, a Streamlit web app, and REST API endpoints to deliver intelligent, context-aware restaurant suggestions.

---

## Key Components

- **Notebook Folder:** Contains Jupyter notebooks for EDA, feature engineering, model training, evaluation, and LLM-powered recommendations.
- **Streamlit App:** Interactive web interface for exploring and querying restaurant recommendations.
- **Models Folder:** Stores trained machine learning models for similarity, clustering, and retrieval.
- **Data Folder:** Contains cleaned and processed restaurant datasets.
- **LLM Integration:** Uses Groq’s Llama-3 model via LangChain for advanced, natural language recommendations.
- **Docker Setup:** Containerizes the environment for reproducibility and easy deployment.

---

## Workflow

1. **Data Preparation:**  
   - Scrape and clean restaurant data.
   - Perform EDA and feature engineering in Jupyter notebooks.

2. **Model Training:**  
   - Train Cosine Similarity, KNN, and KMeans models.
   - Save models for fast inference.

3. **Evaluation:**  
   - Assess models using precision, recall, NDCG, coverage, and diversity.
   - Visualize and compare results.

4. **LLM-Powered Recommendations:**  
   - Use LangChain and Groq LLM to generate context-aware suggestions.
   - Integrate semantic search with FAISS vector store.

5. **Web & API Interface:**  
   - Query recommendations via Streamlit app or FastAPI endpoints.
   - View results interactively.

---

## Technologies Used

- **Python:** Data processing, modeling, and API development.
- **Jupyter Notebook:** Data analysis and experimentation.
- **Streamlit:** Web application for user interaction.
- **FastAPI:** RESTful API for backend services.
- **LangChain & Groq:** LLM-powered recommendation engine.
- **FAISS:** Efficient vector search for semantic retrieval.
- **Docker:** Containerization for reproducible environments.


## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AArafatt/Restaurant_Recommendation_System.git
cd Restaurant_Recommendation_System
```

### 2. Set Up Python Environment

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
# Or: source venv/bin/activate   # On Linux/Mac
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Docker Setup (Optional)

Ensure Docker Desktop is running, then:

```bash
docker-compose up --build
```

Access Jupyter at [http://localhost:8888](http://localhost:8888)  
Access Streamlit at [http://localhost:8501](http://localhost:8501)

---

## Running the Streamlit App

```bash
streamlit run streamlit/main.py
```

Or via Docker:

```bash
docker-compose up streamlit
```

---

## API Usage

The FastAPI backend exposes endpoints for recommendations.  
Example request:

```bash
curl -X POST "http://localhost:8000/recommendations" -H "Content-Type: application/json" -d '{"query": "Find North Indian restaurants in Banashankari"}'
```

---

## Notebooks Overview

- **01_eda.ipynb:** Exploratory data analysis and visualization.
- **02_feature_engineering.ipynb:** Data cleaning and feature creation.
- **03_model_training.ipynb:** Training similarity, KNN, and clustering models.
- **04_evaluation.ipynb:** Model evaluation and comparison.
- **05_LLM.ipynb:** LLM-powered recommendation system using LangChain and Groq.

---

## Streamlit, API, and LLM Integration

- **Streamlit App:**  
  User-friendly interface for querying and visualizing recommendations.

- **FastAPI Backend:**  
  RESTful API for serving recommendations, powered by semantic search and LLM.

- **LLM (Groq Llama-3):**  
  Generates personalized, context-aware restaurant suggestions using advanced natural language processing.

---

## Testing

Run basic setup tests with pytest:

```bash
pytest test_setup.py
```

---

## Admin & Data Management

- Access admin panel at [http://localhost:8000/admin](http://localhost:8000/admin) (if enabled).
- Manage restaurant data, models, and recommendations.

---

## Notes

- **GitHub Repository:** All updates should be pushed to the public repo.
- **Documentation:** Add further instructions or insights as needed.
- **Memory Limit:** Adjust Docker memory settings if processing large datasets or models.

---

## Customization

- To process more restaurants, adjust the relevant code in the notebooks or CLI commands.
- Update `.env` for different API keys or model configurations.

---