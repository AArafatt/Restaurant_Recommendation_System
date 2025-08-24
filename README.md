# Restaurant Recommendation System

This project is a comprehensive restaurant recommendation system built using machine learning, natural language processing, and modern web technologies. It leverages clustering, similarity search, and large language models (LLMs) to provide personalized restaurant recommendations based on user preferences.

## Project Structure

### Notebooks (`notebook` folder)
The notebooks guide you through the entire workflow, from data exploration to deploying an LLM-powered recommendation API:

**01_eda.ipynb.ipynb**
- Exploratory Data Analysis (EDA): Loads and visualizes the restaurant dataset. Examines distributions, missing values, and key features such as ratings, cost, votes, cuisines, and locations. Helps understand the data and guides feature selection for modeling.

**02_feature_engineering.ipynb**
- Feature Engineering: Cleans and preprocesses the raw data. Creates new features (e.g., standardized ratings, encoded categorical variables). Prepares the dataset for machine learning models by transforming and scaling features.

**03_model_training.ipynb**
- Model Training: Trains multiple recommendation models—Cosine Similarity for feature-based recommendations, K-Nearest Neighbors (KNN) for similarity search, and KMeans for clustering restaurants into groups. Saves trained models for later use. Includes code for evaluating model performance and saving results.

**04_evaluation.ipynb**
- Model Evaluation: Loads saved models and the processed dataset. Analyzes recommendation quality using metrics like precision, recall, F1-score, NDCG, coverage, and diversity. Benchmarks model performance and visualizes results. Compares models and determines the best approach for restaurant recommendations. Saves evaluation results for reporting.

**05_LLM.ipynb**
- LLM-Powered Recommendations: Integrates Large Language Models (LLMs) using LangChain and Groq APIs. Loads restaurant data and converts it to LangChain Documents. Uses semantic chunking and FAISS vector store for efficient retrieval. Builds a Retrieval-Augmented Generation (RAG) system to answer user queries with context-aware recommendations. Provides code for querying the system and getting personalized restaurant suggestions.

**requirements.txt**
- Dependencies: Lists all Python packages required to run the notebooks, including machine learning, data processing, and LLM integration libraries.

### Models (`models` folder)
Pre-trained models and evaluation results used by the system:

- **cosine_similarity.pkl**: Cosine similarity model for comparing restaurant features.
- **kmeans.pkl**: KMeans clustering model for grouping similar restaurants.
- **knn.pkl**: K-Nearest Neighbors model for similarity-based recommendations.
- **model_evaluation_results.pkl**: Saved results and metrics from model evaluation.

### Streamlit App (`streamlit` folder)
Interactive web interface for users to get restaurant recommendations:

- **app.py**: Main Streamlit application for the recommendation system UI.
- **main.py**: Additional logic or API integration for the Streamlit app.
- **faiss_index.pkl**: FAISS vector index used for fast similarity search in the app.
- **streamlit_UI.png**: Screenshot of the Streamlit user interface.
- **__pycache__/**: Python cache files (can be ignored).


## Streamlit App, API, and LLM Integration

This project includes an interactive Streamlit web application and a FastAPI-based backend for serving restaurant recommendations.

### Streamlit App
- Provides a user-friendly interface for entering queries and viewing restaurant recommendations.
- Connects to the backend API to fetch recommendations based on user input.
- Visualizes results and allows users to explore similar restaurants interactively.

### FastAPI Backend (`streamlit/main.py`)
- Implements a RESTful API endpoint (`/recommendations`) for generating restaurant recommendations.
- On startup, loads and processes restaurant data from CSV, splits documents semantically, and builds or loads a FAISS vector index for fast similarity search.
- Uses LangChain and Groq’s LLM to answer user queries with context-aware recommendations.
- The API accepts a query string and returns a list of recommended restaurants, considering features like cuisine, type, location, rating, cost, online ordering, and table booking.

### LLM Integration
- Integrates Groq’s Llama-3.3-70b-versatile model via LangChain for advanced natural language understanding and generation.
- The LLM is prompted with user queries and relevant restaurant context, enabling it to generate personalized, high-quality recommendations.
- The system uses Retrieval-Augmented Generation (RAG): it retrieves relevant restaurant data using semantic search, then passes this context to the LLM for response generation.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AArafatt/Restaurant_Recommendation_System.git
   cd Restaurant_Recommendation_System
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Jupyter Notebooks:**
   ```bash
   jupyter notebook
   ```
   Open and run the notebooks in the `notebook` folder to explore data and models.
4. **Run the Streamlit App:**
   ```bash
   cd streamlit
   streamlit run app.py
   ```

## Docker Setup

You can also run the entire environment in Docker:

1. **Build and start the container:**
   ```bash
   docker-compose build
   docker-compose up
   ```
2. **Access Jupyter Notebook:**
   Visit [http://localhost:8888](http://localhost:8888) in your browser.

