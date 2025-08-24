import streamlit as st
import requests

st.title("🍽️ Restaurant Recommendation System")
st.write("Enter your preferences to get personalized restaurant recommendations.")

query = st.text_input("Enter your query (e.g., 'Best Italian restaurants with online order')")

if st.button("🔍 Get Recommendations"):
    if not query.strip():
        st.error("Please enter a query.")
    else:
        try:
            with st.spinner("Fetching recommendations..."):
                response = requests.post(
                    "http://127.0.0.1:8000/recommendations",
                    json={"query": query},
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                recommendations = data.get("recommendations", "No recommendations found.")
                st.success("✅ Recommendations:")
                st.markdown(recommendations)
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error connecting to backend: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")