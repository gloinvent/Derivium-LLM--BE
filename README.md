# RAG Streamlit Application

This is a basic Retrieval Augmented Generation (RAG) application built with Streamlit, designed to chat with PDF documents using Google's Gemini model.

## Setup Instructions

1.  **Navigate to the project directory:**
    ```bash
    cd rag_streamlit_app
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set up your Google API Key:**
    Create a `.env` file in the `rag_streamlit_app` directory and add your Google API Key:
    ```
    GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
    ```
    Replace `"YOUR_GEMINI_API_KEY_HERE"` with your actual Google Gemini API key. You can obtain one from [Google AI Studio](https://aistudio.google.com/app/apikey).

6.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

    This will open the application in your web browser.

## How to Use

1.  **Upload PDF Files:** On the sidebar, click "Browse files" to upload one or more PDF documents.
2.  **Process Documents:** Click the "Process" button after uploading your PDFs. This will extract text, split it into chunks, and create a vector store.
3.  **Ask Questions:** Once processing is complete, type your questions related to the uploaded PDFs in the text input field and press Enter. The application will use the RAG model to provide answers based on the content of your documents.