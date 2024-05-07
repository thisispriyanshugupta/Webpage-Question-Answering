# Webpage-Question-Answering

This project provides a Python function for extracting answers from web pages based on user-provided questions. The function utilizes the DistilBERT model for accurate question answering.

### Functionality

The `get_answer` function, located in the `webpage_question_answering.py` script, performs the following steps:

1. **Webpage Content Retrieval**: Utilizes the `requests` library to fetch the content of the specified webpage URL.
2. **Text Extraction**: Parses the HTML content using `BeautifulSoup` and extracts text from paragraph (`<p>`) elements.
3. **Model Initialization**: Loads a pre-trained DistilBERT model (`distilbert-base-uncased`) and its tokenizer for question answering.
4. **Tokenization**: Tokenizes the webpage content and the question using the tokenizer.
5. **Answer Extraction**: Passes the tokenized inputs to the model to obtain start and end scores for potential answer spans. The function selects the answer span with the highest score and converts it back to a string.
6. **Default Response Handling**: If no answer is generated, the function returns a default response: "I don’t know the answer".

### Usage

```python
from webpage_question_answering import get_answer

# Sample input
input_data = {
    "url": "https://en.wikipedia.org/wiki/Generative_artificial_intelligence",
    "question": "What are the concerns around Generative AI?"
}

# Get the answer
output = get_answer(input_data['url'], input_data['question'])
print({"answer": output})
```

### Requirements

- `requests`: Required for fetching webpage content.
- `beautifulsoup4`: Necessary for parsing HTML content.
- `transformers`: Needed for loading the DistilBERT model and tokenizer.

### GitHub Repository

Feel free to explore and utilize the `get_answer` function for extracting answers from web pages using specific questions!

---
