# Question Answering with DistilBERT

## Overview

This project demonstrates how to use DistilBERT for question answering. Given a URL of a webpage and a question, the model extracts the answer from the webpage's content. This can be useful for tasks like extracting information from articles or documents.

## Usage

To run this code in a Google Colab notebook, click on the "Open In Colab" button below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/your-colab-notebook-link)

## Installation

No specific installation is required as this project utilizes Python libraries such as `requests` for web scraping and `transformers` for using the DistilBERT model.

## How it Works

1. **Webpage Content Retrieval**: The script fetches the content of the provided webpage using the `requests` module and extracts the text using BeautifulSoup.

2. **Question Answering**: The pre-trained DistilBERT model and tokenizer are loaded. The text content and the question are tokenized and fed into the model. The model predicts the start and end positions of the answer span within the text.

3. **Answer Extraction**: The predicted answer span is converted back into text using the tokenizer, and the extracted answer is returned.

## Example

```python
from bs4 import BeautifulSoup
import requests
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast

def get_answer(url, question):
    # Fetch the webpage content
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    text = ' '.join(p.get_text() for p in soup.find_all('p'))

    # Load pre-trained DistilBERT model and tokenizer
    model_name = "distilbert-base-uncased"
    model = DistilBertForQuestionAnswering.from_pretrained(model_name)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    # Tokenize the input
    inputs = tokenizer(text, question, return_tensors="pt", padding=True, truncation=True)

    # Get the answer
    start_scores, end_scores = model(**inputs).start_logits, model(**inputs).end_logits
    start_idx = start_scores.argmax()
    end_idx = end_scores.argmax()
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx+1]))

    # If no answer is generated, return a default response
    if not answer:
        return "I don’t know the answer"

    return answer

# Sample input
input_data = {
    "url": "https://en.wikipedia.org/wiki/Generative_artificial_intelligence",
    "question": "What are the concerns around Generative AI?"
}

# Get the answer
output = get_answer(input_data['url'], input_data['question'])
print({"answer": output})
```

## Results

The script extracts the answer to the provided question from the webpage content. However, it's essential to note that the accuracy of the answer extraction depends on various factors such as the quality of the webpage content and the performance of the pre-trained DistilBERT model.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests for any improvements or bug fixes.

## Acknowledgements

- The `transformers` library for providing the DistilBERT model and tokenizer.
- BeautifulSoup and requests modules for web scraping.

---
