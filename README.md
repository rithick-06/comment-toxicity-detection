# Comment Toxicity Detection

This project is focused on detecting toxicity in online comments using a deep learning approach. The model classifies comments into different categories of toxicity, such as moderate, high, threatening, and racist. It employs natural language processing (NLP) techniques, and a hybrid architecture of CNN and LSTM to manage word embeddings and text classification. The model is deployed using a Gradio web application, allowing users to input comments and receive real-time toxicity predictions.

## Key Features:

#### 1. Classification
   * **Moderately Toxic Comments**
   * **Highly Toxic Comments**
   * **Threatening Comments**
   * **Racist Comments**

#### 2. Deep Learning Architecture
   * **CNN:** Extracts features from comments.
   * **LSTM:** Captures word dependencies and context.
   * **Dense Layers:** Performs the final classification into toxicity categories.

#### 3. Real-time Detection
   * **Gradio Interface**: Input comments and instantly receive toxicity classification results.

#### 4. Gen Z Term Adaptation
   * **Incorporates Gen Z Language** to ensure accurate detection of modern-day internet slang and terms, improving model efficiency based on feedback.

## Project Structure
├── data/ │ ├── train.csv # Training dataset │ ├── test.csv # Testing dataset ├── models/ │ ├── toxicity_model.h5 # Trained model in h5 format ├── notebooks/ │ ├── data_preprocessing.ipynb # Data cleaning and preprocessing notebook │ ├── model_training.ipynb # Model architecture and training notebook ├── app/ │ ├── gradio_app.py # Gradio web app for model deployment ├── README.md # Project description and instructions └── requirements.txt # Python dependencies


## Installation

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Gradio 3.x

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/comment-toxicity-detection.git
   cd comment-toxicity-detection
## Usage
Once the Gradio web app is running, users can input any comment, and the model will classify it into the following categories:

   * **Moderately Toxic**
   * **Highly Toxic**
   * **Threatening**
   * **Racist**

The output will display a binary label (0 for non-toxic, 1 for toxic) for each category, helping users identify harmful content quickly and effectively.

## Dataset
The dataset used in this project contains a collection of online comments, each labeled with one or more toxicity categories. The comments undergo preprocessing, including:

   * **Tokenization**: Splitting the text into individual words or tokens.
   * **Embedding**: Converting the tokens into numerical vectors for model input.

The dataset is split into training and testing subsets, ensuring the model can generalize well to unseen data.

## Future Improvements
   * **Multilingual Support**: We plan to extend the model to handle multiple languages, allowing for toxicity detection across diverse linguistic contexts.
   * **Sentiment Analysis**: Incorporating sentiment analysis to gauge the overall emotional tone of comments, providing a more comprehensive content analysis.
   * **Additional Toxicity Categories**: Expanding the model to include more specific types of toxicity, such as misogyny or homophobia, to improve detection granularity.

## Developers
   * [Rithick M K](https://github.com/rithick-06)
   * [Anushya Varshini K](https://github.com/anushya03)
   * [Gokulnath G](https://github.com/GOKULNATH004)

