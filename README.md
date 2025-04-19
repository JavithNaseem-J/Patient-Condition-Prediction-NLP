
# Medical Condition Classifier

## Overview
The **Medical Condition Classifier** is a Natural Language Processing (NLP) project designed to predict a patient's medical condition based on their description and recommend top-rated drugs for the identified condition. The application uses machine learning to classify conditions from text input and leverages a dataset of drug reviews to suggest effective medications. The project includes a Streamlit web interface for user interaction, making it accessible for users to input descriptions and receive predictions.

## Dataset
The project utilizes the **Drugs.com dataset** (`drugsComTrain_raw.csv`), which contains:
- **Columns**: `uniqueID`, `drugName`, `condition`, `review`, `rating`, `date`, `usefulCount`.
- **Size**: 161,297 entries with 884 unique conditions and 3,436 unique drugs.
- **Key Usage**: 
  - The `review`,`condition` column is used to train the condition prediction model.
  - The `drugName`, `rating`, and `usefulCount` columns are used to recommend top drugs for predicted conditions.

## Methodology
1. **Data Preprocessing**:
   - Text cleaning: Converts text to lowercase, removes non-alphabetic characters, and splits into words.
   - Stopword removal: Eliminates common English stopwords using NLTK.
   - Lemmatization: Reduces words to their base form using NLTK's WordNetLemmatizer.
   - Vectorization: Transforms text into numerical features using TF-IDF Vectorizer.

2. **Model Training**:
   - **Algorithm**: Passive Aggressive Classifier (chosen for its efficiency with text data).
   - **Feature Extraction**: TF-IDF Vectorizer converts preprocessed text into sparse matrices.
   - **Training Data**: Split from the Drugs.com dataset, with reviews as input and conditions as labels.

3. **Drug Recommendation**:
   - Filters drugs with ratings ≥9 and useful counts ≥100.
   - Sorts by rating and useful count, then selects the top three unique drugs for the predicted condition.

4. **Web Application**:
   - Built with Streamlit for a user-friendly interface.
   - Users input a condition description, and the app displays the predicted condition and recommended drugs.

## Installation
To run this project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/medical-condition-classifier.git
   cd medical-condition-classifier
   ```

2. **Install Dependencies**:
   Ensure Python 3.8+ is installed, then install required packages:
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` should include:
   ```
   pandas
   numpy
   nltk
   scikit-learn
   streamlit
   joblib
   matplotlib
   seaborn
   wordcloud
   ```

3. **Download NLTK Data**:
   Run the following in a Python shell:
   ```python
   import nltk
   nltk.download('popular')
   ```

4. **Download the Dataset**:
   - Place the `drugsComTrain_raw.csv` file in the `dataset` folder.
   - Alternatively, download it from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018).

5. **Run the Application**:
   Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   Open the provided local URL (e.g., `http://localhost:8501`) in a browser.

## Usage
1. **Launch the App**: Run `streamlit run app.py` and access the web interface.
2. **Input Description**: Enter a patient condition description in the text area (e.g., "I have a headache and fever").
3. **Predict**: Click the "Predict Condition" button to view the predicted condition and top drug recommendations.
4. **View Results**:
   - The predicted condition is displayed prominently.
   - Up to three top-rated drugs are listed.

## Files
- **`app.py`**: Main Streamlit application script for the web interface.
- **`Medical Condition Classifier.ipynb`**: Jupyter Notebook with data exploration, preprocessing, model training, and evaluation.
- **`tfidf_vectorizer.pkl`**: Saved TF-IDF Vectorizer model.
- **`model.pkl`**: Saved Passive Aggressive Classifier model.
- **`dataset/drugsComTrain_raw.csv`**: Dataset file (not included in the repository; must be downloaded).
- **`requirements.txt`**: List of Python dependencies.

## Future Improvements
- **Enhanced Model**: Experiment with advanced NLP models like BERT or LSTM for better accuracy.
- **Broader Dataset**: Incorporate additional datasets to cover more conditions and drugs.
- **Real-Time Updates**: Integrate APIs for up-to-date drug information.
- **User Feedback**: Add a feedback mechanism to refine predictions based on user input.
- **Multilingual Support**: Extend preprocessing to handle non-English descriptions.

![Screenshot 2025-04-19 150148](https://github.com/user-attachments/assets/bf8528ca-80e2-4a6d-8fa3-6a34eab9b83e)

<br>

![image](https://github.com/user-attachments/assets/241727d5-b641-43a4-b74a-c404cfc5d4ba)


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
