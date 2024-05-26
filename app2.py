from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__, template_folder='templates')

# Load the dataset
data = pd.read_csv("uni_data2.csv")

# Preprocess data
X = pd.get_dummies(data.drop(['Name'], axis=1))
y = data['Name']

# Train kNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_data = request.json

    user_choice = user_data['test']
    X_user = select_columns(user_data, user_choice)
    X_user = pd.get_dummies(X_user)

    # Calculate cosine similarity
    similarities = cosine_similarity(X_user, X)

    # Find top N most similar universities using cosine similarity
    cosine_top_indices = similarities.argsort()[:, ::-1][:, :5]

    # Refine recommendations using kNN
    knn_top_indices = knn_model.kneighbors(X_user, return_distance=False)

    # Combine recommendations from both methods
    combined_indices = set(cosine_top_indices.flatten()).union(set(knn_top_indices.flatten()))

    # Get the names of top N universities
    top_universities = y.iloc[list(combined_indices)].tolist()

    # Render the recommendation.html template with the recommended universities
    return render_template('recommendation.html', universities=top_universities)

def select_columns(data, choice):
    if choice == 'IELTS':
        return data.drop(['Name', 'Location', 'Cost of Living', 'Intakes',
                          'Application Fees', 'Post Graduate Tuition Fees',
                          'Post Graduate GPA/Percentage', 'Post Graduate PTE', 'Post Graduate Gap Acceptance', 'Agent',
                          'Sample Undergraduate Courses', 'Sample Postgraduate Courses', 'PTE marks required'], axis=1)
    elif choice == 'PTE':
        return data.drop(['Name', 'Location', 'Cost of Living', 'Intakes',
                          'Application Fees', 'Post Graduate Tuition Fees',
                          'Post Graduate GPA/Percentage', 'Post Graduate IELTS',
                          'Post Graduate Gap Acceptance', 'Agent',
                          'Sample Undergraduate Courses', 'Sample Postgraduate Courses', 'IELTS marks required'], axis=1)
    else:
        raise ValueError("Invalid choice. Choose either 'IELTS' or 'PTE'.")

if __name__ == '__main__':
    app.run(debug=True)
