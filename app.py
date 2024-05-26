from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
app = Flask(__name__, template_folder='templates')

# Load the dataset
data = pd.read_csv("uni_data2.csv")

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if request.headers['Content-Type'] == 'application/json':
        user_data = request.json
    else:
        user_data = request.form.to_dict()
    # user_data = request.json
    print("Received user data:", user_data)
    user_choice = user_data['test']
    X = select_columns(data, user_choice)
    y = data['Name']

    # Convert categorical variables into numerical representations
    X = pd.get_dummies(X)

    new_data = {
        'Gap?': 'Yes' if 'gap' in user_data and user_data['gap'].lower() == 'yes' else 'No',
        'Location': user_data['city'],
        'Under Graduate GPA/Percentage': float(user_data['gpa']),
        'Under Graduate Tuition Fees': 40000,  # Default value, adjust as needed
        'IELTS marks required': 6.5 if user_data['test'] == 'IELTS' else 0,
        'Sample Undergraduate Courses': user_data['course'],
        'Intakes': user_data['intake']
    }

    if user_data['rankings'] == 'yes':
        new_data['Ranking'] = 1  # User cares about rankings
    else:
        new_data['Ranking'] = 0  # User does not care about rankings

    new_data_df = pd.DataFrame([new_data])
    data_location_filtered = data[(data['Location'] == new_data['Location']) & 
                                  (data['Sample Undergraduate Courses'].str.contains(new_data['Sample Undergraduate Courses'])) &
                                  (data['Intakes'] == new_data['Intakes'])]
    print(data_location_filtered)
    if data_location_filtered.empty:
        return render_template('no_university_found.html')

    new_data_df = pd.get_dummies(new_data_df)
    new_data_df = new_data_df.reindex(columns=X.columns, fill_value=0)

    similarities = cosine_similarity(new_data_df, X.loc[data_location_filtered.index])
    rankings = data_location_filtered['Ranking'].values
    
    

    # Normalize the rankings
    normalized_rankings = (rankings - rankings.min()) / (rankings.max() - rankings.min())

    # Combine content-based and collaborative filtering scores
    hybrid_scores = 0.5 * similarities.flatten() + 0.5 * normalized_rankings

    # Get indices of top N most similar universities
    N = 5  # Number of top universities to recommend
    top_indices = hybrid_scores.argsort()[-N:][::-1]

    # # Get the names of top N universities
    # top_universities = y.iloc[data_location_filtered.index[top_indices]]
    # print(top_universities)
    # # Render the recommendation.html template with the recommended universities
    # # Redirect to the recommendation page with the recommended universities
    # # return redirect(url_for('recommendation', universities=top_universities))
    # return render_template('recommendation.html', universities=top_universities)

    top_universities = list(y.iloc[data_location_filtered.index[top_indices]])
    df_top_universities = pd.DataFrame(top_universities, columns=['Name'])
    merged_data = pd.merge(df_top_universities, data, on='Name', how='left')
    universities = merged_data.to_dict(orient='records')
    print(universities)
    # Render the recommendation.html template with the recommended universities
    # Redirect to the recommendation page with the recommended universities
    # return redirect(url_for('recommendation', universities=top_universities))
    return render_template('recommendation.html', universities=universities)

@app.route('/recommendation')
def recommendation():
    universities = request.args.getlist('universities')
    print("Am i here?")
    return render_template('recommendation.html', universities=universities)

if __name__ == '__main__':
    app.run(debug=True)