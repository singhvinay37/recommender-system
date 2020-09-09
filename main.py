import numpy as np
import pandas as pd
from flask import Flask, render_template, request
# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds


app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/recommend",methods=['POST'])

def recommend():
    user = request.form['user']
    
    df = pd.read_csv("Dataset/NLP_Product_Reviews_Dataset.csv")
    df2 = df[['id','name','reviews.rating','keys','upc']].copy()
    df2.columns = ['UserId','ProductName','Ratings','keys','ProductCode']
    
    counts1=df2['UserId'].value_counts() 
    counts2=df2['ProductName'].value_counts()
    
    df3=df2[df2['UserId'].isin(counts1[counts1 >=0].index)]
    df3=df3[df3['ProductName'].isin(counts2[counts2 >=0].index)]
    
    
    df3 = df3.drop_duplicates(subset=['ProductName'])
    df3 = df3.sort_values(by='Ratings')
    df3 = df3.reset_index(drop=True)
    
    count_users = df3.groupby("UserId", as_index=False).count()
    count = df3.groupby("ProductName", as_index=False).mean()
    items_df = count[['ProductName']]
    users_df = count_users[['UserId']]
    
    df_clean_matrix = df3.pivot(index='ProductName', columns='UserId', values='Ratings').fillna(0)
    df_clean_matrix = df_clean_matrix.T
    R = (df_clean_matrix).values  
    user_ratings_mean = np.mean(R, axis = 1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)
    U, sigma, Vt = svds(R_demeaned)
    sigma = np.diag(sigma)
    
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns = df_clean_matrix.columns)
    preds_df['UserId'] = users_df
    
    found = df3[df3['UserId'] == user]
    if len(found) > 0:
        preds_df.set_index('UserId', inplace=True)
     
    # Get and sort the user's predictions
        sorted_user_predictions = preds_df.loc[user].sort_values(ascending=False)
    
    # Get the user's data and merge in the item information.
        user_data = df3[df3['UserId'] == user]
        user_full = (user_data.merge(items_df, how = 'left', left_on = 'ProductName', right_on = 'ProductName').
                     sort_values(['Ratings'], ascending=False)
                    )

    
    # Recommend the highest predicted rating items that the user hasn't bought yet.
        recommendations = (items_df[~items_df['ProductName'].isin(user_full['ProductName'])].
               merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'ProductName',
               right_on = 'ProductName').
         rename(columns = {user: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:5, :-1]
                    )
        topk=recommendations.merge(df3,left_index = True, right_on = 'ProductName',left_on='ProductName').drop_duplicates(
               ['ProductName'])[['ProductName']]
    
        products_list = topk['ProductName'].values.tolist()
        
        return render_template('recommend.html',user=user,r=products_list,t=products_list)
  
    else:
        return render_template('recommend.html',user=user,r=user,t='s')


if __name__ == '__main__':
    app.run()
