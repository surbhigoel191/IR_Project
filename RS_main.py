import numpy as np
import pandas as pd
from surprise import KNNWithMeans
from surprise import Dataset
from surprise.model_selection import cross_validate
from sklearn.model_selection import KFold

movies = pd.read_csv("movies.csv",encoding="Latin1")
Ratings = pd.read_csv("ratings.csv")

        
genre_arr=np.zeros((len(usersu),len(genres_dict.keys())))
for i in range(0,len(tags)):
    genre_arr[tags.iloc[i][0]-1][genres_dict[tags.iloc[i][2]]]=1

# stores mean of ratings given by each user
Mean = Ratings.groupby(by="userId",as_index=False)['rating'].mean()
Rating_avg = pd.merge(Ratings,Mean,on='userId')
Rating_avg['adg_rating']=Rating_avg['rating_x']-Rating_avg['rating_y']
usersu = Ratings['userId'].unique()
titles = movies['movieId']
data = pd.merge(movies, Ratings, on="movieId")
n=len(usersu)
m=len(movies)
array = np.zeros((n,m))

#utility matrix (0 if no rating given)
for i in range(0,len(Ratings)):
    array[int(Ratings.iloc[i][0])-1][int(titles[titles==Ratings.iloc[i][1]].index[0])]=Ratings.iloc[i][2]

#pearson corr to find nbrhood of 30
array2=pd.crosstab(data["title"], data["userId"], values=data["rating"], aggfunc="sum")
# Resnick's prediction formula
def find_missing_rating(array,uid,movindx,nbrhood,Mean):
    ans = Mean.iloc[uid-1][1]
    num=0
    den=0
    for i in range(0,10):
        num=num+nbrhood.iloc[i][1]*(array[int(nbrhood.iloc[i][0])-1][movindx]-Mean.iloc[int(nbrhood.iloc[i][0])-1][1])
    for i in range(0,10):
        den=den+(nbrhood.iloc[i][1])
    ans=ans+(num/den)
    return ans

def main_fun(userInput):    
    # finds similarity using pearson method
    similarity = array2.corrwith(array2[userInput], method="pearson")
    correlatedMovies = pd.DataFrame(similarity, columns=["correlation"])
    correlatedMovies = correlatedMovies.sort_values("correlation",ascending=False)
    correlatedMovies.reset_index(inplace=True)
    correlatedMovies=correlatedMovies[correlatedMovies.userId!=userInput]
    
    # stores top 10 most similar users
    nbrhood = correlatedMovies.iloc[0:10]
    
    # stores all the watched movies with ratings given by the user
    rated_df = pd.DataFrame(columns = ['MovieIndx', 'Rating'])
    
    # stores all the unwatched movies with predicted ratings
    unrated_df = pd.DataFrame(columns = ['MovieIndx', 'Rating'])
    for i in range(0,len(movies)):
        if array[userInput-1][i]==0 :
            unrated_df=unrated_df.append({'MovieIndx':i,'Rating':find_missing_rating(array,userInput,i,nbrhood,Mean)},ignore_index = True)
        else:
            rated_df=rated_df.append({'MovieIndx':i,'Rating':array[userInput-1][i]},ignore_index = True)
    unrated_df=unrated_df.sort_values('Rating',ascending=False)
    unrated_df=unrated_df.head()
    rated_df=rated_df.sort_values('Rating',ascending=False)
    rated_df=rated_df.head()
    
    # stores top 5 movies predicted
    finalu=[]
    
    # stores top 5 movies watched already
    finalr=[]
    for i in range(0,5):
        finalu.append(movies.iloc[int(unrated_df.iloc[i][0])][1])
        finalr.append(movies.iloc[int(rated_df.iloc[i][0])][1])
    table = {'Test User Id' : userInput, ('Predicted movies', 'Movies'): finalu, ('Predicted movies', 'Ratings'): unrated_df['Rating'].tolist(), ('Movies seen in past', 'Movies'): finalr, ('Movies seen in past', 'Ratings'): rated_df['Rating'].tolist()}
    return table

# loading dataset
data = Dataset.load_builtin('ml-1m')

algo = KNNWithMeans(k=10)

# Run 5-fold cross-validation and print results.
cross_validate(algo, data, measures=['MAE'], cv=5, verbose=True)
    
