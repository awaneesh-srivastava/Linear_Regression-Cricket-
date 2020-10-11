#Awaneesh Srivastava 
"""MATPLOTLIB WILL NOT WORK HERE!
SO YOU CAN COPY THIS CODE TO AN IDE IN ORDER TO SEE THE REAL OUTPUT """

"""(IDE should have the following modules installed:
numpy,pandas,sklearn,matplotlib)"""

import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
"""
DETAILED ANALYSIS OF A PLAYER'S LIFETIME SCORES
 IT'S NOT REAL! :D
 IT'S RANDOMLY GENERATED! IN ORDER TO INCREASE THE MODEL SCORE I HAVE CONCENTRATED MOST OF THE SCORES AND BALLS BETWEEN SOME RANGES! THE MODEL SCORE IS STILL NOT VERY GOOD BECAUSE OF SO MANY VALUES (300) AND THE BALLS ARE DEPENDENT ON THE RUNS SO THEIR SCORE IS MUCH LESS THEN THE RUNS, BECAUSE THEY ARE CAUSING A LOT OF VARIATION! for example: it can be possible that number of balls in 75 runs can be equal to the number of balls in 100 runs!
 I HAVE ALSO GIVEN A MINOR IMPROVEMENT IN THE PERFORMANCE OF THE PLAYER WITH EVERY MATCH!
 HERE'S THE CODE FOR THE RANDOM DATASET GENERATOR: (you can copy it if you want some dataset to work on)

from random import randint as r
print("S.No.,score,balls,year,MOTM,result,totalWon,totalMOTM")
motm = [1,0,0,0,0]
l = len(motm)-1
won,mm,yr = 0,0,2005
res = [1,1,1,1,1,0,0,0]
s = [i for i in range(60,100)]
s.extend([i for i in range(70,90) for j in range(2)])
s.extend([i for i in range(75,85) for j in range(3)])
s.extend([125,30,60,110])
for i in range(0,300):
    score,result = s[r(0,len(s)-1)]+i//10,res[r(0,len(res)-1)]
    m = motm[r(0,l)] if result == 1 else 0
    if m == 1: mm+=1
    if result == 1: won+=1
    if i%20 == 0: yr += 1
    # BALLS ARE GIVING A VARIATION OF 25 WITH RESPECT TO THE RUNS! 10-(-15) = 25
    print(i,i,score,score+r(-15,10),yr,m,result,won,mm,sep=',')
"""
data = {
    "S.No.":[i for i in range(300)],
    "score":[68,85,80,90,77,76,72,83,76,84,71,95,85,63,82,84,78,74,82,89,91,81,83,83,81,95,86,90,86,94,92,84,79,63,91,83,82,89,89,65,83,83,72,88,76,80,86,87,86,120,101,81,80,86,79,84,80,94,87,92,82,82,102,73,77,93,89,78,95,36,100,86,90,96,91,86,68,95,86,79,79,71,82,83,75,133,88,81,82,96,88,99,108,83,95,69,95,97,119,85,88,82,85,87,84,98,83,87,87,88,81,74,99,94,86,93,93,95,98,87,96,108,91,90,82,108,88,90,94,93,92,104,102,78,97,92,83,87,91,100,90,85,86,95,93,83,81,89,77,104,93,93,99,97,98,90,99,96,98,94,92,95,103,99,99,99,95,95,101,103,88,106,103,94,91,106,100,96,100,99,99,93,102,106,98,103,97,94,98,96,95,102,100,96,83,109,112,99,118,79,95,97,97,93,103,108,109,100,93,105,102,97,97,98,106,81,104,98,97,107,101,98,98,107,110,120,95,83,103,97,96,111,104,111,100,98,98,98,112,98,93,98,108,122,107,108,102,115,112,108,105,104,107,135,115,112,100,108,106,99,109,103,106,102,107,99,102,151,102,100,119,108,107,98,115,109,125,111,115,109,108,108,111,104,138,113,105,107,113,106,110,113,114,115,123,110,105,101,106,112],
    "balls":[76,75,70,93,70,79,69,74,80,76,58,82,77,68,72,73,82,60,77,95,100,85,89,87,80,100,74,80,96,102,83,70,77,66,79,93,74,85,82,54,88,90,71,93,84,68,74,87,71,100,99,88,90,78,80,79,74,103,95,99,90,79,98,63,79,78,76,83,88,35,89,80,85,94,96,82,70,84,78,69,68,79,76,68,80,136,98,72,86,104,89,99,104,83,87,75,94,90,115,76,75,77,93,73,75,92,93,85,77,87,91,84,107,82,93,89,80,94,96,78,81,94,82,95,89,99,89,83,96,100,87,110,106,66,93,98,80,87,85,106,96,79,91,86,84,75,89,94,86,113,81,101,102,106,83,82,105,91,99,90,86,104,92,95,92,107,93,82,93,90,96,110,110,94,84,113,90,89,100,102,93,90,93,94,89,104,98,93,96,102,97,110,106,106,101,119,114,106,118,81,80,85,95,84,103,99,97,98,87,107,107,98,101,94,102,69,97,98,87,102,87,94,87,108,118,122,81,85,105,92,86,104,95,97,99,94,95,97,97,97,78,97,93,121,104,116,109,115,112,114,93,93,112,125,120,121,101,95,103,87,97,110,71,101,113,102,108,148,101,91,105,97,112,91,125,100,110,113,102,97,105,95,110,108,145,99,95,118,102,115,109,117,102,121,109,110,102,95,109,108],
    "year":[j for i in range(20) for j in range(2006,2021)],
    "MOTM":[0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    "result":[1,0,1,1,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0,1,1,1,1,0,1,0,1,1,0,1,1,1,0,0,0,0,0,0,0,0,1,0,1,1,1,1,0,0,1,1,1,1,0,1,0,0,1,0,1,1,0,1,1,1,1,0,1,1,1,1,1,1,0,1,0,0,1,1,0,1,1,0,1,0,1,0,0,0,0,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,0,0,1,0,1,1,1,1,1,1,0,0,1,1,1,0,0,1,1,0,1,0,0,0,1,0,1,1,0,1,1,1,1,0,1,0,1,0,0,1,1,0,0,1,1,0,1,0,1,1,1,1,1,1,1,1,1,0,0,1,0,1,0,1,1,0,0,1,0,0,1,1,1,1,1,0,1,1,1,0,0,1,0,1,1,0,1,0,1,1,0,1,1,1,0,1,0,1,0,1,0,1,0,1,1,0,0,1,1,0,0,0,1,1,1,1,0,0,1,1,1,0,0,0,0,1,0,1,1,1,1,0,1,1,1,0,0,1,0,1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,0,1,1,1,1,1,0,1,1,1,0,0,1,1,1,1,1,0,1,0,0,1,1,1,1,1,1,1,0,1,0,1,1,1]
}

# TELLING THE BASIC INFORMATION
d = pd.DataFrame(data,index=[i for i in range(300)])
print(d,"\n\n")
print("Total Matches Played: 300")
print("Lifetime runs:",d["score"].sum())
print("Lifetime balls:",d["balls"].sum())
print("Total lifetime wins:",d["result"].sum())
print("Lifetime Man Of the Match:",d["MOTM"].sum(),"\n\n")
model = LinearRegression()

# DECIDING WHAT TO PREDICT! i.e. Y1
X1 = d[["S.No."]]
Y1 = d[["score"]]

# FITTING AND TRAINING THE MODEL
X_train1,X_test1,Y_train1,Y_test1 = train_test_split(X1,Y1,test_size=0.3,random_state=1)
model.fit(X_train1,Y_train1)
# FINDING THE y-intercept and coefficient
b1 = model.intercept_
m1 = model.coef_
# LIST OF ALL THE MATCHES I WANT TO PREDICT
to_predict1 = np.array([i for i in range(0,450,5)]).reshape(-1,1)
y_test_predicted1 = model.predict(to_predict1)
# FINAL PART! PRINTING THE PREDICTED ANSWER
"""
n = model.predict([[500]])
IS SAME AS
n = b1 + m1*500
BUT THE UPPER METHOD IS BETTER IN ORDER TO GET THE EXACT VALUE!
BECAUSE IN BETWEEN APPLYING THE FORMULA THE VALUES GET ROUND OF TO SOME DECIMAL PLACES i.e. 5
YOU CAN PRINT BOTH THE VALUES AND YOU WILL SEE THAT THERE'S A MINOR DIFFERENCE
"""
print("           SCORE")
print("1st Match: {0}    500th Match: {1}".format(int(y_test_predicted1[0][0]),int(round((y_test_predicted1[-1][0]+m1*55)[0][0]))))
print("Mean Squared Error:",round(mean_squared_error(y_test_predicted1,Y_test1),2))
print("Model Score:  {0:.2f}%{1}".format(model.score(X_test1,Y_test1)*100,"\n\n"))

# APPLYING THE SAME THING WITH THE BALLS
X2 = d[["S.No."]]
Y2 = d[["balls"]]
X_train2,X_test2,Y_train2,Y_test2 = train_test_split(X2,Y2,test_size=0.3,random_state=1)
model.fit(X_train2,Y_train2)
b2 = model.intercept_
m2 = model.coef_
to_predict2 = np.array([i for i in range(0,450,5)]).reshape(-1,1)
y_test_predicted2 = model.predict(to_predict2)
print("           BALLS")
print("1st Match: {0}    500th Match: {1}".format(int(y_test_predicted2[0][0]),int(round((y_test_predicted2[-1][0]+m2*55)[0][0]))))
print("Mean squared error:",round(mean_squared_error(y_test_predicted2,Y_test2),2))
print("Model score:  {:.2f}%".format(model.score(X_test2,Y_test2)*100))

"""
 PLOTTING STARTS HERE!
 OUTPUT:
PLOTS THE SCORE OF EACH MATCH IN THE FORM OF SMALL DOTS
PLOTS THE BALLS PLAYED OF EACH MATCH IN THE FORM OF SMALL TRIANGLES
PLOTS THE LINE WHICH REPRESENTS THE BEST PATH OF ALL THE DOTS
PLOTS THE LINE WHICH REPRESENTS THE BEST PATH OF ALL THE TRIANGLES

    PLEASE USE AN IDE TO SEE THE OUTPUT AS MENTIONED AT THE TOP!
"""
plt.scatter(d[["S.No."]],d[["score"]],s=10)
plt.scatter(d[["S.No."]],d[["balls"]],s=10,marker = '<')
plt.plot(to_predict1,y_test_predicted1,linewidth=2,color='g')
plt.plot(to_predict2,y_test_predicted2,linewidth=2,color='r')
plt.show()