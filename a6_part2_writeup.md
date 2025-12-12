# Assignment 6 Part 2 - Writeup

---

## Question 1: Feature Importance

Based on your house price model, rank the four features from most important to least important. Explain how you determined this ranking.

**YOUR ANSWER:**
1. square feet
2. bathrooms
3. bedroom
4. age 

**Explanation:**
Square feet has the most linear correlation between itself and price. Bedrooms and bathrooms are similar in their variation, so they are both below square feet. I placed bathrooms higher, however, because since our data accomodated less maximum bathrooms, each one was more valuable than each bedroom. Age had the least cof=herent graph, so I placed it at the bottom.



---

## Question 2: Interpreting Coefficients

Choose TWO features from your model and explain what their coefficients mean in plain English. For example: "Each additional bedroom increases the price by $___"

**Feature 1:**
Each bathroom ammounted to around 80000 dollars in price gained

**Feature 2:**
Each bedroom was worth around 70000 dollars in price

---

## Question 3: Model Performance

What was your model's R² score? What does this tell you about how well your model predicts house prices? Is there room for improvement?

**YOUR ANSWER:**
0.9936246192948249. This means it has high predictive power as the closer it gets to 1 the better the predictions are. I think it could be improved, but I am also very happy with the score I got here.



---

## Question 4: Adding Features

If you could add TWO more features to improve your house price predictions, what would they be and why?

**Feature 1:**
Floors

**Why it would help:**
Some people would want more or less floors on their house, so I think it would return interesting data

**Feature 2:**
doe sit have a garage

**Why it would help:**
People with car would like to have a garage, so they could likely favor having a house with one

---

## Question 5: Model Trust

Would you trust this model to predict the price of a house with 6 bedrooms, 4 bathrooms, 3000 sq ft, and 5 years old? Why or why not? (Hint: Think about the range of your training data)

**YOUR ANSWER:**
Not really, because that data is mostly out of the range of what we used to train the model, so it would probably be at least somewhat innacurate

