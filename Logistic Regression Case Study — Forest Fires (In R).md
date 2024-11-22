
# Logistic Regression Case Study — Forest Fires (In R)

I will be working on a sample dataset to predict the burned area of forest fires in northeast Portugal, in the Montesinho National Park. This will be building off a past work by P. Cortez, and I am just redoing it from scratch solely for practice purposes in R. The dataset for this exercise can be found here. The dataset contains no missing values. I am also going to include all my troubleshooting steps.

## The Data Set
Our dependent variable will be the area (burned area of the forest).
Our independent variables will be the other metrics (aka predictors): x,y spatial coordinates; date variables (month, day); weather/environmental variables (FFMC -fine fuel moisture code, DMC — duff moisture code, DC — drought code, ISI — initial spread index, TEMP — temperature (celsius), RH — relative humidity, WIND — windspeed (km/h), RAIN — outside rain (mm/m (squared)). For this blog post I will be omitting x,y spatial coordinates and the date variables (I want to skip categorical transformations).
![Source Data](https://cdn-images-1.medium.com/max/800/1*_kXXgVKy8cOWkftEy0axDA.png)

## Step 1: Define the Business Problem/Research Question and Desired Output
The goal is to predict whether a forest fire occurred in the northeast region of Portugal based on environmental and weather conditions. Specifically, we aim to predict if the area burned is greater than zero (i.e., fire) or equal to zero (i.e., no fire). This is the research question I have created:

### What factors significantly affect the likelihood of a large forest fire in the Montesinho National Park?

Helpful tip: If you are a consultant, a big part of your job would be to help your clients determine what the relevant research questions should be.

### Picking the Right Algorithm
This is a classification problem, and to assess the right model to use we first have to understand the data and how we plan to handle the dependent variable. This is a regression problem, and if we choose to keep the dependent variable (area) in a continuous format, then a Linear Regression (or another continuous model like decision trees or random forest) would be appropriate. From a prediction perspective, a Linear Regression would predict the size of the burned area based on the input variables.

A Logistic Regression on the other hand is typically used when the dependent variable is categorical or binary. If we want to use logistic regression, we will first need to convert the “burned area” into categories (e.g., “low”, “high”, or “burned” vs. “not burned”). If we decide to use a Logistic Regression, we can consider categorizing the burned area into two categories:

- Small Fire” (area ≤ some threshold)
- “Large Fire” (area > some threshold)

So, in conclusion:

- Logistic Regression: Use if you’re predicting categories (e.g., small vs. large fires).
- Linear Regression (or another continuous model): Use if you’re predicting the actual value of the burned area.

Our research question was:

### What factors significantly affect the likelihood of a large forest fire in the Montesinho National Park?

In this research question:

- The dependent variable is binary (e.g., “small fire” vs. “large fire”), where you would choose a threshold to define “large” and “small.”
- We can use logistic regression to model the probability that a fire will result in a large, burned area based on the independent variables. Logistic regression is ideal for binary classification problems, which fits the research question of predicting whether a fire will burn a large area.

Our model output will also give us additional useful information. So, the final research question and desired outcome would read as:

- Research question: What factors significantly affect the likelihood of a large forest fire in the Montesinho National Park?
- Desired Output: A binary outcome — whether the burned area is “high” or “low.”

## Step 2: Define the Variables and Thresholds
We need to determine how we will define “high” versus “low” burned area for our binary classification. For this blog post, we can use median as the threshold.

Helpful tip: If you are a consultant, you may have to work with your clients to understand what the ideal threshold should be based on the data and business objectives.

We will create a binary variable AreaBinary where burned area > X (e.g., median area) is labeled "1" (high), and burned area ≤ X is labeled "0" (low).
> median_area <- median(Forestfiresnew$area)
> Forestfiresnew$AreaBinary <- ifelse(Forestfiresnew$area > median_area, 1, 0)

## Step 3: Prepare the Data (Handle Missing Values, Outliers)
Clean data is crucial for model performance. Outliers and missing values can distort the model’s ability to find meaningful patterns. Let's check for missing values.

> sum(is.na(Forestfiresnew))
[1] 0

If needed, we can address missing values with mean imputations:

> Forestfiresnew[is.na(Forestfiresnew)] <- lapply(Forestfiresnew, function(x) mean(x, na.rm = TRUE))

Outliers are often defined as values that lie outside 1.5 times the interquartile range (IQR). We can calculate the IQR for each column in our dataset:

> Q1 <- apply(Forestfiresnew[, 1:8], 2, quantile, 0.25)
> Q3 <- apply(Forestfiresnew[, 1:8], 2, quantile, 0.75)
> IQR <- Q3 - Q1
We can define the outliers as those outside 1.5 * IQR

lower_bound <- Q1 - 1.5 * IQR
> upper_bound <- Q3 + 1.5 * IQR

And then find the outliers:

> outliers <- apply(Forestfiresnew[, 1:8], 2, function(x) x < lower_bound | x > upper_bound)

![Outliers in R](https://cdn-images-1.medium.com/max/800/1*O1SwVj4Jv5z3LG9ji3e8Sg.png)

Once we have found our outliers, we can either remove them, replace them with the median or mean, or replace extreme values with the nearest non-outlier value (lower and upper quantiles).

Helpful tip: Deciding on which approach to address the outliers will require you to understand the context behind the data. This is where the synergy between an analyst and strategist comes into play. In our case, environmental data often has natural variability — extreme temperatures or high wind speeds can be meaningful in predicting forest fires. Removing outliers might eliminate important signals. Also removing outliers might reduce the dataset size, which could impact the model's ability to generalize well.

For our project, I am going to choose Capping/Winsorizing, as instead of eliminating the extremes, I will rather adjust their influence, which should help stabilize our predictions.

> Forestfiresnew[, 1:8] <- apply(Forestfiresnew[, 1:8], 2, function(x) pmin(pmax(x, lower_bound), upper_bound))
Helpful tip: The 1:8 in the winsorizing code refers to the column indices in our dataset, ensuring that the winsorizing is only applied to those columns.

## Step 4: Exploratory Data Analysis (EDA)
EDA helps understand the relationships between variables and identify trends or patterns. This step can help identify strong predictors.

Let's start with building a correlation matrix:

> cor(Forestfiresnew)
                  FFMC         DMC           DC        ISI       temp           RH       wind
FFMC        1.00000000 0.795586035  0.594094802 0.48505042 0.55811842  0.795647116 0.46642614
DMC         0.79558604 1.000000000  0.677529266 0.33976260 0.41802282  0.616824578 0.31395418
DC          0.59409480 0.677529266  1.000000000 0.05416634 0.08211946  0.299387830 0.06154114
ISI         0.48505042 0.339762604  0.054166336 1.00000000 0.97617209  0.780623575 0.99167144
temp        0.55811842 0.418022823  0.082119460 0.97617209 1.00000000  0.806348155 0.96224889
RH          0.79564712 0.616824578  0.299387830 0.78062357 0.80634816  1.000000000 0.76993001
wind        0.46642614 0.313954178  0.061541139 0.99167144 0.96224889  0.769930005 1.00000000
rain        0.46756962 0.311784835  0.072710171 0.98776613 0.95936704  0.769113436 0.99723464
area        0.03819112 0.006347472 -0.036451848 0.05417062 0.08298664  0.044515051 0.04990589
AreaBinary -0.00744896 0.009783570  0.003215623 0.02762001 0.03738500 -0.007561772 0.02755835
                 rain         area   AreaBinary
FFMC       0.46756962  0.038191125 -0.007448960
DMC        0.31178484  0.006347472  0.009783570
DC         0.07271017 -0.036451848  0.003215623
ISI        0.98776613  0.054170616  0.027620010
temp       0.95936704  0.082986637  0.037384995
RH         0.76911344  0.044515051 -0.007561772
wind       0.99723464  0.049905890  0.027558348
rain       1.00000000  0.049032010  0.021435874
area       0.04903201  1.000000000  0.339614599
AreaBinary 0.02143587  0.339614599  1.000000000

- ISI and temp have a very high correlation (0.976). Similarly, ISI and wind (0.991), and rain and wind (0.997) also show very high correlations. These high correlations indicate multicollinearity, which could cause problems in our logistic regression model. We may want to consider removing or combining these variables or using regularization techniques to mitigate multicollinearity. We can also consider using techniques like Principal Component Analysis (PCA) to reduce dimensionality.
- The area variable (the dependent variable) shows relatively low correlations with all other variables, suggesting that none of the predictors are strongly related to the burned area of forest fires in this dataset. However, this does not mean the variables will not be significant in a logistic regression model, as the nature of logistic regression allows for non-linear relationships to be captured.
- Similarly, the AreaBinary variable (the binary version of the dependent variable) has very low correlations with all other variables. The threshold settings could impact this in our model.

Our EDA has enabled us to catch some relationships that could affect our model. For this blog post I will focus on the multicollinearity issue.

Helpful tip: Dropping some variables is a simple way to go, but the tradeoff is loss of information, which can have a big impact if the dropped variable has strong predictive power. This again is where strategy meets analytics. If the variables are similar (like wind speed and rain, which are weather-related), you might choose to drop one of them. In our case, we could drop ISI, wind, or rain, as they are highly correlated, but we would need to carefully analyze which one makes more sense to retain based on the domain knowledge of forest fires.

- PCA reduces the number of correlated variables and will maintain all the original information in a compressed form, but it will transform the variables into components that do not have a direct relationship with the original variables. It will also change the nature of the variables, making the results less interpretable.
- Based on all this, I will drop ISI or wind (as they have high correlations with each other and other variables like rain). And after dropping, will check to make sure it was removed (I dropped ISI):
- 
> Forestfiresnew <- Forestfiresnew[ , !names(Forestfiresnew) %in% c("ISI")]
> head(Forestfiresnew)
     FFMC      DMC       DC    temp      RH    wind    rain  area AreaBinary
1 96.1000  96.9500  96.9500 86.1500 86.1500 86.1500 86.1500 40.54          1
2 94.9000 130.3000 259.5000 33.1000 25.0000  4.0000  0.0000 26.43          1
3 96.2000 175.5000 661.8000 32.6000 26.0000 15.1500 15.1500  2.77          1
4 17.1125  17.1125  17.1125 17.1125 17.1125  2.2000  0.0125  0.00          0
5 33.7125  33.7125  33.7125 32.4000 21.0000  4.6125  4.6125  0.00          0
6 83.0000  83.0000  83.0000 32.3000 27.0000  3.0000  3.0000 14.68          1
The correlation analysis proved to be very helpful, but we can also generate box plots. Box plots will be helpful in identifying any remaining outliers.

> boxplot(Forestfiresnew)
![Boxplot visual](https://cdn-images-1.medium.com/max/800/1*8G0tBQIoWTrjduxpBMhcag.png)

From the box plot I see that variables like DC, area, and DMC have significant outliers. But remember that we ran the Winsorization process previously. Winsorization caps extreme values to specified bounds (e.g., at the 5th and 95th percentiles). So, values that are extreme but within the range set by those bounds will still appear as “outliers” in the box plot. This doesn’t mean they are invalid; they are simply values near the new capped limits. Also consider that outliers can sometimes represent valid observations, especially in natural processes (like forest fires). It’s essential to consider the domain knowledge — extreme weather events, for instance, could legitimately lead to large, burned areas, so those “outliers” might still be valuable data points.

Helpful tip: It’s always worth double-checking the upper and lower bounds used in the Winsorization process. If the outliers are still concerning, you can consider further strategies (e.g., transformation, removing extreme cases if justified).

Let's go through the final step of our EDA and create a Histogram to study the distribution of the data. We can do it for temperature:

> hist(Forestfiresnew$temp)
![Histogram](https://cdn-images-1.medium.com/max/800/1*1r0SFPm2uadLMHUchBUtyA.png)

The histogram we created for the variable temp from the dataset seems to show that most of the data is concentrated between 0°C and 40°C, with an outlier (or small cluster of values) at around 80°C. This is important to note, as extreme temperature values could significantly impact our model's predictions, especially if those values are rare compared to the rest of the distribution. The spike around 80°C could represent a potential outlier. It’s important to consider if this temperature is valid based on domain knowledge or if it might be due to data entry errors.

Helpful tip: If you don’t have much domain knowledge, it's always good practice to take a cautious approach with outliers, especially if you are not certain if they are anomalies or valid data points. Context matters. In the absence of domain knowledge, we can’t confidently say whether these outliers represent valid extreme cases (e.g., rare but possible extreme temperatures or wind speeds) or erroneous data (e.g., data entry mistakes). Outliers can sometimes hold valuable information about rare events, which may be crucial in predicting forest fires.

Helpful tip: Logistic regression is sensitive to extreme values, and significant outliers could distort the model’s results, leading to biased coefficients or predictions. However, sometimes models can handle outliers reasonably well, especially if the outliers are not overly extreme.

We’ve already applied Winsorization to cap extreme values, and despite this, there are still some visible outliers in the histograms. This suggests that these values may be naturally occurring and not necessarily harmful to our model. Since we applied a moderate form of Winsorization, the extreme values that remain might still be useful for capturing the variability in the data.

Based on all these considerations, the strategy I am going to employ is:

Build the model with outliers first: Leave the outliers in and build our logistic regression model.
Evaluate the model’s performance: Examine how well the model performs on both the training and test sets. I will check metrics like accuracy, precision, recall, and the confusion matrix. Look for signs of poor fit, such as unusually high variance or erratic predictions.
Sensitivity analysis: In the future we can later experiment by removing the outliers or applying more aggressive outlier handling techniques (such as further Winsorization or removing values beyond a certain threshold) to see how it affects the model’s performance.
Ok we have finished our EDA.

## Step 5: Create Binary Dependent Variable
The dependent variable for logistic regression must be binary (1 or 0). We already handled this when we created the AreaBinary variable:

> table(Forestfiresnew$AreaBinary)

  0   1 
257 257

## Step 6: Split the Dataset into Training and Testing Sets
To avoid overfitting, we will need to split our data so we can train the model on one portion and test it on another.

> set.seed(123)  # Ensures reproducibility
> trainIndex <- sample(1:nrow(Forestfiresnew), 0.7*nrow(Forestfiresnew))
> trainData <- Forestfiresnew[trainIndex, ]
> testData <- Forestfiresnew[-trainIndex, ]

Now we can build the Logistic Regression Model

## Step 7: Build the Logistic Regression Model
This is the core step where the logistic regression algorithm is applied to the training data. Remember that we dropped ISI previously, so the updated Logistic Regression model will read as:

> model <- glm(AreaBinary ~ FFMC + DMC + DC + temp + RH + wind + rain, 
+              data = trainData, family = binomial)
> summary(model)

Call:
glm(formula = AreaBinary ~ FFMC + DMC + DC + temp + RH + wind + 
    rain, family = binomial, data = trainData)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-1.4514  -1.1544  -0.9774   1.1856   1.5272  

Coefficients:
              Estimate Std. Error z value Pr(>|z|)
(Intercept) -0.1970572  0.2471687  -0.797    0.425
FFMC         0.0005159  0.0064922   0.079    0.937
DMC         -0.0001065  0.0039884  -0.027    0.979
DC           0.0010209  0.0009709   1.051    0.293
temp         0.0108398  0.0195515   0.554    0.579
RH          -0.0100775  0.0100437  -1.003    0.316
wind         0.0498156  0.0544072   0.916    0.360
rain        -0.0489753  0.0511243  -0.958    0.338

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 497.45  on 358  degrees of freedom
Residual deviance: 493.16  on 351  degrees of freedom
AIC: 509.16

The model shows the coefficients for each variable, the p-value which determines which values are statistically significant, and values like AIC and deviance to assess the models overall fit.

- The intercept (Estimate = -0.1971, p = 0.425) represents the log-odds of the dependent variable (AreaBinary) being 1 when all predictor variables are set to 0. Since the p-value is greater than 0.05, it’s not statistically significant.
- FFMC: This is a very small positive coefficient, but it’s not statistically significant (high p-value). This suggests FFMC is not a strong predictor in the model.
- DMC: The effect is very close to zero, and the high p-value means it is not statistically significant.
- DC/Drought Code: Logged an Estimate = 0.00102, p = 0.293 . This suggests a positive relationship between DC and the binary burned area, but it is not statistically significant.
- Temperature: Estimate = 0.0108, p = 0.579. A positive coefficient but not statistically significant.
- Relative Humidity: Estimate = -0.0101, p = 0.316. A negative coefficient but not statistically significant.
- Windspeed: Estimate = 0.0498, p = 0.360. A positive coefficient but again not statistically significant.
- Rain: Estimate = -0.04898, p = 0.338. A negative coefficient but also not statistically significant.
- None of the predictors in our model had a p-value less than 0.05, meaning none of the variables are statistically significant in predicting the AreaBinary outcome at the conventional significance level.

Goodness of fit: The residual deviance (493.16) is slightly lower than the null deviance (497.45), but the difference is not large. This suggests the model is only marginally better than a model with no predictors.

AIC (Akaike Information Criterion) is 509.16. This value helps in comparing models (lower AIC means a better model).

Helpful tip: Even though the p-values are not significant, we should evaluate the overall model performance on a test set to see if it can correctly classify instances. We can use a confusion matrix, ROC curve, or other evaluation metrics to assess how well the model predicts.

Helpful tip: If none of the variables are significant, it might be worth trying feature selection techniques such as stepwise regression to identify a subset of predictors that might be more predictive.

## Step 8: Evaluate the Model (Confusion Matrix, Visualization)
Let’s evaluate the performance using accuracy metrics such as confusion matrix, sensitivity, specificity, etc.

> library(caret)
> predictions <- predict(model, testData, type = "response")
> testData$predicted <- ifelse(predictions > 0.5, 1, 0)
> confusionMatrix(as.factor(testData$predicted), as.factor(testData$AreaBinary))
Confusion Matrix and Statistics

          Reference
Prediction  0  1
         0 44 41
         1 29 41
                                          
               Accuracy : 0.5484          
                 95% CI : (0.4665, 0.6283)
    No Information Rate : 0.529           
    P-Value [Acc > NIR] : 0.3443          
                                          
                  Kappa : 0.1018          
                                          
 Mcnemar's Test P-Value : 0.1886          
                                          
            Sensitivity : 0.6027          
            Specificity : 0.5000          
         Pos Pred Value : 0.5176          
         Neg Pred Value : 0.5857          
             Prevalence : 0.4710          
         Detection Rate : 0.2839          
   Detection Prevalence : 0.5484          
      Balanced Accuracy : 0.5514          
                                          
       'Positive' Class : 0

## Confusion Matrix Results
### Accuracy:

The model’s accuracy is 0.5484 (or ~54.8%), which means it correctly classified 54.8% of the test data.
The 95% confidence interval for accuracy is between 46.65% and 62.83%. This is slightly better than random guessing (the No Information Rate of 0.529, meaning that if you predicted the majority class, you’d get around 52.9% accuracy).

### Sensitivity:

Sensitivity is 0.6027 (or ~60.27%), meaning the model correctly identified 60.27% of the actual class ‘0’ instances (burned area less than the threshold).

### Specificity (True Negative Rate)

Specificity is 0.5000 (or 50.00%), meaning the model correctly identified 50% of the actual class ‘1’ instances (burned area greater than or equal to the threshold).

### Positive Predictive Value (Precision for class 0):

This is 0.5176, meaning that 51.76% of the predicted ‘0’ instances were actual ‘0’s.

### Negative Predictive Value (Precision for class 1):

This is 0.5857, meaning that 58.57% of the predicted ‘1’ instances were actual ‘1’s

### Kappa Statistic:

The Kappa value is 0.1018, which indicates very poor agreement between the predicted and actual classifications beyond chance.

### McNemar’s Test P-Value:

The p-value is 0.1886. McNemar’s test checks if there’s a significant difference in classification errors between the two classes. Since the p-value is greater than 0.05, it suggests that the model’s classification errors are not significantly biased in favor of one class.

So far it seems we have an “okay” model, a model that works but could be improved. Let's look at some other evaluations like Precision, Recall and F1 Score:

> library(caret)
> confusionMatrix(as.factor(testData$predicted), as.factor(testData$AreaBinary), mode = "prec_recall")
Confusion Matrix and Statistics

          Reference
Prediction  0  1
         0 44 41
         1 29 41
                                          
               Accuracy : 0.5484          
                 95% CI : (0.4665, 0.6283)
    No Information Rate : 0.529           
    P-Value [Acc > NIR] : 0.3443          
                                          
                  Kappa : 0.1018          
                                          
 Mcnemar's Test P-Value : 0.1886          
                                          
              Precision : 0.5176          
                 Recall : 0.6027          
                     F1 : 0.5570          
             Prevalence : 0.4710          
         Detection Rate : 0.2839          
   Detection Prevalence : 0.5484          
      Balanced Accuracy : 0.5514          
                                          
       'Positive' Class : 0

- Precision (for class ‘0’): The proportion of instances predicted as class ‘0’ that are actually class ‘0’. Here, the precision is 51.76%, meaning that when the model predicts a burned area (class ‘0’), it is correct about half of the time.

- Recall (Sensitivity for class ‘0’): The proportion of actual class ‘0’ instances that are correctly predicted by the model. In this case, the recall is 60.27%, which means the model successfully identifies about 60% of the actual burned areas.

- F1 Score: The harmonic mean of precision and recall. The F1 score is 55.70%, which balances both precision and recall. A low F1 score indicates that the model is not excelling in either precision or recall.

## Using the Model for Predictions
To use our model to make predictions, we can create code that creates a new dataset (or data frame) that contains the feature values for which we want to make a prediction. We can substitute any values we want in the prediction code, and it will apply the algorithm we created to those new values. I am going to create a new prediction data frame called “newdata” to illustrate. Below are two examples with different values:

> newData <- data.frame(FFMC=90, DMC=120, DC=650, temp=30, RH=35, wind=4, rain=0)
> predict(model, newData, type = "response")
        1 
0.6619281 
> newData <- data.frame(FFMC=95, DMC=160, DC=750, temp=20, RH=50, wind=2, rain=5)
> predict(model, newData, type = "response")
        1 
0.5419603
This code is what you can feed into platforms like Google Cloud, Azure and AWS via APIs and automated schedulers to automate prediction algorithms. You would of course have to train the model on whichever Cloud platform you use.

## Conclusion
So, we have successfully built a model that can provide answers to our research question; it's not a very accurate model, but it's a model that works, and that definitely needs improvement/tweaking. Hyperparameter tuning using cross validation, additional feature engineering, resampling techniques (i.e. SMOTE) or adjusting the thresholds can be used to further balance and improve the model's performance, and we will cover these enhancements in separate posts.