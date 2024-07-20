
pnorm(1.75)-(1-pnorm(2))
pnorm(1)

#----Hypothesis testing from scratch programming ---
# -----t-test statistic:------------
n=30
alpha=0.05
mean=50
xbar=51.79
stdev=3.39
# tstat=2.89
tstat=(xbar-mean)/(stdev/sqrt(n))

critical=qt(c(alpha/2,1-alpha/2),df=29)   # (2-tailed) critical value finding codes for two tailed test
critical=qt(c(alpha),df=29)   # (left tailed) critical value finding codes for left tailed test

critical=qt(c(1-alpha),df=29)   # (right-tailed) critical value finding codes for right tailed test

pvalue=2*(1-pt(tstat,n-1))      # p-value for two sided test
#---------z-test statistic:-------------
critical=qnorm(alpha, lower.tail = TRUE)   # (left-tailed): critical value findings for Z-test
critical =qnorm(alpha, lower.tail = FALSE)
        
#---------One sample t test:----
set.seed(100)
df<-rnorm(20,2,1)
# Ho:Mean=2
#Ha:Mean>2

t.test(df,mu=2,alt='greater')
t.test(df, mu=2, alt='two.sided')

# two independent sample t-test
set.seed(10)
df1<-rnorm(18,2,1)
df2<-rnorm(18,3,1.5)
mean(df1)
mean(df2)
# Ho:Mean1=Mean2 => (Mean1-Mean2)=0
#Ha:Mean1 != Mean2

t.test(df1,df2,mu=0, alt='two.sided')

#Paired or Dependent t-test
t.test(df1,df2, mu=0, alt='two.sided',paired=TRUE)


#----------------------------------
# Creating data vector
max.temp<-c('Sun'=22,'Mon'=27,'Tue'=26,'Wen'=24,'Thu'=23,'Fri'=26,'Sat'=28)


par(mfrow=c(1,2))    # set the plotting area into a 1*2 array
barplot(max.temp, main="Barplot")
pie(max.temp, main="Piechart", radius=1)



# -----ANOVA test-------------------------
Group1=c(2,3,7,2,6)
Group2=c(10,8,7,5,10)
Group3=c(10,13,14,13,15)

Combined_Group<-data.frame(cbind(Group1,Group2,Group3))
summary(Combined_Group)

Stacked_Groups<-stack(Combined_Group)
Stacked_Groups
Anova_Results<-aov(values ~ ind, data=Stacked_Groups)
summary(Anova_Results)

#-----ANOVA----------------------------------------------------
team= c(rep('Dalls',17),rep('GB',17),rep('Denver',17),rep('Miami',17),rep('SF',17))

#---------------------------------------------------

#---- Paired t-test statistic--------------

# Creating data 
before<-c(12.2,14.6,13.4,11.2,12.7,10.4,15.8,13.9,9.5,14.2)
after<-c(13.5,15.2,13.6,12.8,13.7,11.3,16.5,13.4,8.7,14.6)

#Creating a dataFrame and observe the data

data<-data.frame(subject=rep(c(1:10),2),time=rep(c("before","after"),each=10),score=c(before,after))
print(data)

#Correlation b/w before and after

cor.test(x=before, y=after, method=c("pearson"),con.level=0.95)
#----plot b/w after and before:---
plot(after,before)
abline(a=0,b=1)
# Box plot b/w score and time
boxplot(data$score ~ data$time)   # First methos

boxplot(data$score ~ data$time, main="ICT training score" ,xlab="Time", ylab="Score")


#Paired t-test

t.test(data$score ~ data$time,mu=0,alternative="greater",  paired=TRUE, var.equal=TRUE, conf.level=0.95)


\\

# two sample Z-proportion test 
morning<-matrix(c(88,112,80,120),ncol=2,byrow = TRUE)

rownames(morning)<-c("Female","Male")
colnames(morning)<-c("Early","Late")

morning=as.table(morning)
morning


#Bar plotting
barplot(morning, beside=TRUE)
tsxt(1.5,92,"Female")

# Now two sample z-proportion test

prop.test(morning, alternative = "greater")

#---2nd method of solving Z-proportion test by importing csv dataset from excel:---
# fist check directory : getwd() and you can set: setwd()

mydat=read.csv("C:/Users/Zeeshan Haleem/Documents/R-code files/Zprop.csv")
names(mydat)
table1=table(mydat$GilbertWorked, mydat$Patient)

table1
prop.test(table1, correct=FALSE)

#----ANOVA of variance for the plants weights:------
# To see all in-build dataset
data()

#---Now importing PlantGrowth dataset:----
data(PlantGrowth)




# Explore data
head(PlantGrowth)
summary(PlantGrowth)

#Levels for group: in order to see unique values

levels(PlantGrowth$group)

#Extracting variable and defining a new variable3
weight= PlantGrowth$weight
group=PlantGrowth$group

# Compare means of weights
mean(weight[group=='ctrl'])
tapply(weight, group, mean)

#Hypothesis setup
# H0: mu(ctrl)=mu(ctrl1)=mu(ctrl2)
# Ha: at lest one mean(weight of plant) is different

# data visualization
boxplot(weight ~ group, main="Plants vs weight", xlab = 'Plant group',col=rainbow(3))

#ANOVA
aov=aov(weight ~ group)
summary(aov)

# Post-hoc test in order to see comparsion

TukeyHSD(aov)

# Plot tukey
plot(TukeyHSD(aov))

# Now non-parametric test Krushkal wallis : alternative to ANOVA
kruskal.test(weight ~ group)

#equality of variances
bartlett.test(weight~group)


#----Non parametric test statistic: Wilcoxon singed rank test an alternative to paired t-test 

# Creating a dataframe of the effect of caffeine on myocardial blood flow:

subject=c(1,2,3,4,5,6,7,8,9,10)
baseline=c(3.43,3.08,3.19,2.65,2.49,2.33,2.31,2.24,2.17,1.34)
caffeline=c(2.72,2.94,1.33,2.16,2.00,2.37,2.35,2.26,1.72,1.22)

MBF= data.frame(subject,baseline,caffeline)

#10 subject had their blood flow measure
# before and after consuming caffeine

diffs= MBF$baseline-MBF$caffeline
shapiro.test(diffs)

# qqplot for Normality check of the dataset
qqnorm(diffs)
qqline(diffs)
# As one can easility get to know after seeing this plot result that the data is not normally distribted
# Therefore we use wilcoxon-signed rank test
 
# Hypotheses:
#H0: There is no difference in MBF before and after caffeine consumption
#Ha: There is a difference in MBf befiore anf after caffeine consumption


wilcox.test(MBF$baseline, MBF$caffeline, paired = T)

# Since p-value=0.032< alpha(0.05), therefore, rejecting null hypothesis
# Conclusion: There is sufficient evidence to support the claim that there is significant differences in  MBF  afte caffein consumption.

#---------------------------------------------
#-------------Multiple linear regression :----
#----applying multiple linear regereesion on fuel consumption dataset:

# Checking all available dataset in the directory
data()
# Now impoting fuel2001 dataset into datafame
data("fuel2001")

# looking structure of the datset by applying the following code
str(fuel2001)
# Pair plot of the given dataset
pairs(fuel2001)
# Correlation
cor(fuel2001)
# creating a multiple linear regession model
ml<-lm(fuel2001$FuelC~fuel2001$Income+fuel2001$Miles+fuel2001$Tax)
summary(ml)

# Prediction

predict(ml, data.frame("Income"=35000,"Miles"=50000,"Tax"=20))


#--------------------------------------------------
#-----------Multiple regression model:---------------
#----web link: https://www.tutorialspoint.com/r/r_multiple_regression.htm:    ---

data()
mtcars
input=mtcars[,c("mpg",'disp','hp','wt')]

print(head(input))


# create relationship model and get the coefficients

model=lm(mpg~disp+hp+wt, data=input)
#show mode;
print(model)
#get the intercept and coefficent as vector elemert

a=coef(model)[1]

Xdisp=coef(model)[2]
Xhp=coef(model)[3]
Xwt=coef(model)[4]

print(Xdisp)
print(Xhp)
print(Xwt)
print(a)



# Craete Equation for regreesion model

Y=a+Xdisp.x1+Xhp.x2+Xwt.x3
or

Y=37.15+(-0.000937)*x1+(-0.0311)*x2+(-3.8008)*x3
#Apply Equation for predicting New Values
#We can use the regression equation created
#above to predict the mileage when a new set 
#of values for displacement, horse power and 
#weight is provided.

#For a car with disp = 221, hp = 102 and wt = 2.91 the predicted mileage is ???

Y=37.15+(-0.000937)*221+(-0.0311)*102+(-3.8008)*2.91
print(Y)

#---------------------------------------------------------------
x=c(29.4,39.2,49.0,58.8,68.6,78.4)
y=c(4.25,5.25,6.50,7.85,8.75,10.00)
model1=lm(y~x)
model1



confint(model1,'x',level = 0.95)

summary(model1)     # Anova table regression table

confint(model1,level = 0.95)    # confidence interval
#-------------------------------------------

y=c(38,43,29,32,26,33,19,27,23,14,19,21)
x=c(30,30,30,50,50,50,70,70,70,90,90,90)

model=lm(y~x)
model

summary(model)

new_data=data.frame(x=65)          
predict(model,newdata=new_data, interval='confidence')    # predicting new value at x=65


##---------------Linear regression machine learning model: -------For interview------------------


Link1: https://www.youtube.com/watch?v=1-URCcgTBf4
Link2:https://www.kaggle.com/andyxie/regression-with-r-boston-housing-price
Link3:https://rstudio-pubs-static.s3.amazonaws.com/596477_8cd4d2ae33e244b0b666e1949e7f4498.html
Link4:https://towardsdatascience.com/linear-regression-on-boston-housing-dataset-f409b7e4a155


\\

#To load dataset we can use the following code 
library(MASS)
data(Boston)
attach(Boston)


# Checking na values persent in thdatasets:
sum(is.na(Boston))
# Making a dataset without missing values
Boston<-na.omit(Boston)
# For description of the data we can use 
?Boston
# Descriptive statististic of thdatasets::
dim(Boston)
str(Boston)
summary(Boston)

#Exploratory dataset
# pairplot
pairs(Boston[,1:4],pch=19)

# Hitogram plot
hist(medv,xlab='medv',main='Medv distribution')

hist(medv, col='red',xlab='medv',main='Medv distribution')
\\
hist(medv,breaks = 20, col='red',xlab='medv',main='Medv distribution')

\\

hist(medv,breaks = 20, 
     col='red',xlab='medv',
     main='Medv dist',
     )
rug(jitter(Bosto$mdev))
lines(density(Boston$medv),col='blue',lwd=2)
# KDE_plot
par(mfrow=c(2,1))
d<-density(Boston$medv)
plot(d)
d<-density(Boston$medv)
plot(d,main='KDE of mdedv')
polygon(d, col='red', border = 'blue')
rug(Boston$medv,col='brown')

# Boxplot
boxplot(Boston$medv, Boston$crim,
        Boston$chas,
        ,col=('red','green','gold'),
        main='Boxplot of medv')

\\
boxplot(crim~indus,
        data=Boston,
        varwidth='True'
        ,col=c('red','green'),
        main='Boxplot of medv')
\\
data("airquality")
attach(airquality)

boxplot(Ozone, ozone_norm, temp, temp_norm,
        main = "Multiple boxplots for comparision",
        at = c(1,2,4,5),
        names = c("ozone", "normal", "temp", "normal"),
        las = 2,
        col = c("orange","red"),
        border = "brown",
        horizontal = TRUE,
        notch = TRUE)
\\
install.packages('vioplot')
library(vioplot)
vioplot(Boston$medv,col='red',names = c('zee'))
title('violin plots')
\\
# dotplot
dotchart(Boston$medv)

# barplot
barplot(Boston$medv,col='red')
\\
# Correlation plot

cr<-cor(Boston)


install.packages('corrplot')
library(corrplot)
corrplot(cr, type='lower')
corrplot(cr,method = 'number')

\\


plot(rm,medv)
abline(lm(medv~rm), col='red')

# we will split the dataset into training and testing sets
# 1st method
install.packages('cartools') 
install.packages('caTools') 
require(caTools)
library(caTools)

set.seed(132)
split<-sample.split(Boston$medv, SplitRatio=0.7)
#we devide the dataset with the ratio 0.7

training_data=subset(Boston, split=='True')
testing_data=subset(Boston, split=='False')

# 2nd method

set.seed(100)
# row for indices for training datset
trainingRowIndex<-sample(1:nrow(Boston),0.7*nrow(Boston))  # sampel(1st part: seq of data, 2nd part: size of the dataset)
# training dataset
trainingData<- Boston[trainingRowIndex,] #model training data
testData<-Boston[-trainingRowIndex,] # test data

\\

# Finding Multioclinearlity checking:
installed.packages()
installed.packages('caret')
library(caret)

#or
names(Boston) # to see the all columns names
model <- lm(medv ~., data = trainingData )

#or
model=lm(medv~crim+zn+inud+chas+nox+rm+age+dis+rad+ration+black+lstat,data=training_data)
\\
install.packages('car')
library(car)
vif(model)
# to get the summary of the model 
summary(model)
# Calculate akaike information criterion
AIC(model)
# Capture model as summary object
modelSummary<- summary(model)          # Capture model summary
modelCoeffs<- modelSummary$coefficients # model coefficient
#  get the estimate foe 'crim'
beta.estimate<-modelCoeffs['crim','Estimate']
# get std.error for speed
std.error<-modelCoeffs['crim','Std. Error']
#Calculate t statistic
t_value<-beta.estimate/std.error
#Calc p-value
p_value<- 2*pt(-abs(t_value),df=nrow(Boston)-ncol(Boston))

f_statistic<-model$fstatistic[1]
f<-summary(model)$fstatistic
model_p<-pf(f[1],f[2],f[3], lower=FALSE)

#Predicting the model
predict<-predict(model,testData)
#plotting
plot(testData$medv, type='l',col='green')
lines(predict,type = 'l',col='blue')

#--------------End complete regression analysis:---------------

#-------- Decision tree ML algorithm :---------

Link: https://online.datasciencedojo.com/blogs/a-comprehensive-tutorial-on-classification-using-decision-trees



#-------Normality check----------------------
data=c(418,421,422,425,427,431,434,437,439,446,447,448,453,454,463,465)
plot(density(data))   # plot density
boxplot(data)          # boxplot
shapiro.test(data)     # Shaipro test
qqnorm(data)           # qqplot
#--------------------------------------
# install.packages('MPV')
library(MPV)
'??MPV'
knitr::include_graphics("table5.png")
#-----Importing the ggplot library:----
library(ggplot2)
#----Fitting the linear regression model and its Residuals:----
co2<-MPV::table.b5
d<-co2
attach(data)
#-----First model fitting:---
model1<-lm(y~x6,data = d)
summary(model1)
#---- Saving predicted values by model1:---

d$predicted1<-predict(model1)
#----Saving residuals created by model1
d$residuals1<-residuals(model1)

#-----https://rpubs.com/iabrady/residual-analysis:-------
#-------residuals plotting for model1-----------

ggplot(d, aes(x = x6, y = y)) +
  geom_smooth(method = "lm", se = FALSE, color = "lightgrey") +     # regression line  
  geom_segment(aes(xend = x6, yend = predicted1), alpha = .2) +      # draw line from point to line
  geom_point(aes(color = abs(residuals1), size = abs(residuals1))) +  # size of the points
  scale_color_continuous(low = "green", high = "red") +             # colour of the points mapped to residual size - green smaller, red larger
  guides(color = FALSE, size = FALSE) +                             # Size legend removed
  geom_point(aes(y = predicted1), shape = 1) +
  theme_bw()

#-----------
plot(model1, which = 1, col=c('blue'))  #Residuals vs Fitted plot
plot(model1, which = 2, col=c('blue'))  #Q-Q Plot
plot(model1, which = 3, col=c('blue'))  #Scalen-Location Plot
plot(model1, which = 4, col=c('blue'))  #
plot(model1, which = 5, col=c('blue'))  #Residuals vs Leaverage plot

#Summary
'''Residual analysis plots are a very useful tool 
for assessing aspects of veracity of a 
linear regression model on a particular 
dataset and testing that the attributes 
of a dataset meet the requirements for 
linear regression.'''


#--------------------------------------------------------------


#model2<-lm(y~x6+x7, data = d)

#-----Second model fitting:---

model2<-lm(y~x6+x7,data=d)
summary(model2)
#---- Saving predicted values by model1:---

d$predicted2<-predict(model2)
#----Saving residuals created by model1
d$residuals2<-residuals(model2)
#-------residuals plotting for model1-----------

ggplot(d, aes(x = x6+x7, y = y)) +
  geom_smooth(method = "lm", se = FALSE, color = "lightgrey") +     # regression line  
  geom_segment(aes(xend = x6+x7, yend = predicted2), alpha = .2) +      # draw line from point to line
  geom_point(aes(color = abs(residuals2), size = abs(residuals2))) +  # size of the points
  scale_color_continuous(low = "green", high = "red") +             # colour of the points mapped to residual size - green smaller, red larger
  guides(color = FALSE, size = FALSE) +                             # Size legend removed
  geom_point(aes(y = predicted2), shape = 1) +
  theme_bw()

#-----------
plot(model2, which = 1, col=c('blue'))  #Residuals vs Fitted plot
plot(model2, which = 2, col=c('blue'))  #Q-Q Plot
plot(model2, which = 3, col=c('blue'))  #Scalen-Location Plot
plot(model2, which = 4, col=c('blue'))  #
plot(model2, which = 5, col=c('blue'))  #Residuals vs Leaverage plot

#Summary
'''Residual analysis plots are a very useful tool 
for assessing aspects of veracity of a 
linear regression model on a particular 
dataset and testing that the attributes 
of a dataset meet the requirements for 
linear regression.'''

#---PRESS (predicted residual error sum of squares)
https://en.wikipedia.org/wiki/PRESS_statistic
'''In statistics, the predicted residual error sum of squares (PRESS) 
is a form of cross-validation used in regression analysis to provide 
a summary measure of the fit of a model to a sample of observations 
that were not themselves used to estimate the model. It is calculated 
as the sums of squares of the prediction residuals for those observations'''

https://www.statology.org/press-statistic/
  
#---Creating models
  
model1 <- lm(y~x6, data=d)

model2 <- lm(y~x6+x7, data=d)

#create custom function to calculate the PRESS statistic
PRESS <- function(model) {
  i <- residuals(model)/(1 - lm.influence(model)$hat)
  sum(i^2)
}
  
#calculate PRESS for model 1
PRESS(model1)
#calculate PRESS for model 2
PRESS(model2)
#Summary:
'''It turns out that the model with the lowest PRESS statistic is model 2 
with a PRESS statistic of 3388.604. Thus, we would choose this model as 
the one that is best suited to make predictions on a new dataset.'''


######---------------------------------------------------------------------------------
#---Ploting log normal probability distribution curve:-----

#create density plots
curve(dlnorm(x, meanlog=0, sdlog=.3), from=0, to=5, col='blue')
curve(dlnorm(x, meanlog=0, sdlog=.5), from=0, to=5, col='red', add=TRUE)
curve(dlnorm(x, meanlog=0, sdlog=1), from=0, to=5, col='purple', add=TRUE)

#add legend
legend(6, 1.2, legend=c("sdlog=.3", "sdlog=.5", "sdlog=1"),
       col=c("blue", "red", "purple"), lty=1, cex=1.2)



#----------------------------------

#-----t-test :Hypothesis:----------
mydata= read.csv('2014times.csv', header= TRUE)
view('mydata')
attach(mydata)
xbar=mean(Times)
s=sd(Times)
n=length(Times)
Mu=144 #null hypothesis
tstat=(xbar-Mu)/(s/n^0.5)
tstat
Pvalue=2*pt(tstat, df=n-1, lower=FALSE)
Pvalue
if(Pvalue<0.05) NullHypothesis else "Accepted"

#-----Independent t-test two sample:----


Titan_insurance<-read.csv("Titan_isurance.csv")
Titan_insurance$Old_Scheme<-Titan_insurance$Old_Scheme*1000
Titan_insurance$New_Scheme<-Titan_insurance$New_Scheme*1000

mean(Titan_insurance$Old_Scheme)
mean(Titan_insurance$New_Scheme)

sd(Titan_insurance$Old_Scheme)
sd(Titan_insurance$New_Scheme)

t.test(Titan_insurance$New_Scheme,
       Titan_insurance$Old_Scheme,
       paired = TRUE,
       alt='greater')

#-------Fitting distributions to continous non-censored data:----

install.packages("fitdistrplus")

library('fitdistrplus')
data('groundbeef')
str(groundbeef)
plotdist(groundbeef$serving, histo = TRUE, demp=TRUE)

descdist(groundbeef$serving, boot = 1000)

#----Fit distribution by MLE:---

fw= fitdist(groundbeef$serving,'weibull')
summary(fw)
fg= fitdist(groundbeef$serving,'gamma')
fln= fitdist(groundbeef$serving,'lnorm')
par(mfrow=c(2,2))
plot.legend<-c('weibull','lognormal','gamma')

denscomp(list(fw, fln, fg), legendtext = plot.legend)
qqcomp(list(fw, fln, fg),legendtext = plot.legend)
cdfcomp(list(fw, fln, fg),legendtext = plot.legend)
ppcomp(list(fw, fln, fg),legendtext = plot.legend)

#-----------------------------
#---------DATA Analysis on Crime data:----
# How to Visualize and Compare Distributions in R
# Link:https://flowingdata.com/2012/05/15/how-to-visualize-and-compare-distributions/

# Load crime data:

crime<- read.csv('http://datasets.flowingdata.com/crimeRatesByState-formatted.csv')


# Remove Washington, D.C.
crime.new <- crime[crime$state != "District of Columbia",]

# Remove national averages
crime.new <- crime.new[crime.new$state != "United States ",]

# Box plot
boxplot(crime.new$robbery, horizontal=TRUE, main="Robbery Rates in US")
# Box plots for all crime rates
boxplot(crime.new[,-1], horizontal=TRUE, main="Crime Rates in US")

# Histogram
hist(crime.new$robbery, breaks=10)



# Multiple histograms
par(mfrow=c(3, 3))
colnames <- dimnames(crime.new)[[2]]
for (i in 2:8) {
  hist(crime[,i], xlim=c(0, 3500), breaks=seq(0, 3500, 100),
       main=colnames[i], probability=TRUE, col="gray", border="white")
}


# Density plot
par(mfrow=c(3, 3))
colnames <- dimnames(crime.new)[[2]]
for (i in 2:8) {
  d <- density(crime.new[,i])
  plot(d, type="n", main=colnames[i])
  polygon(d, col="red", border="gray")
}


# Histograms and density lines
par(mfrow=c(3, 3))
colnames <- dimnames(crime.new)[[2]]
for (i in 2:8) {
  hist(crime[,i], xlim=c(0, 3500), breaks=seq(0, 3500, 100), main=colnames[i], probability=TRUE, col="gray", border="white")
  d <- density(crime[,i])
  lines(d, col="red")
}

# Histograms and density lines
par(mfrow=c(3, 3))
colnames <- dimnames(crime.new)[[2]]
for (i in 2:8) {
  hist(crime[,i], xlim=c(0, 3500), breaks=seq(0, 3500, 100), main=colnames[i], probability=TRUE, col="gray", border="white")
  d <- density(crime[,i])
  lines(d, col="red")
}

# Density and rug
d <- density(crime$robbery)
plot(d, type="n", main="robbery")
polygon(d, col="lightgray", border="gray")
rug(crime$robbery, col="red")


# Violin plot
library(vioplot)
vioplot(crime.new$robbery, horizontal=TRUE, col="gray")


# Bean plot
library(beanplot)
beanplot(crime.new[,-1])

#---------------------------------
#-----lognormal plots:-----
#create density plots
curve(dlnorm(x, meanlog=0, sdlog=10), from=0, to=10, col='blue')
curve(dlnorm(x, meanlog=0, sdlog=1.5), from=0, to=10, col='red', add=TRUE)
curve(dlnorm(x, meanlog=0, sdlog=1), from=0, to=10, col='purple', add=TRUE)
curve(dlnorm(x, meanlog=0, sdlog=0.5), from=0, to=10, col='green', add=TRUE)
curve(dlnorm(x, meanlog=0, sdlog=0.25), from=0, to=10, col='orange', add=TRUE)
curve(dlnorm(x, meanlog=0, sdlog=0.125), from=0, to=10, col='black', add=TRUE)

#add legend
legend(6, 1.2, legend=c("sdlog=.3", "sdlog=.5", "sdlog=1"),
       col=c("blue", "red", "purple"), lty=1, cex=1.2)

#---------
# Same standard deviation, different mean
#-----------------------------------------
# Grid of X-axis values
x <- seq(-4, 8, 0.1)
# Mean 0, sd 1
plot(x, dnorm(x, mean = 0, sd = 1), type = "l",ylim = c(0, 0.6), ylab = "", lwd = 2, col = "red")
# Same mean, different standard deviation
#-----------------------------------------
# Mean 1, sd 1
plot(x, dnorm(x, mean = 1, sd = 1), type = "l",
     ylim = c(0, 1), ylab = "", lwd = 2, col = "red")
# Mean 1, sd 0.5
lines(x, dnorm(x, mean = 1, sd = 0.5), col = "blue", lty = 1, lwd = 2)

# Adding a legend
legend("topright", legend = c("1 1", "1 0.5"), col = c("red", "blue"),
       title = expression(paste(mu, " ", sigma)),
       title.adj = 0.75, lty = 1, lwd = 2, box.lty = 0)

#-----------
lb <- min(x) # Lower bound
ub <- 1010   # Upper bound

x2 <- seq(min(x), ub, length = 100) # New Grid
y <- dnorm(x2, Mean, Sd) # Density

plot(x, f, type = "l", lwd = 2, col = "blue", ylab = "", xlab = "Weight")
abline(v = ub) 

polygon(c(lb, x2, ub), c(0, y, 0), col = rgb(0, 0, 1, alpha = 0.5))
text(995, 0.01, "84.13%")


# Create Residual Plots in R

my_data<-mtcars
model_1=lm(mpg~wt, data=my_data)
my_data$predicted<-predict(model_1)     # add a predicted new column and fills it with predicted values generated by the created linear model
my_data$residuals<-residuals(model_1)   # Residuals

library(ggplot2)

#Simple1
ggplot(data=my_data, aes(x=wt, y=mpg))+
  geom_smooth(method = 'lm', se=FALSE)+
  geom_segment(aes(xend=wt, yend=predicted), alpha=.3)
  

#simple2 plots

ggplot(data=my_data, aes(x=wt, y=mpg))+
  geom_smooth(method = 'lm', se=FALSE)+
  geom_segment(aes(xend=wt, yend=predicted), alpha=.3)+
  geom_point(aes(color=abs(residuals), size= abs(residuals)))



# Advanced plots

ggplot(data=my_data, aes(x=wt, y=mpg))+
  geom_smooth(method = 'lm', se=FALSE)+
  geom_segment(aes(xend=wt, yend=predicted), alpha=.3)+
  geom_point(aes(color=abs(residuals), size= abs(residuals)))+
  scale_color_continuous(low='green', high= 'red')+
  guides(color= FALSE, size= FALSE)+
  theme_bw()

#----All residuals plots

plot(model_1)


#--------------------

#----- Create Simple Graphs in R Studio:-----------
setwd('E:\\Z-Jupyter')

titanic<- read.csv('titanic.csv')
View(titanic)

# Set up factors

titanic$Pclass<-as.factor(titanic$Pclass)
titanic$Survived<-as.factor(titanic$Survived)
titanic$Sex<-as.factor(titanic$Sex)
titanic$Embarked<-as.factor(titanic$Embarked)


#----------------------------------------------------------
#----Intro to Data Visualization with R & ggplot2:-------------
#-------------------------------------------------------------

#-----bar:-------
library(ggplot2)

ggplot(data=titanic, aes(x=Survived))+
  geom_bar()

#--if you really want percentages
prop.table(table(titanic$Survived))

# Add some customization for labels and theme

ggplot(titanic, aes(x=Survived))+
  theme_bw()+
  geom_bar()+
  labs(y='Passenger Count', title='Titanic Survival Rates')
#---WE can use color at two aspects 


ggplot(titanic, aes(x=Sex, fill= Survived))+
  theme_bw()+
  geom_bar()+
  labs(y='Passenger Count', title='Titanic Survival Rates')


ggplot(titanic, aes(x=Pclass, fill= Survived))+
  theme_bw()+
  geom_bar()+
  labs(y='Passenger Count', title='Titanic Survival Rates')


#-Q: What was the Survival rate by class of ticket and gender?

ggplot(titanic, aes(x=Sex, fill= Survived))+
  theme_bw()+
  facet_wrap(~Pclass)+
  geom_bar()+
  labs(y='Passenger Count',
       title='Titanic Survival Rates by Pclass and sex')

#Q:- what is the distribution of the ages

ggplot(titanic, aes(x=Age))+
  theme_bw()+
  geom_histogram(binwidth = 5)+
  labs(y='Passenger Count',
       x='Age (Bindwidth=5)',
      title='Titanic Survival Rates')

#Q:- what are the survival rates by age?

ggplot(titanic, aes(x=Age, fill=Survived))+
  theme_bw()+
  geom_histogram(binwidth = 5)+
  labs(y='Passenger Count',
       x='Age (Bindwidth=5)',
       title='Titanic Survival Rates by Age')


#Another way to see the answer using boxplot

ggplot(titanic, aes(x=Survived,y=Age))+
  theme_bw()+
  geom_boxplot()+
  labs(y='Age',
       x='Survived',
       title='Titanic Survival Rates')

#Q7:- What is the survival rates by age when segmented by gender and class of tickes

ggplot(titanic, aes(x=Age, fill=Survived))+
  theme_bw()+
  facet_wrap(Sex~Pclass)+
  geom_density(alpha = 0.5)+
  labs(x='age',
       y='Survived',
       title='Titanic Survival Rates by Age, Pclass and Sex')


#---You can also use histogram

ggplot(titanic, aes(x=Age, fill=Survived))+
  theme_bw()+
  facet_wrap(Sex~Pclass)+
  geom_histogram(binwidth = 5)+
  labs(x='age',
       y='Survived',
       title='Titanic Survival Rates by Age, Pclass and Sex')

#----------------------
#-----ggplot2 Tutorial in R Data Visualization in R:--------

data("iris")

View(iris)

table(iris$Species)

#Scatter plot
plot(iris$Sepal.Length ~ iris$Petal.Length, 
     ylab='Sepal Length', xlab = 'Petal Length', 
     main = 'Sepel vs Petal length',
     col='blue',pch=16)

# -----Histogram 
hist(iris$Sepal.Width)
#----modify Histogram

hist(iris$Sepal.Width, xlab = 'Sepal width',
     main='Distribution of Sepal width',
     col = 'aquamarine3')
#---Boxplot
boxplot(iris$Sepal.Length ~ iris$Species)
#--
boxplot(iris$Sepal.Length ~ iris$Species, xlab='Species',
        ylab='Sepal Length',
        main='Sepal')

boxplot(iris$Sepal.Length ~ iris$Species, xlab='Species',
        ylab='Sepal Length',
        main='Sepal',
        col='green')

#----ggplot
library(ggplot2)
#--Scatter plot
ggplot(data=iris, aes(x=Petal.Length, y=Sepal.Length,col=Species))+geom_point()

#---House dataset
# install.packages("mosaicData")

library(mosaicData)

data('SaratogaHouses')

# write.csv(house, 'houseprice.csv')
# house<-read.csv('E:\\Z-Jupyter\\Houseprice.csv')
View(SaratogaHouses)
attach(SaratogaHouses)

library(dplyr)


#Histogram
ggplot(data=SaratogaHouses, aes(x=price))+ geom_histogram()

ggplot(data=SaratogaHouses, aes(x=price))+ geom_histogram(bins=50, fill='palegreen4',col='green')

ggplot(data=SaratogaHouses, aes(x=price, fill= centralAir))+ geom_histogram(position = 'fill')

# Frequency -Polygon
ggplot(data=SaratogaHouses, aes(x=price))+ geom_freqpoly()

ggplot(data=SaratogaHouses, aes(x=price))+ geom_freqpoly(bins=60)

ggplot(data=SaratogaHouses, aes(x=price, col=centralAir))+ geom_freqpoly(bins=60)

#--Box plot:---

ggplot(data=SaratogaHouses, aes(x=factor(rooms), y=price))+ geom_boxplot()

ggplot(data=SaratogaHouses, aes(x=factor(rooms), y=price, fill= centralAir))+ geom_boxplot()

ggplot(data=SaratogaHouses, aes(x=factor(rooms), y=price, fill= sewer))+ geom_boxplot()

#Smooth-Line
ggplot(data=SaratogaHouses, aes(x=livingArea, y=price))+ geom_smooth()

ggplot(data=SaratogaHouses, aes(x=livingArea, y=price, col=centralAir))+ geom_smooth()
ggplot(data=SaratogaHouses, aes(x=livingArea, y=price, col=centralAir))+ geom_smooth(se=F)
ggplot(data=SaratogaHouses, aes(x=livingArea, y=price, col=heating))+ geom_smooth(se=F)

ggplot(data=SaratogaHouses, aes(x=livingArea, y=price, col=centralAir))+geom_point()+ geom_smooth()
ggplot(data=SaratogaHouses, aes(x=livingArea, y=price, col=centralAir))+geom_point(col='black')+ geom_smooth(method = 'lm', se=F)

#Facetting -(Multiple plots)

ggplot(data=SaratogaHouses, aes(x=livingArea, y=price, col=centralAir))+geom_point()+ geom_smooth(method = 'lm', se=F)+facet_grid(~centralAir)

ggplot(data=SaratogaHouses, aes(x=livingArea, y=price, col=factor(fireplaces)))+geom_point()+ geom_smooth(method = 'lm', se=F)+facet_grid(~fireplaces)

#Theme-1

obj1<-ggplot(data=SaratogaHouses, aes(x=factor(rooms), y=price, fill=factor(rooms)))+ geom_boxplot()


obj1+labs(title = 'Price w.r.t rooms', x='rooms', fill='palegreen1')->obj2
obj2+ theme(panel.background = element_rect('palegreen1'))->obj3
obj3+theme(plot.title = element_text(hjust = 0.5, face = 'bold', colour = 'cadetblue'))->obj4
obj4+scale_y_continuous()


#Theme-2

obj1<-ggplot(data=SaratogaHouses, aes(x=factor(rooms), y=price, fill=factor(rooms)))+ geom_boxplot()

obj1+labs(title = 'Price w.r.t rooms', x='rooms', fill='palegreen1')->obj2
obj2+ theme(panel.background = element_rect('palegreen1'))->obj3
obj3+theme(plot.title = element_text(hjust = 0.5, face = 'bold', colour = 'cadetblue'))->obj4
obj4+scale_y_continuous()

#--------------------------------------------------------------
#--------Uniform distribution plot:---------------------
# define  x-axis
X<-seq(-4, 4, length=100)

# calculate  uniform  distribution probabilities
y<-dunif(X, min=-3, max=3)
plot(X,y, type='l', lwd='3', ylim=c(0,0.25), col='blue',
     xlab='x', ylab='Probability', main='Uniform Distribution plot')

#----------------------------------------------------------------------

#-----R plots tutorial for beginners:-------------------

library(tidyverse)

mpg
View(mpg)
attach(mpg)

ggplot(data=mpg)+geom_point(aes(x=displ, y=hwy))

ggplot(data=mpg)+geom_point(aes(x=displ, y=hwy,color=class))

ggplot(data=mpg)+geom_point(aes(x=displ, y=hwy,size=class))

ggplot(data=mpg)+geom_point(aes(x=displ, y=hwy, alpha=class))

ggplot(data=mpg)+geom_point(aes(x=displ, y=hwy, shape=class))

ggplot(data=mpg)+geom_point(aes(x=displ, y=hwy, color='blue'))

#-----Multiple plots using facet feature available in R ggplot

ggplot(data=mpg)+
  geom_point(aes(x=displ, y=hwy))+
  facet_wrap(~class, nrow=2)

ggplot(data=mpg)+
  geom_point(aes(x=displ, y=hwy))+
  facet_grid(drv~cyl)



ggplot(data=mpg)+geom_smooth(aes(x=displ, y=hwy))


ggplot(data=mpg)+geom_smooth(aes(x=displ, y=hwy, linetype=drv))

ggplot(data=mpg)+geom_smooth(aes(x=displ, y=hwy, color=drv,linetype=drv))

ggplot(data=mpg)+geom_smooth(aes(x=displ, y=hwy, color=drv,linetype=drv))+
  geom_point(aes(x=displ, y=hwy, color=drv))

#-----------
diamonds
View(diamonds)

ggplot(data=diamonds)+geom_bar(aes(x=cut))   # bar plot

ggplot(data=diamonds)+stat_count(aes(x=cut))   # bar plot

demo<- tribble(
  + ~cut, ~freq,
  + 'Fair', 1610,
  + 'Good', 4906,
  + 'Very Good', 12082,
  + 'Premium',   13791,
  + 'Ideal',     21551
  )
 
ggplot(data= diamonds)+ geom_bar(mapping= aes(x=cut))

ggplot(data= diamonds)+ geom_bar(mapping= aes(x=cut, color= cut))

ggplot(data= diamonds)+ geom_bar(mapping= aes(x=cut, fill= cut))
ggplot(data= diamonds)+ geom_bar(mapping= aes(x=cut, fill= clarity),
                                 position = 'dodge')

bar<- ggplot(data= diamonds)+ 
  geom_bar(mapping= aes(x=cut, fill= cut),
           show.legend = FALSE,
           width=1
           )+
  theme( aspect.ratio = 1)+
  labs( x= NULL, y= NULL)

bar

bar+coord_flip()

bar+coord_polar()


ggplot(data= diamonds)+ geom_bar(mapping= aes(x=cut, fill= clarity))


ggplot(data= diamonds)+ geom_bar(mapping= aes(x=cut, y=stat(prop), group=1))

ggplot(data=diamonds)+
  stat_summary(
    mapping = aes(x=cut, y=depth),
    fun.ymin = min,
    fun.ymax = max,
    fun.y =  median
  )


ggplot(data= mpg)+ geom_boxplot(mapping= aes(x=class, y=hwy))

ggplot(data= mpg)+ geom_boxplot(mapping= aes(x=class, y=hwy))+ coord_flip()

ggplot(data= mpg)+ geom_boxplot(mapping= aes(x=class, y=hwy, fill=class))+ coord_flip()

ggplot(data= mpg)+ geom_point(mapping= aes(x=displ, y=hwy))+ geom_smooth( aes(x=displ, y= hwy))

ggplot(data= mpg)+ geom_point(mapping= aes(x=displ, y=hwy, color=drv))+ geom_smooth( aes(x=displ, y= hwy))


# install.packages('maps')

#--------map in ggplot:--------

nz<- map_data('nz')

ggplot(nz, aes( long, lat, group= group))+
  geom_polygon( fill='white', colour= 'black')


ggplot(nz, aes( long, lat, group= group))+
  geom_polygon( fill='white', colour= 'black')+ coord_quickmap()



# install.packages('plotly')
library(ggplot2)

library(plotly)

d<- diamonds[sample(nrow(diamonds),1000),]

plot_ly(d, x= ~carat, y= ~price, color=~carat,
        size=~carat, text=~paste("Clarity", clarity))


#--------------------------------------------------------
#---- Mannually creating plot for any specific function:---------

x1<-seq(-10, 10, 0.01)
x2<-seq(0, 1, 0.01)

eq1 <-x1^2
eq2 <- (1+x2)^3

plot(x1, eq1, col="black")
par(new = TRUE)
plot(x2, eq2, type = "l", col = "green")


#-----------


#--------------------2nd plot:---------------------------------

#-------Creating Multiple plots using par (mfrow/ mfcol= c(2,2)):----------------

# --------Creating data vector:-----------
max.temp<-c('Sun'=22,'Mon'=27,'Tue'=26,'Wen'=24,'Thu'=23,'Fri'=26,'Sat'=28)

par(mfrow=c(1,2))    # set the plotting area into a 1*2 array
barplot(max.temp, main="Barplot")
pie(max.temp, main="Piechart", radius=1)

#---------------------------------

Temperature <- airquality$Temp
Ozon <- airquality$Ozone
par(mfrow=c(2,2))
hist(Temperature)
boxplot(Temperature, horzontal= TRUE)
hist(Ozon)
boxplot(Ozon, horizontal= TRUE)

#-------Now orientation:--------------------
par(mfcol=c(2,2))
hist(Temperature)
boxplot(Temperature, horzontal= TRUE)
hist(Ozon)
boxplot(Ozon, horizontal= TRUE)

#-------------------------------------------------


# make labels and margins smaller
par(cex=0.7, mai=c(0.1,0.1,0.2,0.1))
Temperature <- airquality$Temp
# define area for the histogram
par(fig=c(0.1,0.7,0.3,0.9))
hist(Temperature)
# define area for the boxplot
par(fig=c(0.8,1,0,1), new=TRUE)
boxplot(Temperature)
# define area for the stripchart
par(fig=c(0.1,0.67,0.1,0.25), new=TRUE)
stripchart(Temperature, method="jitter")

#---------------------------------------------------------------------------



# Piecwise function  (wring function from scratch)

f.t=function(t){
  if (t<2){
    t^2                     # result for the first case
  } else if (t==2){
    6                       # result for the second case
  } else if (t>2 & t<=6){
    10-t                   # result for fourth case
  } else if (t>6){
    print(" Error : Undefined")  # result for final case
  }
}

f.t(-3)    # display result for case 1
f.t(2)      # display result for case 2
f.t(3)       # display result for case 3
f.t(6.5)      # display result for case 4






















