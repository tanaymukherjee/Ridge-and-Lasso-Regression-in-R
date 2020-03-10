# Ridge and LAsso regression:

library(glmnet)
library(leaps)
library(ridge)
library(tidyr)
library(tidyverse)

data <- read.csv("C:\\Users\\its_t\\Documents\\CUNY\\Fall 2019\\9750 - Software Tools and Techniques_Data Science\\Regression_RnL.csv")
data <- data %>% select(1:30)

x <- model.matrix(data$LTCM_Net ~ ., data)[ , -1]
y <- data$LTCM_Net

grid <- 10^seq(10, -2, length = 100)
ridge.mod <- glmnet(x, y, alpha = 0, lambda = grid)
# ridge.mod <- glmnet(x, y, alpha = 0, lambda = grid, family = "binomial")

dim(coef(ridge.mod))

ridge.mod$lambda[50]
coef(ridge.mod)[ , 50]
sqrt(sum(coef(ridge.mod)[-1, 50]^2))

ridge.mod$lambda[60]
coef(ridge.mod)[ , 60]
sqrt(sum(coef(ridge.mod)[-1, 60]^2))

predict(ridge.mod, s = 50, type = "coefficients")[1:20, ]

set.seed(1)
train <- sample(1:nrow(x), nrow(x)/2)
test <- -train
y.test <- y[test]

ridge.mod <- glmnet(x[train, ], y[train], alpha = 0, lambda = grid, 
                    thresh = 1e-12)
ridge.pred <- predict(ridge.mod, s = 4, newx = x[test, ])
mean((ridge.pred - y.test)^2)

mean((mean(y[train]) - y.test)^2)

ridge.pred <- predict(ridge.mod, s = 1e10, newx = x[test, ])
mean((ridge.pred - y.test)^2)

# ridge.pred <- predict(ridge.mod, s = 0, newx = x[test, ], exact = TRUE)
ridge.pred <- predict(ridge.mod, s = 0, newx = x[test, ])

mean((ridge.pred - y.test)^2)
lm(y ~ x, subset = train)

# Predict ridge module
predict(ridge.mod, s = 0, type = "coefficients")[1:20, ]

# MSE plot v/s log of lambda
set.seed(1)
cv.out <- cv.glmnet(x[train, ], y[train], alpha = 0)
plot(cv.out)

# Getting the lambda value
bestlambda <- cv.out$lambda.min
bestlambda

ridge.pred <- predict(ridge.mod, s = bestlambda, newx = x[test, ])
mean((ridge.pred - y.test)^2)

out <- glmnet(x, y, alpha = 0)
predict(out, type = "coefficients", s = bestlambda)[1:20, ]


# Get the lasso regression
lasso.mod <- glmnet(x[train, ], y[train], alpha = 1, lambda = grid)
plot(lasso.mod)

set.seed(1)
cv.out <- cv.glmnet(x[train, ], y[train], alpha = 1)
plot(cv.out)

bestlambda <- cv.out$lambda.min
lasso.pred <- predict(lasso.mod, s = bestlambda, newx = x[test, ])
mean((lasso.pred - y.test)^2)

out <- glmnet(x, y, alpha = 1, lambda = grid)
lasso.coef <- predict(out, type = "coefficients", s = bestlambda)[1:20, ]
lasso.coef
lasso.coef[lasso.coef != 0]
