library(mice)

data <- airquality
data[4:10,3] <- rep(NA,7)
data[1:5,4] <- NA

data <- data[-c(5,6)]
summary(data)

data2 <- data
colnames(data2) <- c("Ozone", "Solar.R", "Wind", "T emp")

mice(data = data, seed = 1)
mice(data = data2)


# https://datascienceplus.com/imputing-missing-data-with-r-mice-package/
# https://stackoverflow.com/questions/54666548/imputation-using-mice-is-showing-error-in-parse