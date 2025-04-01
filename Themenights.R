### predictive modeling theme nights
## revamp of theme night models 3/26/25
## load required packages for EDA visualization
#install.packages("esquisse")
#install.packages("ggplot2")  # Ensure ggplot2 is installed
#install.packages("dplyr")    # Useful for data manipulation
#install.packages("readr")    # To read CSV files
#install.packages("skimr")    # For summary statistics
library(esquisse)

# Load Data
attach(Theme3)
summary(Theme3)
#View(Theme3)
library(dplyr)
Theme3$OPPONENTCAT<-as.factor(Theme3$OPPONENTCAT) 
Oldyears<-subset(Theme3,YEAR!="2025")
Newyears<-subset(Theme3,YEAR=="2025")
summary(Oldyears)


##EDA
# Function to run Kruskal-Wallis test for each variable
run_kruskal <- function(var, data) {
  formula <- as.formula(paste("TICKETS ~", var))
  test_result <- kruskal.test(formula, data = data)
  cat("\n------------------------\n")
  cat("Kruskal-Wallis Test for:", var, "\n")
  print(test_result)
}



# Run Kruskal-Wallis for all variables
variables <- c("YEAR", "OPPONENT", "ThemeCategory", "Promo", "Day_OF_WEEK",
               "Item", "TimeofDay", "TwoThemes","BRANDED","Popular","LocationInHomestand","Month","OPPONENTCAT")

for (var in variables) {
  run_kruskal(var, Oldyears)
}


#Weights for each factor
weight_event_score <- 0.6
weight_early_purchase <- 0.4
# Final popularity score with weighted average
final_popularity_score <- (weight_event_score * EVENTSCORE) + 
  (weight_early_purchase * (TIX35DAYSAFTER))/2
Theme3$final_popularity_score<-final_popularity_score
Oldyears<-subset(Theme3,YEAR!="2025")
Oldyears$TEMPAVGHIGH<-as.numeric(Oldyears$TEMPAVGHIGH)
Oldyears$TEMPAVGLOW<-as.numeric(Oldyears$TEMPAVGLOW)
Oldyears$ThemeNoShow<-as.numeric(Oldyears$ThemeNoShow)
View(Oldyears)
numeric_vars <- Oldyears %>%
  dplyr::select(final_popularity_score,MONTHUER,TEMPAVGHIGH,TEMPAVGLOW,ThemeNoShow)
str(numeric_vars)
# Compute Spearman correlation matrix
corr_matrix <- cor(numeric_vars, method = "spearman")
# Print correlation matrix
print(corr_matrix)
#install.packages("ggcorrplot")

# Plot Spearman correlation heatmap
#install.packages("ggcorrplot")
library(ggcorrplot)
ggcorrplot(corr_matrix, 
           method = "circle", 
           type = "lower", 
           lab = TRUE, 
           title = "Spearman Correlation Heatmap of Numerical Variables")


cor.test(Oldyears$TICKETS, Oldyears$EVENTSCORE, method = "spearman")

############################Random forest model#########################################
##Step 1. Prepare the data for predictive modeling after EDA 

# Select only the relevant columns and convert categorical variables to factors
Oldyears_clean <- Oldyears%>%
  dplyr::select(YEAR,TICKETS,Day_OF_WEEK,Item,ThemeCategory.1,TimeofDay,EVENTSCORE,Popular,Promo) %>%
  mutate(
    Item = as.factor(Item),
    Day_OF_WEEK = as.factor(Day_OF_WEEK),
    ThemeCategory.1=as.factor(ThemeCategory.1),
    Promo=as.factor(Promo),
    TimeofDay=as.factor(TimeofDay),
    Popular=as.factor(Popular))
Oldyears_clean <- Oldyears_clean %>%
  mutate(
    Bobblehead_Tuesday = ifelse(Item == "Bobblehead" & Day_OF_WEEK == "Tuesday", 1, 0)
  )
#View(Oldyears_clean)
# Install and load necessary package
#install.packages("ranger")
library(ranger)

# OPTIONAL: Set seed for reproducibility
set.seed(42)


# Assign observation weights based on YEAR
# 2024 = most relevant, 2023 = medium, 2022 = less relevant
Oldyears_clean$weight <- ifelse(Oldyears_clean$YEAR == 2024, 2,
                                ifelse(Oldyears_clean$YEAR == 2023, 1,
                                       0.5))  # 2022 gets less weight

#View(Oldyears_clean)
  # 🎯 STEP 2: Split Data (90% Train, 10% Test)

trainIndex <- createDataPartition(Oldyears_clean$TICKETS, p = 0.9, list = FALSE)
train_data <- Oldyears_clean[trainIndex, ]
test_data <- Oldyears_clean[-trainIndex, ]
train_weights <- train_data$weight
test_weights <- test_data$weight
# ----------------------------
# 📢 STEP 3: Feature Selection Using Boruta
# ----------------------------
#install.packages("Boruta")
library(Boruta)
set.seed(42)
boruta_result <- Boruta(TICKETS ~ ., data = train_data, doTrace = 0,maxRuns=200)
print(boruta_result)
# Get Confirmed Features Only
confirmed_features <- getSelectedAttributes(boruta_result, withTentative = TRUE)
confirmed_features <- c(confirmed_features, "TICKETS")  # Add target back in

# Filter Data to Only Include Important Features
train_data <- train_data[, confirmed_features]
test_data <- test_data[, confirmed_features]

# ----------------------------
# 🌲 STEP 4: Train Weighted Random Forest Model
# ----------------------------

rf_model_weighted <- ranger(
  TICKETS ~ ., 
  data = train_data,
  importance = "impurity",
  num.trees = 5000,
  case.weights = train_weights  # <- clearer, uses saved weights
)
# ----------------------------
# 🔮 STEP 5: Predict on Test Set
# ----------------------------

predictions_test <- predict(rf_model_weighted, data = test_data)$predictions

# ----------------------------
# 📏 STEP 6: Evaluate Model Performance
# ----------------------------

# Calculate RMSE and R²
rmse_test <- sqrt(mean((test_data$TICKETS - predictions_test)^2))
r_squared_test <- cor(test_data$TICKETS, predictions_test)^2

# Print results
cat("✅ Test RMSE:", round(rmse_test, 2), "\n")
cat("✅ Test R²:", round(r_squared_test, 3), "\n")


# ----------------------------
# 🎉 STEP 8: 2025 Predictions
# ----------------------------

# Prepare 2025 Data for Prediction
year_2025_data <- Newyears %>% 
  dplyr::select(Day_OF_WEEK, Item, ThemeCategory.1, Popular, Promo, EVENTSCORE,TimeofDay) %>%
  mutate(
    Item = as.factor(Item),
    Day_OF_WEEK = as.factor(Day_OF_WEEK),
    ThemeCategory.1 = as.factor(ThemeCategory.1),
    Popular = as.factor(Popular),
    TimeofDay=as.factor(TimeofDay),
    Promo=as.factor(Promo)
  )
year_2025_data <- year_2025_data %>%
  mutate(
    Bobblehead_Tuesday = ifelse(Item == "Bobblehead" & Day_OF_WEEK == "Tuesday", 1, 0)
  )

# Match factor levels between training and new data
year_2025_data$Day_OF_WEEK <- factor(year_2025_data$Day_OF_WEEK, levels = levels(train_data$Day_OF_WEEK))
year_2025_data$Item <- factor(year_2025_data$Item, levels = levels(train_data$Item))
year_2025_data$ThemeCategory.1 <- factor(year_2025_data$ThemeCategory.1, levels = levels(train_data$ThemeCategory.1))
year_2025_data$Popular <- factor(year_2025_data$Popular, levels = levels(train_data$Popular))

# Predict for 2025 Data
predictions_2025 <- predict(rf_model_weighted, data = year_2025_data)$predictions

# Add Predictions to DataFrame
Newyears$PredictedTix <- predictions_2025
View(Newyears)

# --------------------------------------
# STEP 9 BOOTSTRAPPING 2025 PREDICTIONS
# --------------------------------------
library(ranger)

set.seed(2025)
n_boot <- 100
bootstrap_preds <- matrix(NA, nrow = nrow(year_2025_data), ncol = n_boot)

for (i in 1:n_boot) {
  boot_index <- sample(1:nrow(train_data), replace = TRUE)
  boot_data <- train_data[boot_index, ]
  
  rf_boot <- ranger(
    TICKETS ~ ., 
    data = boot_data, 
    importance = "impurity",
    num.trees = 1000,
    case.weights = boot_data$weight
  )
  
  boot_pred <- predict(rf_boot, data = year_2025_data)$predictions
  bootstrap_preds[, i] <- boot_pred
}

# Create summary stats
pred_mean <- apply(bootstrap_preds, 1, mean)
pred_low <- apply(bootstrap_preds, 1, quantile, probs = 0.05)
pred_high <- apply(bootstrap_preds, 1, quantile, probs = 0.95)

# Add to Newyears
Newyears$PredictedTix_Mean <- pred_mean
Newyears$PredictedTix_Low <- pred_low
Newyears$PredictedTix_High <- pred_high

# Optional: Export to CSV
write.csv(Newyears, "C:/Users/rbrant/Desktop/predictions_bootstrap2.csv", row.names = FALSE)

# Optional: Visual check
View(Newyears)
summary(Newyears$PredictedTix_High - Newyears$PredictedTix_Low)

library(ggplot2)

ggplot(Newyears, aes(x = reorder(TimeofDay, PredictedTix_Mean), y = PredictedTix_Mean)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = PredictedTix_Low, ymax = PredictedTix_High), width = 0.2) +
  coord_flip() +
  labs(
    title = "🎟️ 2025 Predicted Ticket Sales by Theme Night (with 90% Range)",
    x = "Theme Night",
    y = "Predicted Tickets"
  ) +
  theme_minimal()

lm(TICKETS ~ (.)^2, data = train_data)
#bobblehead on tuesday important
