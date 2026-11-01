# IMPORT
install.packages("dplyr")
install.packages("xgboost")
install.packages("tensorflow")
install.packages("keras")
install.packages("tensorflow")
install.packages("randomForest")
install.packages("e1071")
install.packages("tseries")
install.packages("readxl") 

library(keras)
library(dplyr)
library(xgboost)
library(randomForest)
library(e1071)
library(ggplot2)
library(forecast)
library(tseries)
library(readxl)


#LOADING DATA SET
df_elec<-read_excel("Elec-train.xlsx",sheet = "Feuil1")
# ============================================================
#  GLOBAL FUNCTION
# ============================================================

## 1. METRICS

MAE <- function(actual, predicted) {
  mean(abs(actual - predicted), na.rm = TRUE)
}

RMSE <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2, na.rm = TRUE))
}

MAPE <- function(actual, predicted) {
  idx <- actual != 0
  mean(
    abs(
      (actual[idx] - predicted[idx]) /
        actual[idx]
    ),
    na.rm = TRUE
  ) * 100
}

## 2. MODEL EVAL
evaluate_forecast <- function(actual, predicted, model_name, fold) {
  data.frame(
    Model = model_name,
    Fold = fold,
    MAE = MAE(actual, predicted),
    RMSE = RMSE(actual, predicted),
    MAPE = MAPE(actual, predicted)
  )
}

#PREPROCESSING
## 1. Print head

head(df_elec)
## 2.Few statistic
summary(df_elec)

## 2.1 Columns type
sapply(df_elec, class)

##3. Checking missing Values
colSums(is.na(df_elec))

missing_values <- data.frame(
  Variable = names(df_elec),
  Manquants = colSums(is.na(df_elec)),
  Pourcentage = round(colMeans(is.na(df_elec)) * 100, 2)
)

missing_values <- missing_values[missing_values$Manquants > 0, ]
missing_values

##Note
"
There are 4795 entries and 3 variables. There is no missing value for the columns:Timestamp and Temp but 96 missing values for Power (kW).
It's normal because it that we have to predict.
The Observation recorded for each 15 min.
Some types is not in a right a format(Timestamp).
"
## 3.1. Transform Timestamp  to datetimes and renaming

df_elec$Timestamp <- as.POSIXct(
  df_elec$Timestamp,
  format = "%d/%m/%Y %H:%M"
)

str(df_elec)

names(df_elec) <- c(
  "Timestamp",
  "Power_kW",
  "Temp_C"
)
names(df_elec)
## 4.  Distribution 


library(ggplot2)
df_elec_no_na <- df_elec[!is.na(df_elec$Power_kW), ]

ggplot(df_elec_no_na, aes(x = Timestamp, y = `Power_kW`, group = 1)) +
  geom_line() +
  labs(
    title = "Power consumption over time",
    x = "Timestamp",
    y = "Power (kW)"
  ) +
  theme_minimal()

## 5. Focus on Two months(5760 Obs.)

ggplot(df_elec_no_na[1:103, ], aes(x = Timestamp, y = `Power_kW`, group = 1)) +
  geom_line() +
  scale_x_datetime(
    date_breaks = "1 days",
    date_labels = "%d/%m"
  ) +
  labs(
    title = "Power consumption over time",
    x = "Date",
    y = "Power (kW)"
  ) +
  theme_minimal()

## Note part PREPROCESSING
"
The dataset contains three variables: 
Timestamp, Power (kW), and Temp (°C). 
Our target variable is Power (kW).
This situation involves two aspects of the time series: 

1. Power depends only on Timestamp.
2. Power depends on both Timestamp and Temp (°C). 
The first case corresponds to a standard time series problem,
while the second is a more specific type of time series (a multivariate one, influenced by an exogenous variable).
"

# ANALYSE Case 1 (standard time series problem)
## Trend and seasonalities

### 1. ACF

tmp1<-acf(df_elec_no_na$Power_kW,type="cor",plot = FALSE)
plot(tmp1)

### 2. PACF
tmp2=pacf(df_elec_no_na$Power_kW,type="cor",plot = FALSE)
plot(tmp2)
### Note 1(ACF&PACF):
"
L'ACF montre une forte autocorrélation de la consommation électrique, avec une décroissance relativement lente des corrélations.
Cette structure suggère la présence d'une tendance ou d'une non-stationnarité dans la série. 
La PACF présente quant à elle un pic très important au premier retard, indiquant une forte dépendance de la consommation actuelle à la consommation observée 15 minutes auparavant.
De ce fait, on conclut qu'Aucune saisonnalité claire ne peut être identifiée à partir de ces graphiques, car l'analyse actuelle ne couvre que 36 retards, soit 9 heures. 
Il est donc nécessaire d'étendre l'ACF jusqu'à au moins 1 jour (96 retards) afin de rechercher une saisonnalité.
"
### Extension to several Days
two_days_acf<- acf(
  df_elec_no_na$Power_kW,
  lag.max = 192,
  main = "ACF - Power consumption"
)
plot(two_days_acf)
week_acf<-acf(
  df_elec_no_na$Power_kW,
  lag.max = 672,
  main = "ACF - Power consumption"
)
plot(week_acf)

### Note extension
"
L'analyse de l'ACF met en évidence une forte structure périodique dans la consommation électrique. 
On observe notamment un pic important autour du lag 96.
Étant donné que les observations sont réalisées toutes les 15 minutes, 96 observations correspondent à 24 heures.
La série présente donc une saisonnalité quotidienne clairement identifiable.
Le motif se répète également autour du lag 192, correspondant à deux jours, ce qui confirme la périodicité journalière.

Compte de tenu de la decroissance lente de l'acf,on en conclut que cette serie a:
- Une tendance
- Une saisonnalité journalière(soit 96 observations correspondent à 24 heures.)
"
### 3. Series decomposition
library(forecast)

power_ts <- ts(df_elec_no_na$Power_kW, frequency = 96)
decomp <- stl(power_ts, s.window = "periodic")

plot(decomp)
# Test de Ljung-Box
residus <- decomp$time.series[, "remainder"]
residus <- na.omit(residus)
Box.test(residus)

decomp <- decompose(power_ts, type = "additive")
plot(decomp)
# Test de Ljung-Box
Box.test(decomp$random)

decomp <- decompose(power_ts, type = "multiplicative")
plot(decomp)
# Test de Ljung-Box
Box.test(decomp$random)

### Note finale ANALYSE
"
La décomposition de la série Power_kW met en évidence une forte composante saisonnière, caractérisée par un motif qui se répète régulièrement au cours du temps.
Une composante de tendance est également présente relativement baissière(pas tres significatif), avec des fluctuations relativement lentes du niveau moyen de puissance.
Enfin, les résidus sont globalement centrés autour de zéro, bien que plusieurs valeurs extrêmes soient observées. Cependant, le test de Box-Pierce appliqué aux résidus indique une autocorrélation significative, avec une p-value inférieure à 2,2 × 10^-16.
Les résidus ne peuvent donc pas être considérés comme un bruit blanc. Alors une partie de la structure temporelle de la série reste présente après décomposition.
Cette analyse confirme ainsi que la série possède une structure temporelle importante, notamment une saisonnalité marquée, qui devra être prise en compte lors de la phase de modélisation et de comparaison des méthodes de prévision.

De fait, nous allons :

- Tester les modèles SARIMA,ETS,XGBoost,Holt-Winters,Random Forest, SVM, Neural Net et LSTM.

"
# MODELS
### 1. Spliting data.
n <- length(power_ts)
n_days <- floor(n / 96)

train_days <- floor(n_days * 0.70)
valid_days <- floor(n_days * 0.15)
test_days  <- n_days - train_days - valid_days

# Nb. Obs
n_train <- train_days * 96
n_valid <- valid_days * 96
n_test  <- test_days * 96
#Split dataset
train <- power_ts[1:n_train]
validation <- power_ts[(n_train + 1):(n_train + n_valid)]
test <- power_ts[(n_train + n_valid + 1):(n_train + n_valid + n_test)]

### 2. Visualyze the split data set
plot(power_ts, main = "Train / Validation / Test")
abline( v = time(power_ts)[n_train],lty = 2)
abline(v = time(power_ts)[n_train + n_valid],lty = 2)

# ============================================================
# 3. TIME SERIES CROSS-VALIDATION
# ============================================================

horizon <- 96       
n_folds <- 5         

## Taille initiale d'entraînement
initial_window <- length(train) - n_folds * horizon

cat("Initial training window :", initial_window, "\n")
cat("Forecast horizon :", horizon, "\n")
cat("Number of folds :", n_folds, "\n")

## Build folders
cv_folds <- list()
for (i in 1:n_folds) {
  train_end <- initial_window + (i - 1) * horizon
  valid_start <- train_end + 1
  valid_end <- train_end + horizon
  cv_folds[[i]] <- list(
    train_start = 1,
    train_end = train_end,
    valid_start = valid_start,
    valid_end = valid_end
  )
}

cv_folds

# ============================================================
## 4.0 Holt winters(trend+Seasonal)
# ============================================================

results_hw_add <- list()

for (i in 1:n_folds) {
  
  fold <- cv_folds[[i]]
  
  # Extraction du train
  train_fold <- power_ts[
    fold$train_start:fold$train_end
  ]
  
  # Extraction de la validation
  valid_fold <- power_ts[
    fold$valid_start:fold$valid_end
  ]
  
  # Restaurer la fréquence saisonnière
  train_fold <- ts(
    as.numeric(train_fold),
    frequency = 96
  )

  
  
  # Model Holt-Winters additif
  model <- HoltWinters(
    train_fold,
    seasonal = "additive"
  )
  
  # Forecast
  pred <- forecast(
    model,
    h = horizon
  )$mean
  
  results_hw_add[[i]] <- evaluate_forecast(
    actual = valid_fold,
    predicted = pred,
    model_name = "HW_additive",
    fold = i
  )
}

results_hw_add <- bind_rows(results_hw_add)
results_hw_add

#Mult
results_hw_mult <- list()

for (i in 1:n_folds) {
  
  fold <- cv_folds[[i]]
  
  train_fold <- power_ts[
    fold$train_start:fold$train_end
  ]
  
  valid_fold <- power_ts[
    fold$valid_start:fold$valid_end
  ]
  
  # Restaurer la fréquence saisonnière
  train_fold <- ts(
    as.numeric(train_fold),
    frequency = 96
  )
  # Model Holt-Winters multiplicatif
  model <- HoltWinters(
    train_fold,
    seasonal = "multiplicative"
  )
  
  # Forecast
  pred <- forecast(
    model,
    h = horizon
  )$mean
  
  results_hw_mult[[i]] <- evaluate_forecast(
    actual = valid_fold,
    predicted = pred,
    model_name = "HW_multiplicative",
    fold = i
  )
}

results_hw_mult <- bind_rows(results_hw_mult)
results_hw_mult
#Summary

results_hw <- bind_rows(
  results_hw_add,
  results_hw_mult
)

hw_summary <- results_hw %>%
  group_by(Model) %>%
  summarise(
    MAE = mean(MAE, na.rm = TRUE),
    RMSE = mean(RMSE, na.rm = TRUE),
    MAPE = mean(MAPE, na.rm = TRUE)
  ) %>%
  arrange(MAPE)
hw_summary


#Note
#'Le modèle multiplicatif obtient les erreurs les plus faibles sur les trois indicateurs. Son MAPE de 3,77 % signifie que les prévisions s'écartent en moyenne d'environ 3,77 % des valeurs réelles.
#'Ainsi, pour cette série Power_kW, le modèle Holt-Winters multiplicatif semble mieux adapté que le modèle additif. Il sera donc retenu comme le meilleur modèle HW pour l'évaluation finale sur le jeu de test.
# Model Holt-Winters multiplicatif


train_hw <- ts(
  as.numeric(train),
  frequency = 96
)

fit_hw <- HoltWinters(
  train_hw,
  seasonal = "multiplicative"
)

# 3. Prévoir la longueur du jeu de test
forecast_hw <- forecast(
  fit_hw,
  h = length(test)
)$mean

# 4. Convertir en numérique pour l'évaluation
forecast_hw <- as.numeric(forecast_hw)
test_validation <- as.numeric(validation)
test_actual <- as.numeric(test)

result_hw_validation <- evaluate_forecast(
  actual = test_validation,
  predicted = forecast_hw,
  model_name = "HW_multiplicative",
  fold = "Validation"
)
result_hw_test <- evaluate_forecast(
  actual = test_actual,
  predicted = forecast_hw,
  model_name = "HW_multiplicative",
  fold = "Test"
)

result_hw_test
result_hw_validation

# GRAPH
# TRAIN
train_plot <- data.frame(
  Index = 1:length(train),
  Valeur = as.numeric(train),
  Type = "Train"
)

# VALIDATION
valid_plot <- data.frame(
  Index = (length(train) + 1):
    (length(train) + length(validation)),
  Valeur = as.numeric(validation),
  Type = "Validation"
)

# TEST RÉEL
test_start <- length(train) + length(validation) + 1

test_plot <- data.frame(
  Index = test_start:
    (test_start + length(test) - 1),
  Valeur = as.numeric(test),
  Type = "Test réel"
)

# PRÉVISIONS HW SUR LE TEST
forecast_plot <- data.frame(
  Index = test_start:
    (test_start + length(forecast_hw) - 1),
  Valeur = as.numeric(forecast_hw),
  Type = "Prévision HW"
)

# COMBINAISON
plot_data <- rbind(
  train_plot,
  valid_plot,
  test_plot,
  forecast_plot
)

# PLOT
ggplot(plot_data, aes(x = Index, y = Valeur, colour = Type)) +
  geom_line(linewidth = 0.7) +
  labs(
    title = "Holt-Winter Mult Power_kW : Train, Validation, Test and Prédiction",
    x = "Observation",
    y = "Power (kW)",
    colour = ""
  ) +
  theme_minimal()

#'CONCLUSION HW
#'Les performances diminuent légèrement sur le test, mais restent bonnes.
#'le modèle Holt-Winters multiplicatif est actuellement le meilleur modèle de cette famille.
best_hw_result<-result_hw_test
best_hw_result
# ============================================================
## 4. ETS(trend+Seasonal)
# ============================================================

results_ets <- list()

for (i in 1:n_folds) {
  
  fold <- cv_folds[[i]]
  train_fold <- power_ts[
    fold$train_start:fold$train_end
  ]
  
  valid_fold <- power_ts[
    fold$valid_start:fold$valid_end
  ]
  


  model <- ets(train_fold)
  pred <- forecast(
    model,
    h = length(valid_fold)
  )$mean
  
  results_ets[[i]] <- evaluate_forecast(
    actual = as.numeric(valid_fold),
    predicted = as.numeric(pred),
    model_name = "ETS",
    fold = i
  )
}

results_ets <- bind_rows(results_ets)
results_ets

#' NOTE 1
#' This model makes too error with default params

# with trend case
ets_configs <- list(
  list(lambda = NULL, biasadj = FALSE),
  list(lambda = 0,    biasadj = FALSE),
  list(lambda = "auto", biasadj = FALSE),
  list(lambda = 0,    biasadj = TRUE),
  list(lambda = "auto", biasadj = TRUE))

results_ets_tuning <- list()

for (c in seq_along(ets_configs)) {
  
  for (i in 1:n_folds) {
    
    fold <- cv_folds[[i]]
    
    train_fold <- power_ts[
      fold$train_start:fold$train_end
    ]
    
    valid_fold <- power_ts[
      fold$valid_start:fold$valid_end
    ]
    
    config <- ets_configs[[c]]
    
    model <- tryCatch(
      {
        if (is.null(config$lambda)) {
          ets(
            train_fold,
            biasadj = config$biasadj
          )
        } else {
          ets(
            train_fold,
            lambda = config$lambda,
            biasadj = config$biasadj
          )
        }
      },
      error = function(e) NULL
    )
    
    if (is.null(model)) next
    
    pred <- forecast(
      model,
      h = length(valid_fold)
    )$mean
    
    results_ets_tuning[[length(results_ets_tuning) + 1]] <-
      evaluate_forecast(
        actual = valid_fold,
        predicted = pred,
        model_name = paste0(
          "ETS_lambda_", config$lambda,
          "_bias_", config$biasadj
        ),
        fold = i
      )
  }
}

results_ets_tuning <- bind_rows(results_ets_tuning)
#results_ets_tuning <- bind_rows(results_ets_tuning)
ets_summary <- results_ets_tuning %>%
  group_by(Model) %>%
  summarise(
    MAE = mean(MAE),
    RMSE = mean(RMSE),
    MAPE = mean(MAPE)
  ) %>%
  arrange(MAPE)

ets_summary
#Note2 :
"
The model ets(train_fold,lambda = 0,biasadj = TRUE) seem the best one of ets.
"

fit <- ets(
  train,
  lambda = 0,
  biasadj = TRUE
)

forecast_values <- forecast(
  fit,
  h = length(test)
)$mean


result_best <- evaluate_forecast(
  actual = test,
  predicted = forecast_values,
  model_name = "ETS_0_true",
  fold = "Test"
)

ets_summary_best <- result_best %>%
  group_by(Model) %>%
  summarise(
    MAE = mean(MAE, na.rm = TRUE),
    RMSE = mean(RMSE, na.rm = TRUE),
    MAPE = mean(MAPE, na.rm = TRUE)
  ) %>%
  arrange(MAPE)

ets_summary_best


# Graph
plot_data <- data.frame(
  Index = 1:length(test),
  Actual = as.numeric(test),
  Forecast = as.numeric(forecast_values)
)
library(ggplot2)

ggplot(plot_data, aes(x = Index)) +
  geom_line(aes(y = Actual, colour = "Réel"), linewidth = 0.8) +
  geom_line(aes(y = Forecast, colour = "Prévision ETS"), linewidth = 0.8) +
  labs(
    title = "ETS(λ = 0, biasadj = TRUE) : Réel vs Prévision",
    x = "Observation",
    y = "Power (kW)",
    colour = ""
  ) +
  theme_minimal()

#' CONCLUSION
#' This model generalization error is too large,that mean impossible to predict this time series.No one will not going to be selected.


# ============================================================
# 5. ARIMA BY CONSIDERING TREND+seasonal
# ============================================================

train_arima <- ts(
  as.numeric(train),
  frequency = 96
)

plot(train_arima,
     main = "Série Power_kW - Train",
     ylab = "Power (kW)",
     xlab = "Temps")
#Verification stationnarity
Box.test(train_arima)

#Note 1:
#' La série Power_kW présente une autocorrélation significative.


nsdiffs(train_arima, m = 96)
train_diff <- diff(train_arima, differences = 1,lag = 96)

#Verification stationnarity
par(mfrow = c(1, 2))

acf(
  train_diff,
  lag.max = 100,
  main = "ACF - Série différenciée"
)

pacf(
  train_diff,
  lag.max = 100,
  main = "PACF - Série différenciée"
)

par(mfrow = c(1, 1))

# Note : this diff is not suffisant

train_diff <- diff(train_arima, differences = 3,lag = 96)
ggtsdisplay(train_diff)

#' CONCLUSION ARIMA
#' L'analyse des fonctions d'autocorrélation et d'autocorrélation partielle met en évidence une forte saisonnalité aux retards 96, 192 et 288. La période de 96 observations correspond à une journée, puisque les données sont enregistrées toutes les 15 minutes. Cette structure confirme donc la présence d'une saisonnalité quotidienne importante dans la série.
#' Après application d'une différenciation saisonnière de période 96, l'ACF montre encore une autocorrélation importante aux premiers retards, tandis que la PACF présente notamment un pic marqué au retard 1.
#'  Cela indique qu'une composante non saisonnière reste présente dans la série et qu'une différenciation classique pourrait également être nécessaire.
#' Ainsi, l'analyse oriente vers un modèle SARIMA avec une période saisonnière de 96, avec une différenciation saisonnière D = 1 et potentiellement une différenciation non saisonnière d = 0.
#'  Les fonctions ACF et PACF permettront ensuite d'identifier les paramètres p, q, P et Q du modèle.

# ============================================================
# 5. SARIMA auto with default params
# ============================================================


#  SARIMA params to be tested

d <- ndiffs(power_ts)
D <- nsdiffs(power_ts, m = 96)

cat("d =", d, "\n")
cat("D =", D, "\n")


sarima_models <- list(
  list(
    order = c(1, d, 0),
    seasonal = c(1, D, 0),
    name = paste0("SARIMA(1,", d, ",0)(1,", D, ",0)[96]")
  ),
  
  list(
    order = c(0, d, 1),
    seasonal = c(1, D, 0),
    name = paste0("SARIMA(0,", d, ",1)(1,", D, ",0)[96]")
  ),
  
  list(
    order = c(1, d, 1),
    seasonal = c(1, D, 0),
    name = paste0("SARIMA(1,", d, ",1)(1,", D, ",0)[96]")
  ),
  
  list(
    order = c(1, d, 0),
    seasonal = c(0, D, 1),
    name = paste0("SARIMA(1,", d, ",0)(0,", D, ",1)[96]")
  ),
  
  list(
    order = c(0, d, 1),
    seasonal = c(0, D, 1),
    name = paste0("SARIMA(0,", d, ",1)(0,", D, ",1)[96]")
  ),
  
  list(
    order = c(1, d, 1),
    seasonal = c(0, D, 1),
    name = paste0("SARIMA(1,", d, ",1)(0,", D, ",1)[96]")
  )
)


# ---------------------------------------------------------
results_sarima <- list()

for (i in 1:n_folds) {
  
  cat("\n=====================================\n")
  cat("FOLD", i, "\n")
  cat("=====================================\n")
  
  fold <- cv_folds[[i]]
  
  train_fold <- power_ts[
    fold$train_start:fold$train_end
  ]
  
  valid_fold <- power_ts[
    fold$valid_start:fold$valid_end
  ]
  
  train_fold <- ts(
    as.numeric(train_fold),
    frequency = 96
  )
  
  
  # -------------------------------------------------------
  # TEST the 6th manual  MODELS
  
  for (m in sarima_models) {
    cat("\nTest :", m$name, "\n")
    result <- tryCatch({
      model <- Arima(
        train_fold,
        order = m$order,
        seasonal = list(
          order = m$seasonal,
          period = 96
        )
      )
      
      # Forcast
      pred <- forecast(
        model,
        h = length(valid_fold)
      )$mean
      
      # Evaluation
      evaluate_forecast(
        actual = as.numeric(valid_fold),
        predicted = as.numeric(pred),
        model_name = m$name,
        fold = i
      )
      
    }, error = function(e) {
      # Err
      cat(
        "ERREUR :", 
        conditionMessage(e), 
        "\n"
      )
      
      #  ERR Result
      data.frame(
        Model = m$name,
        Fold = i,
        MAE = NA_real_,
        RMSE = NA_real_,
        MAPE = NA_real_,
        Status = "ERR",
        Error = conditionMessage(e)
      )
    })
    
    
    # Stockage du résultat
    results_sarima[[length(results_sarima) + 1]] <- result
  }
}

# ---------------------------------------------------------
# Result consolidation

results_sarima <- do.call(
  rbind,
  results_sarima
)

rownames(results_sarima) <- NULL
# Display
results_sarima
#'NOTE SARIMA 1: 
#'Le meilleur modèle sur tes 5 folds est SARIMA(0,0,1)(0,1,1)[96]
#'
model_sarima <- Arima(
  train,
  order = c(0, 0, 1),
  seasonal = list(
    order = c(0, 1, 1),
    period = 96
  )
)

summary(model_sarima)
checkresiduals(model_sarima)
#' NOTE SARIMA 2: 
#' Le model n'est pas satifaisant comme le montre le test mais reste meilleur de la famille.

#' Global test
best_model_sarima <- Arima(
  train,
  order = c(0, 0, 1),
  seasonal = list(
    order = c(0, 1, 1),
    period = 96
  )
)
#'Pred on validation and test
pred_val <- forecast(
  best_model_sarima,
  h = length(validation)
)$mean

pred_test <- forecast(
  best_model_sarima,
  h = length(test)
)$mean

# Evaluation
evaluate_forecast(
  actual = as.numeric(validation),
  predicted = as.numeric(pred_val),
  model_name = m$name,
  fold = "Validation dataset"
)
evaluate_forecast(
  actual = as.numeric(test),
  predicted = as.numeric(pred_test),
  model_name = m$name,
  fold = "Test dataset"
)

#' NOTE  SARIMA 3: 
#' Bien qu'il fasse encore des erreurs mais il a un taux de reussite de plus de 95%. Les erreur commise peuvent être à une selection erronné de parametres.
#' Alors on fera un test automatique pour voir.


results_sarima_auto <- list()
models_sarima_auto <- list()
for (i in 1:n_folds) {
  
  fold <- cv_folds[[i]]
  train_fold <- power_ts[fold$train_start:fold$train_end]
  valid_fold <- power_ts[fold$valid_start:fold$valid_end]
  model <- auto.arima(
    train_fold,
    seasonal = TRUE,
    stepwise = FALSE,
    approximation = FALSE
  )
  
  pred <- forecast(
    model,
    h = horizon
  )$mean
  
  models_sarima_auto[[i]] <- model
  
  # Évaluation
  results_sarima_auto[[i]] <- evaluate_forecast(
    valid_fold,
    pred,
    "SARIMA",
    i
  )
}

results_sarima_auto <- bind_rows(results_sarima_auto)
results_sarima_auto

params_sarima <- data.frame(
  Fold = 1:n_folds,
  Model = sapply(models_sarima_auto, function(m) {
    
    o <- arimaorder(m)
    
    paste0(
      "SARIMA(",
      o[1], ",", o[2], ",", o[3], ")(",
      o[4], ",", o[5], ",", o[6], ")[",
      o[7], "]"
    )
  })
)

params_sarima
#' CONCLUSION SARIMA
#' La sélection automatique des modèles réalisée avec auto.arima() conduit principalement à des modèles non saisonniers.
#'Les quatre premiers folds retiennent un ARIMA(3,0,1), tandis que le cinquième fold sélectionne un ARIMA(3,0,2).
#'Malgré la présence d'une saisonnalité journalière identifiée précédemment à la période 96, l'algorithme ne retient pas de composante saisonnière dans cette procédure de sélection.
#'Les performances obtenues restent toutefois inférieures à celles des modèles SARIMA spécifiquement construits pour intégrer cette saisonnalité.
#'
#' Alors le meilleur modele de la famille est SARIMA(0,0,1)(0,1,1)[96]

# Index temporels
result_validation <- evaluate_forecast(
  actual = as.numeric(validation),
  predicted = as.numeric(pred_val),
  model_name = "SARIMA(0,0,1)(0,1,1)[96]",
  fold = "Validation dataset"
)

result_test <- evaluate_forecast(
  actual = as.numeric(test),
  predicted = as.numeric(pred_test),
  model_name = "SARIMA(0,0,1)(0,1,1)[96]",
  fold = "Test dataset"
)

time_train <- seq_along(train)
time_val <- length(train) + seq_along(validation)
time_test <- length(train) + length(validation) + seq_along(test)

time_pred_val <- time_val
time_pred_test <- time_test

df_train <- data.frame(
  Time = time_train,
  Power = as.numeric(train),
  Type = "Train"
)

df_val <- data.frame(
  Time = time_val,
  Power = as.numeric(validation),
  Type = "Validation"
)

df_test <- data.frame(
  Time = time_test,
  Power = as.numeric(test),
  Type = "Test"
)

df_pred_val <- data.frame(
  Time = time_pred_val,
  Power = as.numeric(pred_val),
  Type = "Validation prediction"
)

df_pred_test <- data.frame(
  Time = time_pred_test,
  Power = as.numeric(pred_test),
  Type = "Test prediction"
)

df_plot <- rbind(
  df_train,
  df_val,
  df_test,
  df_pred_val,
  df_pred_test
)

performance_text <- paste0(
  "Validation : MAE = ", round(result_validation$MAE, 2),
  " kW | RMSE = ", round(result_validation$RMSE, 2),
  " kW | MAPE = ", round(result_validation$MAPE, 2), "%\n",
  "Test : MAE = ", round(result_test$MAE, 2),
  " kW | RMSE = ", round(result_test$RMSE, 2),
  " kW | MAPE = ", round(result_test$MAPE, 2), "%"
)

ggplot(df_plot, aes(x = Time, y = Power, color = Type)) +
  geom_line(linewidth = 0.7) +
  annotate(
    "text",
    x = max(df_plot$Time) * 0.72,
    y = max(df_plot$Power) * 0.92,
    label = performance_text,
    hjust = 0,
    size = 4
  ) +
  labs(
    title = "Prediction SARIMA(0,0,1)(0,1,1)[96]",
    subtitle = "Validation et test",
    x = "Times",
    y = "Power (kW)",
    color = "Série"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "bottom"
  )

performance_text

# ============================================================
# 6. NEURAL NETWORK
# ============================================================

results_nn <- list()

for (i in 1:n_folds) {
  fold <- cv_folds[[i]]
  
  train_fold <- power_ts[fold$train_start:fold$train_end]
  valid_fold <- power_ts[fold$valid_start:fold$valid_end]
  model <- nnetar(
    train_fold,
    p = 96,
    P = 1,
    size = 10,
    repeats = 10
  )
  
  pred <- forecast(
    model,
    h = horizon
  )$mean
  
  results_nn[[i]] <- evaluate_forecast(
    valid_fold,
    pred,
    "Neural Net",
    i
  )
}

results_nn <- bind_rows(results_nn)
results_nn

# AutoSelect 
results_nn_auto <- list()
models_nn <- list()

for (i in 1:n_folds) {
  fold <- cv_folds[[i]]
  
  train_fold <- power_ts[fold$train_start:fold$train_end]
  valid_fold <- power_ts[fold$valid_start:fold$valid_end]
  
  model <- nnetar(
    train_fold,
    P = 1,
    repeats = 10
  )
  
  models_nn[[i]] <- model
  
  cat(
    "Fold", i,
    ": NNAR(",
    model$p, ",",
    model$size,
    ")[96]\n"
  )
  
  pred <- forecast(
    model,
    h = horizon
  )$mean
  
  results_nn_auto[[i]] <- evaluate_forecast(
    valid_fold,
    pred,
    "NNAR",
    i
  )
}

results_nn_auto <- bind_rows(results_nn_auto)
results_nn_auto

#' NOTE NNAR

#' Le NNAR manuel est nettement plus performant que la sélection automatique, avec des erreurs globalement plus faibles (MAPE de 2,05 % à 7,56 % contre 11,76 % à 17,76 % pour le NNAR automatique).
#' Le modèle manuel est donc retenu, car il offre une meilleure précision et une meilleure stabilité sur les cinq folds.

## PLOTTING

best_model_nn <- nnetar(
  train,
  p = 96,
  P = 1,
  size = 10,
  repeats = 10
)

pred_val_nn <- forecast(
  best_model_nn,
  h = length(validation)
)$mean

pred_test_nn <- forecast(
  best_model_nn,
  h = length(test)
)$mean

result_validation_nn <- evaluate_forecast(
  actual = as.numeric(validation),
  predicted = as.numeric(pred_val_nn),
  model_name = "NNAR(96,10)[96]",
  fold = "Validation dataset"
)

result_test_nn <- evaluate_forecast(
  actual = as.numeric(test),
  predicted = as.numeric(pred_test_nn),
  model_name = "NNAR(96,10)[96]",
  fold = "Test dataset"
)

result_validation_nn
result_test_nn

n_train <- length(train)
n_validation <- length(validation)
n_test <- length(test)

df_train_nn <- data.frame(
  Time = 1:n_train,
  Power = as.numeric(train),
  Type = "Train"
)

df_validation_nn <- data.frame(
  Time = (n_train + 1):(n_train + n_validation),
  Power = as.numeric(validation),
  Type = "Validation"
)

df_test_nn <- data.frame(
  Time = (n_train + n_validation + 1):
    (n_train + n_validation + n_test),
  Power = as.numeric(test),
  Type = "Test"
)

df_pred_val_nn <- data.frame(
  Time = (n_train + 1):(n_train + n_validation),
  Power = as.numeric(pred_val_nn),
  Type = "Validation prediction"
)

df_pred_test_nn <- data.frame(
  Time = (n_train + n_validation + 1):
    (n_train + n_validation + n_test),
  Power = as.numeric(pred_test_nn),
  Type = "Test prediction"
)

df_plot_nn <- rbind(
  df_train_nn,
  df_validation_nn,
  df_test_nn,
  df_pred_val_nn,
  df_pred_test_nn
)

ggplot(
  df_plot_nn,
  aes(x = Time, y = Power, color = Type)
) +
  geom_line(linewidth = 0.7) +
  labs(
    title = "Prédiction  NNAR(96,10)[96]",
    subtitle = "Train, validation and test",
    x = "Observations",
    y = "Power (kW)",
    color = ""
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "bottom"
  )

#SCORE 80% success
results_nn_global <- bind_rows(
  result_validation_nn,
  result_test_nn
)

results_nn_global


# ============================================================
## 7. CREATION DES LAGS POUR XGBOOST
# ============================================================

create_lag_data <- function(x, lags = 96) {
  df <- data.frame(y = as.numeric(x))
  for (lag in 1:lags) {
    df[[paste0("lag_", lag)]] <- dplyr::lag(
      df$y,
      lag
    )
  }
  df <- na.omit(df)
  return(df)
}

results_xgb <- list()

for (i in 1:n_folds) {
  fold <- cv_folds[[i]]
  train_fold <- power_ts[fold$train_start:fold$train_end]
  valid_fold <- power_ts[fold$valid_start:fold$valid_end]
  
  # Dataset 
  train_ml <- create_lag_data(
    train_fold,
    lags = 96
  )
  
  X_train <- as.matrix(
    train_ml[, -1]
  )
  y_train <- train_ml$y
  
  # Model
  model <- xgboost(
    data = X_train,
    label = y_train,
    nrounds = 200,
    max_depth = 6,
    eta = 0.05,
    objective = "reg:squarederror",
    verbose = 0
  )
  
  # 
  history <- as.numeric(train_fold)
  predictions <- numeric(horizon)
  
  for (j in 1:horizon) {
    x_input <- tail(history, 96)
    x_input <- rev(x_input)
    x_input <- matrix(
      x_input,
      nrow = 1
    )
    
    pred <- predict(
      model,
      x_input
    )
    
    predictions[j] <- pred
    history <- c(
      history,
      pred
    )
  }
  
  results_xgb[[i]] <- evaluate_forecast(
    valid_fold,
    predictions,
    "XGBoost",
    i
  )
}

results_xgb <- bind_rows(results_xgb)
results_xgb


## Tuning some params

### 1. max_depth et min_child_weight
results_xgb_step1 <- list()

grid_step1 <- expand.grid(
  max_depth = c(3, 5, 7, 9),
  min_child_weight = c(1, 3, 5)
)

for (g in 1:nrow(grid_step1)) {
  
  max_depth <- grid_step1$max_depth[g]
  min_child_weight <- grid_step1$min_child_weight[g]
  
  fold_results <- list()
  
  for (i in 1:n_folds) {
    
    fold <- cv_folds[[i]]
    
    train_fold <- power_ts[fold$train_start:fold$train_end]
    
    valid_fold <- power_ts[
      fold$valid_start:fold$valid_end
    ]
    
    train_ml <- create_lag_data(
      train_fold,
      lags = 96
    )
    
    X_train <- as.matrix(
      train_ml[, -1]
    )
    
    y_train <- train_ml$y
    
    model <- xgboost(
      data = X_train,
      label = y_train,
      nrounds = 200,
      max_depth = max_depth,
      min_child_weight = min_child_weight,
      eta = 0.3,
      objective = "reg:squarederror",
      verbose = 0
    )
    
    history <- as.numeric(train_fold)
    predictions <- numeric(horizon)
    
    for (j in 1:horizon) {
      
      x_input <- tail(history, 96)
      x_input <- rev(x_input)
      x_input <- matrix(
        x_input,
        nrow = 1
      )
      
      pred <- predict(
        model,
        x_input
      )
      
      predictions[j] <- pred
      
      history <- c(
        history,
        pred
      )
    }
    
    fold_results[[i]] <- evaluate_forecast(
      valid_fold,
      predictions,
      "XGBoost",
      i
    )
  }
  
  tmp <- bind_rows(fold_results)
  
  results_xgb_step1[[g]] <- data.frame(
    max_depth = max_depth,
    min_child_weight = min_child_weight,
    MAE = mean(tmp$MAE),
    RMSE = mean(tmp$RMSE),
    MAPE = mean(tmp$MAPE)
  )
}

results_xgb_step1 <- bind_rows(results_xgb_step1)

results_xgb_step1 <- results_xgb_step1[
  order(results_xgb_step1$RMSE),
]

results_xgb_step1
best_step1 <- results_xgb_step1[1, ]
best_step1

### 2. Regulation of our best found in step 1

best_max_depth <- best_step1$max_depth
best_min_child_weight <- best_step1$min_child_weight

grid_step2 <- expand.grid(
  reg_alpha = c(0, 0.1, 1),
  reg_lambda = c(1, 5, 10),
  subsample = c(0.7, 0.85, 1),
  colsample_bytree = c(0.7, 0.85, 1)
)

results_xgb_step2 <- list()

for (g in 1:nrow(grid_step2)) {
  
  reg_alpha <- grid_step2$reg_alpha[g]
  reg_lambda <- grid_step2$reg_lambda[g]
  subsample <- grid_step2$subsample[g]
  colsample_bytree <- grid_step2$colsample_bytree[g]
  
  fold_results <- list()
  
  for (i in 1:n_folds) {
    
    fold <- cv_folds[[i]]
    
    train_fold <- power_ts[
      fold$train_start:fold$train_end
    ]
    
    valid_fold <- power_ts[
      fold$valid_start:fold$valid_end
    ]
    
    train_ml <- create_lag_data(
      train_fold,
      lags = 96
    )
    
    X_train <- as.matrix(
      train_ml[, -1]
    )
    
    y_train <- train_ml$y
    
    model <- xgboost(
      data = X_train,
      label = y_train,
      nrounds = 200,
      max_depth = best_max_depth,
      min_child_weight = best_min_child_weight,
      eta = 0.3,
      reg_alpha = reg_alpha,
      reg_lambda = reg_lambda,
      subsample = subsample,
      colsample_bytree = colsample_bytree,
      objective = "reg:squarederror",
      verbose = 0
    )
    
    history <- as.numeric(train_fold)
    predictions <- numeric(horizon)
    
    for (j in 1:horizon) {
      
      x_input <- tail(history, 96)
      x_input <- rev(x_input)
      x_input <- matrix(
        x_input,
        nrow = 1
      )
      
      pred <- predict(
        model,
        x_input
      )
      
      predictions[j] <- pred
      
      history <- c(
        history,
        pred
      )
    }
    
    fold_results[[i]] <- evaluate_forecast(
      valid_fold,
      predictions,
      "XGBoost",
      i
    )
  }
  
  tmp <- bind_rows(fold_results)
  
  results_xgb_step2[[g]] <- data.frame(
    reg_alpha = reg_alpha,
    reg_lambda = reg_lambda,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    MAE = mean(tmp$MAE),
    RMSE = mean(tmp$RMSE),
    MAPE = mean(tmp$MAPE)
  )
}

results_xgb_step2 <- bind_rows(results_xgb_step2)

results_xgb_step2 <- results_xgb_step2[
  order(results_xgb_step2$RMSE),
]

results_xgb_step2

best_step2 <- results_xgb_step2[1, ]
best_step2

### 3. Observation

best_reg_alpha <- best_step2$reg_alpha
best_reg_lambda <- best_step2$reg_lambda
best_subsample <- best_step2$subsample
best_colsample_bytree <- best_step2$colsample_bytree

grid_step3 <- expand.grid(
  eta = c(0.01, 0.03, 0.05, 0.1),
  nrounds = c(100, 200, 300, 500)
)

results_xgb_step3 <- list()

for (g in 1:nrow(grid_step3)) {
  
  eta <- grid_step3$eta[g]
  nrounds <- grid_step3$nrounds[g]
  
  fold_results <- list()
  
  for (i in 1:n_folds) {
    
    fold <- cv_folds[[i]]
    
    train_fold <- power_ts[
      fold$train_start:fold$train_end
    ]
    
    valid_fold <- power_ts[
      fold$valid_start:fold$valid_end
    ]
    
    train_ml <- create_lag_data(
      train_fold,
      lags = 96
    )
    
    X_train <- as.matrix(
      train_ml[, -1]
    )
    
    y_train <- train_ml$y
    
    model <- xgboost(
      data = X_train,
      label = y_train,
      nrounds = nrounds,
      max_depth = best_max_depth,
      min_child_weight = best_min_child_weight,
      eta = eta,
      reg_alpha = best_reg_alpha,
      reg_lambda = best_reg_lambda,
      subsample = best_subsample,
      colsample_bytree = best_colsample_bytree,
      objective = "reg:squarederror",
      verbose = 0
    )
    
    history <- as.numeric(train_fold)
    predictions <- numeric(horizon)
    
    for (j in 1:horizon) {
      
      x_input <- tail(history, 96)
      x_input <- rev(x_input)
      x_input <- matrix(
        x_input,
        nrow = 1
      )
      
      pred <- predict(
        model,
        x_input
      )
      
      predictions[j] <- pred
      
      history <- c(
        history,
        pred
      )
    }
    
    fold_results[[i]] <- evaluate_forecast(
      valid_fold,
      predictions,
      "XGBoost",
      i
    )
  }
  
  tmp <- bind_rows(fold_results)
  
  results_xgb_step3[[g]] <- data.frame(
    eta = eta,
    nrounds = nrounds,
    MAE = mean(tmp$MAE),
    RMSE = mean(tmp$RMSE),
    MAPE = mean(tmp$MAPE)
  )
}

results_xgb_step3 <- bind_rows(results_xgb_step3)

results_xgb_step3 <- results_xgb_step3[
  order(results_xgb_step3$RMSE),
]

results_xgb_step3

best_xgb <- results_xgb_step3[1, ]
best_xgb

# NOTE
#' According this, the best model should have this params: max_depth = 7,min_child_weight = 5,reg_alpha = 0,reg_lambda = 1,subsample = 1,colsample_bytree = 1

best_xgb_model <- xgboost(
  data = as.matrix(create_lag_data(train, lags = 96)[, -1]),
  label = create_lag_data(train, lags = 96)$y,
  nrounds = 200,
  max_depth = 7,
  min_child_weight = 5,
  eta = 0.3,
  reg_alpha = 0,
  reg_lambda = 1,
  subsample = 1,
  colsample_bytree = 1,
  objective = "reg:squarederror",
  verbose = 0
)

train_ml <- create_lag_data(train, lags = 96)

pred_train_xgb <- predict(
  best_xgb_model,
  as.matrix(train_ml[, -1])
)

history <- as.numeric(train)
pred_val_xgb <- numeric(length(validation))

for (j in 1:length(validation)) {
  x_input <- tail(history, 96)
  x_input <- rev(x_input)
  x_input <- matrix(x_input, nrow = 1)
  
  pred <- predict(best_xgb_model, x_input)
  
  pred_val_xgb[j] <- pred
  history <- c(history, pred)
}

history <- c(
  as.numeric(train),
  as.numeric(validation)
)

pred_test_xgb <- numeric(length(test))

for (j in 1:length(test)) {
  x_input <- tail(history, 96)
  x_input <- rev(x_input)
  x_input <- matrix(x_input, nrow = 1)
  
  pred <- predict(best_xgb_model, x_input)
  
  pred_test_xgb[j] <- pred
  history <- c(history, pred)
}

result_train_xgb <- evaluate_forecast(
  actual = as.numeric(train)[97:length(train)],
  predicted = pred_train_xgb,
  model_name = "XGBoost",
  fold = "Train dataset"
)

result_val_xgb <- evaluate_forecast(
  actual = as.numeric(validation),
  predicted = pred_val_xgb,
  model_name = "XGBoost",
  fold = "Validation dataset"
)

result_test_xgb <- evaluate_forecast(
  actual = as.numeric(test),
  predicted = pred_test_xgb,
  model_name = "XGBoost",
  fold = "Test dataset"
)

results_xgb_global <- bind_rows(
  result_train_xgb,
  result_val_xgb,
  result_test_xgb
)

results_xgb_global

n_train <- length(train)
n_val <- length(validation)
n_test <- length(test)

df_train_xgb <- data.frame(
  Time = 97:n_train,
  Power = as.numeric(train)[97:n_train],
  Type = "Train"
)

df_pred_train_xgb <- data.frame(
  Time = 97:n_train,
  Power = as.numeric(pred_train_xgb),
  Type = "pred Train"
)

df_val_xgb <- data.frame(
  Time = (n_train + 1):(n_train + n_val),
  Power = as.numeric(validation),
  Type = "Validation"
)

df_pred_val_xgb <- data.frame(
  Time = (n_train + 1):(n_train + n_val),
  Power = as.numeric(pred_val_xgb),
  Type = "pred Validation"
)

df_test_xgb <- data.frame(
  Time = (n_train + n_val + 1):(n_train + n_val + n_test),
  Power = as.numeric(test),
  Type = "Test "
)

df_pred_test_xgb <- data.frame(
  Time = (n_train + n_val + 1):(n_train + n_val + n_test),
  Power = as.numeric(pred_test_xgb),
  Type = "pred Test"
)

df_plot_xgb <- rbind(
  df_train_xgb,
  df_pred_train_xgb,
  df_val_xgb,
  df_pred_val_xgb,
  df_test_xgb,
  df_pred_test_xgb
)

ggplot(
  df_plot_xgb,
  aes(x = Time, y = Power, color = Type)
) +
  geom_line(linewidth = 0.7) +
  labs(
    title = "Prediction XGBoost",
    subtitle = "Training set, validation set  and test set",
    x = "Observations",
    y = "Power (kW)",
    color = ""
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "bottom"
  )
# CONCLUSION
#' The model score is 79%


# ============================================================
# 8. PREPARATION LSTM
# ============================================================

create_sequences <- function(data, lookback = 96) {
  
  X <- list()
  y <- list()
  
  for (i in (lookback + 1):length(data)) {
    X[[length(X) + 1]] <- data[(i - lookback):(i - 1)]
    y[[length(y) + 1]] <- data[i]
  }
  
  X <- array(
    unlist(X),
    dim = c(
      length(X),
      lookback,
      1
    )
  )
  
  y <- unlist(y)
  
  list(
    X = X,
    y = y
  )
}

# Run LSTM
library(reticulate)
py_require("tensorflow")
library(tensorflow)
py_config()
tf$config$list_physical_devices("CPU")
tensorflow::tf_version()


results_lstm <- list()
for (i in 1:n_folds) {
  
  fold <- cv_folds[[i]]
  train_fold <- power_ts[fold$train_start:fold$train_end]
  
  valid_fold <- power_ts[
    fold$valid_start:fold$valid_end
  ]
  
  train_values <- as.numeric(train_fold)
  
  # Normalisation basée UNIQUEMENT sur le train
  train_min <- min(train_values)
  train_max <- max(train_values)
  
  train_scaled <- (train_values - train_min) / (train_max - train_min)
  
  sequences <- create_sequences(
    train_scaled,
    lookback = 96
  )
  
  model <- keras_model_sequential() %>%
    layer_lstm(
      units = 50,
      input_shape = c(96, 1)
    ) %>%
    layer_dense(
      units = 1
    )
  
  model %>% compile(
    optimizer = "adam",
    loss = "mse"
  )
  
  model %>% fit(
    sequences$X,
    sequences$y,
    epochs = 20,
    batch_size = 32,
    verbose = 0
  )
  
  history <- tail(
    train_scaled,
    96
  )
  
  predictions_scaled <- numeric(horizon)
  for (j in 1:horizon) {
    x_input <- array(
      history,
      dim = c(1, 96, 1)
    )
    
    pred <- model %>%
      predict(
        x_input,
        verbose = 0
      )
    
    pred <- as.numeric(pred)
    predictions_scaled[j] <- pred
    history <- c(
      history[-1],
      pred
    )
  }
  
  # Retour à l'échelle originale
  predictions <- (
    predictions_scaled *
      (train_max - train_min)
  ) + train_min
  
  results_lstm[[i]] <- evaluate_forecast(
    valid_fold,
    predictions,
    "LSTM",
    i
  )
}

results_lstm <- bind_rows(results_lstm)
results_lstm

# ============================================================
# 9. RANDOM FOREST
# ============================================================

results_rf <- list()

for (i in 1:n_folds) {
  fold <- cv_folds[[i]]
  train_fold <- power_ts[fold$train_start:fold$train_end]
  
  valid_fold <- power_ts[fold$valid_start:fold$valid_end]
  
  train_ml <- create_lag_data(
    train_fold,
    lags = 96
  )
  
  model <- randomForest(
    y ~ .,
    data = train_ml,
    ntree = 500,
    mtry = 10
  )
  
  history <- as.numeric(train_fold)
  predictions <- numeric(horizon)
  
  for (j in 1:horizon) {
    
    x_input <- data.frame(
      matrix(
        rev(tail(history, 96)),
        nrow = 1
      )
    )
    
    names(x_input) <- paste0(
      "lag_",
      1:96
    )
    
    pred <- predict(
      model,
      x_input
    )
    
    predictions[j] <- pred
    history <- c(
      history,
      pred
    )
  }
  
  results_rf[[i]] <- evaluate_forecast(
    valid_fold,
    predictions,
    "Random Forest",
    i
  )
}

results_rf <- bind_rows(results_rf)
results_rf
#'
#'


train_ml <- create_lag_data(
  train,
  lags = 96
)

X_train <- as.matrix(
  train_ml[, -1]
)

y_train <- train_ml$y

best_rf_model <- randomForest(
  x = X_train,
  y = y_train,
  ntree = 500,
  mtry = floor(sqrt(96)),
  nodesize = 5
)

pred_train_rf <- predict(
  best_rf_model,
  X_train
)

history <- as.numeric(train)

pred_val_rf <- numeric(length(validation))

for (j in 1:length(validation)) {
  
  x_input <- tail(history, 96)
  x_input <- rev(x_input)
  x_input <- matrix(x_input, nrow = 1)
  
  pred <- predict(
    best_rf_model,
    x_input
  )
  
  pred_val_rf[j] <- pred
  
  history <- c(
    history,
    pred
  )
}

history <- c(
  as.numeric(train),
  as.numeric(validation)
)

pred_test_rf <- numeric(length(test))

for (j in 1:length(test)) {
  
  x_input <- tail(history, 96)
  x_input <- rev(x_input)
  x_input <- matrix(x_input, nrow = 1)
  
  pred <- predict(
    best_rf_model,
    x_input
  )
  
  pred_test_rf[j] <- pred
  
  history <- c(
    history,
    pred
  )
}

result_train_rf <- evaluate_forecast(
  actual = as.numeric(train)[97:length(train)],
  predicted = pred_train_rf,
  model_name = "Random Forest",
  fold = "Train dataset"
)

result_val_rf <- evaluate_forecast(
  actual = as.numeric(validation),
  predicted = pred_val_rf,
  model_name = "Random Forest",
  fold = "Validation dataset"
)

result_test_rf <- evaluate_forecast(
  actual = as.numeric(test),
  predicted = pred_test_rf,
  model_name = "Random Forest",
  fold = "Test dataset"
)

results_rf_global <- bind_rows(
  result_train_rf,
  result_val_rf,
  result_test_rf
)

results_rf_global

n_train <- length(train)
n_val <- length(validation)
n_test <- length(test)

df_train_rf <- data.frame(
  Time = 97:n_train,
  Power = as.numeric(train)[97:n_train],
  Type = "Train"
)

df_pred_train_rf <- data.frame(
  Time = 97:n_train,
  Power = as.numeric(pred_train_rf),
  Type = "pred Train"
)

df_val_rf <- data.frame(
  Time = (n_train + 1):(n_train + n_val),
  Power = as.numeric(validation),
  Type = "Validation"
)

df_pred_val_rf <- data.frame(
  Time = (n_train + 1):(n_train + n_val),
  Power = as.numeric(pred_val_rf),
  Type = "pred Validation "
)

df_test_rf <- data.frame(
  Time = (n_train + n_val + 1):(n_train + n_val + n_test),
  Power = as.numeric(test),
  Type = "Test"
)

df_pred_test_rf <- data.frame(
  Time = (n_train + n_val + 1):(n_train + n_val + n_test),
  Power = as.numeric(pred_test_rf),
  Type = " pred Test "
)

df_plot_rf <- bind_rows(
  df_train_rf,
  df_pred_train_rf,
  df_val_rf,
  df_pred_val_rf,
  df_test_rf,
  df_pred_test_rf
)

ggplot(
  df_plot_rf,
  aes(x = Time, y = Power, color = Type)
) +
  geom_line(linewidth = 0.7) +
  labs(
    title = "Prédiction Random Forest",
    subtitle = "Training set , validation set and test test",
    x = "Observations",
    y = "Power (kW)",
    color = ""
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "bottom"
  )


# ============================================================
# 10. SVM
# ============================================================

results_svm <- list()

for (i in 1:n_folds) {
  
  fold <- cv_folds[[i]]
  train_fold <- power_ts[fold$train_start:fold$train_end]
  
  valid_fold <- power_ts[fold$valid_start:fold$valid_end]
  
  train_ml <- create_lag_data(
    train_fold,
    lags = 96
  )
  
  model <- svm(
    y ~ .,
    data = train_ml,
    kernel = "radial",
    cost = 10,
    gamma = 0.01,
    epsilon = 0.1
  )
  
  history <- as.numeric(train_fold)
  predictions <- numeric(horizon)
  
  for (j in 1:horizon) {
    
    x_input <- data.frame(
      matrix(
        rev(tail(history, 96)),
        nrow = 1
      )
    )
    
    names(x_input) <- paste0(
      "lag_",
      1:96
    )
    
    pred <- predict(
      model,
      x_input
    )
    
    predictions[j] <- pred
    history <- c(
      history,
      pred
    )
  }
  
  results_svm[[i]] <- evaluate_forecast(
    valid_fold,
    predictions,
    "SVM",
    i
  )
}

results_svm <- bind_rows(results_svm)
results_svm

# CONSOLIDATION

#
all_results <- bind_rows(
  results_ets,
  best_hw_result
  results_sarima,
  results_nn,
  results_xgb,
  results_rf,
  results_svm,
  results_lstm
)
all_results

# NOTE MODEL
"
According the result,we can classify as:
1.Random Forest
MAE = 7,09
RMSE = 11,78
MAPE = 3,43 %

2.XGBoost
MAE = 7,83
RMSE = 12,26
MAPE = 3,82 %

3.SVM
MAE = 9,23
RMSE = 13,37
MAPE = 4,51 %

4.Neural Network
MAE = 11,00
RMSE = 18,24
MAPE = 4,97 %

5. SARIMA
MAE = 34,88
RMSE = 42,42
MAPE = 16,60 %

6. ETS
MAE = 72,74
RMSE = 90,20
MAPE = 27,96 %

The cross-validation results show a clear performance gap between the forecasting methods.
Among the models evaluated so far, the machine learning approaches achieve substantially lower forecasting errors than the classical time-series models.
Random Forest obtains the best overall performance, with an average MAE of 7.09, RMSE of 11.78 and MAPE of 3.43% across the five validation folds. XGBoost follows closely, with a MAPE of 3.82%, while SVM and the Neural Network achieve MAPEs of 4.51% and 4.97%, respectively.
SARIMA and ETS perform considerably worse, with MAPEs of 16.60% and 27.96%.
However, the final model selection cannot yet be established because the LSTM model still needs to be evaluated under the same cross-validation procedure.


"


#'================================================================================
#' ------------------- Modélisation multivariée -----------------------------------------
#'===============================================================================
# ============================================================
## 1. CREATION DES LAGS POUR XGBOOST
# ============================================================

names(df_elec_no_na)

create_lag_data_m <- function(x, lags = 96) {
  
  # Si x est une liste-column
  if (is.list(x)) {
    x <- unlist(x, use.names = FALSE)
  }
  
  # Conversion numérique
  x <- as.numeric(x)
  
  # Création du dataframe
  df <- data.frame(y = x)
  
  # Création des lags
  for (lag in 1:lags) {
    df[[paste0("lag_", lag)]] <- dplyr::lag(
      df$y,
      lag
    )
  }
  
  # Suppression des NA
  df <- na.omit(df)
  
  return(df)
}


results_xgb_m <- list()

for (i in 1:n_folds) {
  
  cat("\n========================\n")
  cat("FOLD :", i, "\n")
  cat("========================\n")
  
  fold <- cv_folds[[i]]
  
  train_fold <- df_elec_no_na[
    fold$train_start:fold$train_end,
    "Power_kW"
  ]
  
  valid_fold <- df_elec_no_na[
    fold$valid_start:fold$valid_end,
    "Power_kW"
  ]
  
  train_fold <- as.numeric(unlist(train_fold, use.names = FALSE))
  valid_fold <- as.numeric(unlist(valid_fold, use.names = FALSE))
  
  cat("Train :", length(train_fold), "observations\n")
  cat("Valid :", length(valid_fold), "observations\n")
  
  train_ml <- create_lag_data_m(
    train_fold,
    lags = 96
  )
  
  cat(
    "Dataset ML :",
    nrow(train_ml),
    "lignes x",
    ncol(train_ml),
    "colonnes\n"
  )
  
  X_train <- as.matrix(
    train_ml[, -1]
  )
  
  y_train <- train_ml$y
  
  model <- xgboost(
    x = X_train,
    y = y_train,
    nrounds = 200,
    max_depth = 6,
    learning_rate = 0.05,
    objective = "reg:squarederror"
  )
  
  cat("Modèle XGBoost entraîné\n")
  
  history <- train_fold
  horizon <- length(valid_fold)
  predictions <- numeric(horizon)
  
  for (j in 1:horizon) {
    
    x_input <- tail(history, 96)
    x_input <- rev(x_input)
    
    x_input <- matrix(
      x_input,
      nrow = 1
    )
    
    colnames(x_input) <- colnames(X_train)
    
    pred <- predict(
      model,
      x_input
    )
    
    predictions[j] <- pred
    history <- c(history, pred)
  }
  
  cat(
    "Prévisions générées :",
    length(predictions),
    "\n"
  )
  
  results_xgb_m[[i]] <- evaluate_forecast(
    valid_fold,
    predictions,
    "XGBoost",
    i
  )
  
  cat("Fold", i, "terminé\n")
}

results_xgb_m <- dplyr::bind_rows(
  results_xgb_m
)

print(results_xgb_m)

## Tuning some params

### 1. max_depth et min_child_weight
results_xgb_step1 <- list()

grid_step1 <- expand.grid(
  max_depth = c(3, 5, 7, 9),
  min_child_weight = c(1, 3, 5)
)

for (g in 1:nrow(grid_step1)) {
  
  max_depth <- grid_step1$max_depth[g]
  min_child_weight <- grid_step1$min_child_weight[g]
  
  fold_results <- list()
  
  for (i in 1:n_folds) {
    
    fold <- cv_folds[[i]]
    
    train_fold <- power_ts[fold$train_start:fold$train_end]
    
    valid_fold <- power_ts[
      fold$valid_start:fold$valid_end
    ]
    
    train_ml <- create_lag_data(
      train_fold,
      lags = 96
    )
    
    X_train <- as.matrix(
      train_ml[, -1]
    )
    
    y_train <- train_ml$y
    
    model <- xgboost(
      data = X_train,
      label = y_train,
      nrounds = 200,
      max_depth = max_depth,
      min_child_weight = min_child_weight,
      eta = 0.3,
      objective = "reg:squarederror",
      verbose = 0
    )
    
    history <- as.numeric(train_fold)
    predictions <- numeric(horizon)
    
    for (j in 1:horizon) {
      
      x_input <- tail(history, 96)
      x_input <- rev(x_input)
      x_input <- matrix(
        x_input,
        nrow = 1
      )
      
      pred <- predict(
        model,
        x_input
      )
      
      predictions[j] <- pred
      
      history <- c(
        history,
        pred
      )
    }
    
    fold_results[[i]] <- evaluate_forecast(
      valid_fold,
      predictions,
      "XGBoost",
      i
    )
  }
  
  tmp <- bind_rows(fold_results)
  
  results_xgb_step1[[g]] <- data.frame(
    max_depth = max_depth,
    min_child_weight = min_child_weight,
    MAE = mean(tmp$MAE),
    RMSE = mean(tmp$RMSE),
    MAPE = mean(tmp$MAPE)
  )
}

results_xgb_step1 <- bind_rows(results_xgb_step1)

results_xgb_step1 <- results_xgb_step1[
  order(results_xgb_step1$RMSE),
]

results_xgb_step1
best_step1 <- results_xgb_step1[1, ]
best_step1

### 2. Regulation of our best found in step 1

best_max_depth <- best_step1$max_depth
best_min_child_weight <- best_step1$min_child_weight

grid_step2 <- expand.grid(
  reg_alpha = c(0, 0.1, 1),
  reg_lambda = c(1, 5, 10),
  subsample = c(0.7, 0.85, 1),
  colsample_bytree = c(0.7, 0.85, 1)
)

results_xgb_step2 <- list()

for (g in 1:nrow(grid_step2)) {
  
  reg_alpha <- grid_step2$reg_alpha[g]
  reg_lambda <- grid_step2$reg_lambda[g]
  subsample <- grid_step2$subsample[g]
  colsample_bytree <- grid_step2$colsample_bytree[g]
  
  fold_results <- list()
  
  for (i in 1:n_folds) {
    
    fold <- cv_folds[[i]]
    
    train_fold <- power_ts[
      fold$train_start:fold$train_end
    ]
    
    valid_fold <- power_ts[
      fold$valid_start:fold$valid_end
    ]
    
    train_ml <- create_lag_data(
      train_fold,
      lags = 96
    )
    
    X_train <- as.matrix(
      train_ml[, -1]
    )
    
    y_train <- train_ml$y
    
    model <- xgboost(
      data = X_train,
      label = y_train,
      nrounds = 200,
      max_depth = best_max_depth,
      min_child_weight = best_min_child_weight,
      eta = 0.3,
      reg_alpha = reg_alpha,
      reg_lambda = reg_lambda,
      subsample = subsample,
      colsample_bytree = colsample_bytree,
      objective = "reg:squarederror",
      verbose = 0
    )
    
    history <- as.numeric(train_fold)
    predictions <- numeric(horizon)
    
    for (j in 1:horizon) {
      
      x_input <- tail(history, 96)
      x_input <- rev(x_input)
      x_input <- matrix(
        x_input,
        nrow = 1
      )
      
      pred <- predict(
        model,
        x_input
      )
      
      predictions[j] <- pred
      
      history <- c(
        history,
        pred
      )
    }
    
    fold_results[[i]] <- evaluate_forecast(
      valid_fold,
      predictions,
      "XGBoost",
      i
    )
  }
  
  tmp <- bind_rows(fold_results)
  
  results_xgb_step2[[g]] <- data.frame(
    reg_alpha = reg_alpha,
    reg_lambda = reg_lambda,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    MAE = mean(tmp$MAE),
    RMSE = mean(tmp$RMSE),
    MAPE = mean(tmp$MAPE)
  )
}

results_xgb_step2 <- bind_rows(results_xgb_step2)

results_xgb_step2 <- results_xgb_step2[
  order(results_xgb_step2$RMSE),
]

results_xgb_step2

best_step2 <- results_xgb_step2[1, ]
best_step2

### 3. Observation

best_reg_alpha <- best_step2$reg_alpha
best_reg_lambda <- best_step2$reg_lambda
best_subsample <- best_step2$subsample
best_colsample_bytree <- best_step2$colsample_bytree

grid_step3 <- expand.grid(
  eta = c(0.01, 0.03, 0.05, 0.1),
  nrounds = c(100, 200, 300, 500)
)

results_xgb_step3 <- list()

for (g in 1:nrow(grid_step3)) {
  
  eta <- grid_step3$eta[g]
  nrounds <- grid_step3$nrounds[g]
  
  fold_results <- list()
  
  for (i in 1:n_folds) {
    
    fold <- cv_folds[[i]]
    
    train_fold <- power_ts[
      fold$train_start:fold$train_end
    ]
    
    valid_fold <- power_ts[
      fold$valid_start:fold$valid_end
    ]
    
    train_ml <- create_lag_data(
      train_fold,
      lags = 96
    )
    
    X_train <- as.matrix(
      train_ml[, -1]
    )
    
    y_train <- train_ml$y
    
    model <- xgboost(
      data = X_train,
      label = y_train,
      nrounds = nrounds,
      max_depth = best_max_depth,
      min_child_weight = best_min_child_weight,
      eta = eta,
      reg_alpha = best_reg_alpha,
      reg_lambda = best_reg_lambda,
      subsample = best_subsample,
      colsample_bytree = best_colsample_bytree,
      objective = "reg:squarederror",
      verbose = 0
    )
    
    history <- as.numeric(train_fold)
    predictions <- numeric(horizon)
    
    for (j in 1:horizon) {
      
      x_input <- tail(history, 96)
      x_input <- rev(x_input)
      x_input <- matrix(
        x_input,
        nrow = 1
      )
      
      pred <- predict(
        model,
        x_input
      )
      
      predictions[j] <- pred
      
      history <- c(
        history,
        pred
      )
    }
    
    fold_results[[i]] <- evaluate_forecast(
      valid_fold,
      predictions,
      "XGBoost",
      i
    )
  }
  
  tmp <- bind_rows(fold_results)
  
  results_xgb_step3[[g]] <- data.frame(
    eta = eta,
    nrounds = nrounds,
    MAE = mean(tmp$MAE),
    RMSE = mean(tmp$RMSE),
    MAPE = mean(tmp$MAPE)
  )
}

results_xgb_step3 <- bind_rows(results_xgb_step3)

results_xgb_step3 <- results_xgb_step3[
  order(results_xgb_step3$RMSE),
]

results_xgb_step3

best_xgb <- results_xgb_step3[1, ]
best_xgb

# NOTE
#' According this, the best model should have this params: max_depth = 7,min_child_weight = 5,reg_alpha = 0,reg_lambda = 1,subsample = 1,colsample_bytree = 1

best_xgb_model <- xgboost(
  data = as.matrix(create_lag_data(train, lags = 96)[, -1]),
  label = create_lag_data(train, lags = 96)$y,
  nrounds = 200,
  max_depth = 7,
  min_child_weight = 5,
  eta = 0.3,
  reg_alpha = 0,
  reg_lambda = 1,
  subsample = 1,
  colsample_bytree = 1,
  objective = "reg:squarederror",
  verbose = 0
)

train_ml <- create_lag_data(train, lags = 96)

pred_train_xgb <- predict(
  best_xgb_model,
  as.matrix(train_ml[, -1])
)

history <- as.numeric(train)
pred_val_xgb <- numeric(length(validation))

for (j in 1:length(validation)) {
  x_input <- tail(history, 96)
  x_input <- rev(x_input)
  x_input <- matrix(x_input, nrow = 1)
  
  pred <- predict(best_xgb_model, x_input)
  
  pred_val_xgb[j] <- pred
  history <- c(history, pred)
}

history <- c(
  as.numeric(train),
  as.numeric(validation)
)

pred_test_xgb <- numeric(length(test))

for (j in 1:length(test)) {
  x_input <- tail(history, 96)
  x_input <- rev(x_input)
  x_input <- matrix(x_input, nrow = 1)
  
  pred <- predict(best_xgb_model, x_input)
  
  pred_test_xgb[j] <- pred
  history <- c(history, pred)
}

result_train_xgb <- evaluate_forecast(
  actual = as.numeric(train)[97:length(train)],
  predicted = pred_train_xgb,
  model_name = "XGBoost",
  fold = "Train dataset"
)

result_val_xgb <- evaluate_forecast(
  actual = as.numeric(validation),
  predicted = pred_val_xgb,
  model_name = "XGBoost",
  fold = "Validation dataset"
)

result_test_xgb <- evaluate_forecast(
  actual = as.numeric(test),
  predicted = pred_test_xgb,
  model_name = "XGBoost",
  fold = "Test dataset"
)

results_xgb_global <- bind_rows(
  result_train_xgb,
  result_val_xgb,
  result_test_xgb
)

results_xgb_global

n_train <- length(train)
n_val <- length(validation)
n_test <- length(test)

df_train_xgb <- data.frame(
  Time = 97:n_train,
  Power = as.numeric(train)[97:n_train],
  Type = "Train"
)

df_pred_train_xgb <- data.frame(
  Time = 97:n_train,
  Power = as.numeric(pred_train_xgb),
  Type = "pred Train"
)

df_val_xgb <- data.frame(
  Time = (n_train + 1):(n_train + n_val),
  Power = as.numeric(validation),
  Type = "Validation"
)

df_pred_val_xgb <- data.frame(
  Time = (n_train + 1):(n_train + n_val),
  Power = as.numeric(pred_val_xgb),
  Type = "pred Validation"
)

df_test_xgb <- data.frame(
  Time = (n_train + n_val + 1):(n_train + n_val + n_test),
  Power = as.numeric(test),
  Type = "Test "
)

df_pred_test_xgb <- data.frame(
  Time = (n_train + n_val + 1):(n_train + n_val + n_test),
  Power = as.numeric(pred_test_xgb),
  Type = "pred Test"
)

df_plot_xgb <- rbind(
  df_train_xgb,
  df_pred_train_xgb,
  df_val_xgb,
  df_pred_val_xgb,
  df_test_xgb,
  df_pred_test_xgb
)

ggplot(
  df_plot_xgb,
  aes(x = Time, y = Power, color = Type)
) +
  geom_line(linewidth = 0.7) +
  labs(
    title = "Prediction XGBoost",
    subtitle = "Training set, validation set  and test set",
    x = "Observations",
    y = "Power (kW)",
    color = ""
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "bottom"
  )

#' NOTE XGBOOST
#' Il mets trop de temps
#' 
# ============================================================
# 2. RANDOM FOREST
# ============================================================


results_rf <- list()
results_rf_m <- list()

for (i in 1:n_folds) {
  
  fold <- cv_folds[[i]]
  
  train_fold <- df_elec_no_na[
    fold$train_start:fold$train_end,
    "Power_kW"
  ]
  
  valid_fold <- df_elec_no_na[
    fold$valid_start:fold$valid_end,
    "Power_kW"
  ]
  
  train_fold <- as.numeric(
    unlist(train_fold, use.names = FALSE)
  )
  
  valid_fold <- as.numeric(
    unlist(valid_fold, use.names = FALSE)
  )
  
  train_ml <- create_lag_data(
    train_fold,
    lags = 96
  )
  
  model <- randomForest(
    y ~ .,
    data = train_ml,
    ntree = 500,
    mtry = 10
  )
  
  history <- train_fold
  
  horizon_val <- length(valid_fold)
  predictions_val <- numeric(horizon_val)
  
  for (j in 1:horizon_val) {
    
    x_input <- data.frame(
      matrix(
        rev(tail(history, 96)),
        nrow = 1
      )
    )
    
    names(x_input) <- paste0(
      "lag_",
      1:96
    )
    
    pred <- predict(
      model,
      x_input
    )
    
    predictions_val[j] <- pred
    
    history <- c(
      history,
      pred
    )
  }
  
  results_rf[[i]] <- evaluate_forecast(
    valid_fold,
    predictions_val,
    "Random Forest",
    i
  )
}

results_rf_m <- bind_rows(results_rf)
results_rf_m


#'
#'


train_ml <- create_lag_data(
  train,
  lags = 96
)

X_train <- as.matrix(
  train_ml[, -1]
)

y_train <- train_ml$y

best_rf_model <- randomForest(
  x = X_train,
  y = y_train,
  ntree = 500,
  mtry = floor(sqrt(96)),
  nodesize = 5
)

pred_train_rf <- predict(
  best_rf_model,
  X_train
)

history <- as.numeric(train)

pred_val_rf <- numeric(length(validation))

for (j in 1:length(validation)) {
  
  x_input <- tail(history, 96)
  x_input <- rev(x_input)
  x_input <- matrix(x_input, nrow = 1)
  
  pred <- predict(
    best_rf_model,
    x_input
  )
  
  pred_val_rf[j] <- pred
  
  history <- c(
    history,
    pred
  )
}

history <- c(
  as.numeric(train),
  as.numeric(validation)
)

pred_test_rf <- numeric(length(test))

for (j in 1:length(test)) {
  
  x_input <- tail(history, 96)
  x_input <- rev(x_input)
  x_input <- matrix(x_input, nrow = 1)
  
  pred <- predict(
    best_rf_model,
    x_input
  )
  
  pred_test_rf[j] <- pred
  
  history <- c(
    history,
    pred
  )
}

result_train_rf <- evaluate_forecast(
  actual = as.numeric(train)[97:length(train)],
  predicted = pred_train_rf,
  model_name = "Random Forest",
  fold = "Train dataset"
)

result_val_rf <- evaluate_forecast(
  actual = as.numeric(validation),
  predicted = pred_val_rf,
  model_name = "Random Forest",
  fold = "Validation dataset"
)

result_test_rf <- evaluate_forecast(
  actual = as.numeric(test),
  predicted = pred_test_rf,
  model_name = "Random Forest",
  fold = "Test dataset"
)

results_rf_global <- bind_rows(
  result_train_rf,
  result_val_rf,
  result_test_rf
)

results_rf_global

n_train <- length(train)
n_val <- length(validation)
n_test <- length(test)

df_train_rf <- data.frame(
  Time = 97:n_train,
  Power = as.numeric(train)[97:n_train],
  Type = "Train"
)

df_pred_train_rf <- data.frame(
  Time = 97:n_train,
  Power = as.numeric(pred_train_rf),
  Type = "pred Train"
)

df_val_rf <- data.frame(
  Time = (n_train + 1):(n_train + n_val),
  Power = as.numeric(validation),
  Type = "Validation"
)

df_pred_val_rf <- data.frame(
  Time = (n_train + 1):(n_train + n_val),
  Power = as.numeric(pred_val_rf),
  Type = "pred Validation "
)

df_test_rf <- data.frame(
  Time = (n_train + n_val + 1):(n_train + n_val + n_test),
  Power = as.numeric(test),
  Type = "Test"
)

df_pred_test_rf <- data.frame(
  Time = (n_train + n_val + 1):(n_train + n_val + n_test),
  Power = as.numeric(pred_test_rf),
  Type = " pred Test "
)

df_plot_rf <- bind_rows(
  df_train_rf,
  df_pred_train_rf,
  df_val_rf,
  df_pred_val_rf,
  df_test_rf,
  df_pred_test_rf
)

ggplot(
  df_plot_rf,
  aes(x = Time, y = Power, color = Type)
) +
  geom_line(linewidth = 0.7) +
  labs(
    title = "Prédiction Random Forest multivariate",
    subtitle = "Training set , validation set and test test",
    x = "Observations",
    y = "Power (kW)",
    color = ""
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "bottom"
  )

#' ==================================================================
#' ------------- NNAR------------------------------------------------
#' =================================================================


# ============================================================
# 1. TRAIN / VALIDATION / TEST
# ============================================================

train_data <- as.numeric(train)
validation_data <- as.numeric(validation)
test_data <- as.numeric(test)


# ============================================================
# 2. ENTRAÎNEMENT NNAR SUR TRAIN
# ============================================================

train_ts <- ts(
  train_data,
  frequency = 96
)

model_nn <- nnetar(
  train_ts,
  p = 96,
  P = 1,
  size = 10,
  repeats = 10
)


# ============================================================
# 3. PRÉDICTION VALIDATION
# ============================================================

pred_validation <- forecast(
  model_nn,
  h = length(validation_data)
)$mean


# ============================================================
# 4. PERFORMANCE VALIDATION
# ============================================================

result_validation_nn <- evaluate_forecast(
  validation_data,
  pred_validation,
  "Neural Net",
  "Validation"
)

result_validation_nn


# ============================================================
# 5. TRAIN + VALIDATION
# ============================================================

train_valid_data <- c(
  train_data,
  validation_data
)

train_valid_ts <- ts(
  train_valid_data,
  frequency = 96
)


# ============================================================
# 6. RÉ-ENTRAÎNEMENT NNAR
# ============================================================

model_nn_final <- nnetar(
  train_valid_ts,
  p = 96,
  P = 1,
  size = 10,
  repeats = 10
)


# ============================================================
# 7. PRÉDICTION TEST
# ============================================================

pred_test <- forecast(
  model_nn_final,
  h = length(test_data)
)$mean


# ============================================================
# 8. PERFORMANCE TEST
# ============================================================

result_test_nn <- evaluate_forecast(
  test_data,
  pred_test,
  "Neural Net",
  "Test"
)

result_test_nn

# AutoSelect 
results_nn_auto <- list()
models_nn <- list()

for (i in 1:n_folds) {
  fold <- cv_folds[[i]]
  
  train_fold <- df_elec_no_na[fold$train_start:fold$train_end]
  valid_fold <- df_elec_no_na[fold$valid_start:fold$valid_end]
  
  model <- nnetar(
    train_fold,
    P = 1,
    repeats = 10
  )
  
  models_nn[[i]] <- model
  
  cat(
    "Fold", i,
    ": NNAR(",
    model$p, ",",
    model$size,
    ")[96]\n"
  )
  
  pred <- forecast(
    model,
    h = horizon
  )$mean
  
  results_nn_auto[[i]] <- evaluate_forecast(
    valid_fold,
    pred,
    "NNAR",
    i
  )
}

results_nn_auto <- bind_rows(results_nn_auto)
results_nn_auto

#' NOTE NNAR

#' Le NNAR manuel est nettement plus performant que la sélection automatique, avec des erreurs globalement plus faibles (MAPE de 2,05 % à 7,56 % contre 11,76 % à 17,76 % pour le NNAR automatique).
#' Le modèle manuel est donc retenu, car il offre une meilleure précision et une meilleure stabilité sur les cinq folds.

## PLOTTING

best_model_nn <- nnetar(
  train,
  p = 96,
  P = 1,
  size = 10,
  repeats = 10
)

pred_val_nn <- forecast(
  best_model_nn,
  h = length(validation)
)$mean

pred_test_nn <- forecast(
  best_model_nn,
  h = length(test)
)$mean

result_validation_nn <- evaluate_forecast(
  actual = as.numeric(validation),
  predicted = as.numeric(pred_val_nn),
  model_name = "NNAR(96,10)[96]",
  fold = "Validation dataset"
)

result_test_nn <- evaluate_forecast(
  actual = as.numeric(test),
  predicted = as.numeric(pred_test_nn),
  model_name = "NNAR(96,10)[96]",
  fold = "Test dataset"
)

result_validation_nn
result_test_nn

n_train <- length(train)
n_validation <- length(validation)
n_test <- length(test)

df_train_nn <- data.frame(
  Time = 1:n_train,
  Power = as.numeric(train),
  Type = "Train"
)

df_validation_nn <- data.frame(
  Time = (n_train + 1):(n_train + n_validation),
  Power = as.numeric(validation),
  Type = "Validation"
)

df_test_nn <- data.frame(
  Time = (n_train + n_validation + 1):
    (n_train + n_validation + n_test),
  Power = as.numeric(test),
  Type = "Test"
)

df_pred_val_nn <- data.frame(
  Time = (n_train + 1):(n_train + n_validation),
  Power = as.numeric(pred_val_nn),
  Type = "Validation prediction"
)

df_pred_test_nn <- data.frame(
  Time = (n_train + n_validation + 1):
    (n_train + n_validation + n_test),
  Power = as.numeric(pred_test_nn),
  Type = "Test prediction"
)

df_plot_nn <- rbind(
  df_train_nn,
  df_validation_nn,
  df_test_nn,
  df_pred_val_nn,
  df_pred_test_nn
)

ggplot(
  df_plot_nn,
  aes(x = Time, y = Power, color = Type)
) +
  geom_line(linewidth = 0.7) +
  labs(
    title = "Prédiction  NNAR(96,10)[96]",
    subtitle = "Train, validation and test",
    x = "Observations",
    y = "Power (kW)",
    color = ""
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "bottom"
  )

#SCORE 80% success
results_nn_global <- bind_rows(
  result_validation_nn,
  result_test_nn
)

results_nn_global


#'=========================================================================
#'------------------XGBOOST-----------------------------------------------
#'========================================================================
#'
#'

train_data <- as.numeric(train)
validation_data <- as.numeric(validation)
test_data <- as.numeric(test)


# ============================================================
# 2. FONCTION POUR CRÉER LES LAGS
# ============================================================

create_lag_data <- function(x, lags = c(1, 2, 4, 8, 96, 192, 288)) {
  
  data <- data.frame(
    Power = as.numeric(x)
  )
  
  for (lag in lags) {
    data[[paste0("lag_", lag)]] <- dplyr::lag(
      data$Power,
      lag
    )
  }
  
  data <- na.omit(data)
  
  return(data)
}


# ============================================================
# 3. CRÉATION DES DONNÉES TRAIN
# ============================================================

lags <- c(1, 2, 4, 8, 96, 192, 288)

train_lag <- create_lag_data(
  train_data,
  lags = lags
)

X_train <- train_lag[, -1]
y_train <- train_lag$Power


# ============================================================
# 4. CRÉATION DES DONNÉES VALIDATION
# ============================================================

# On utilise les dernières observations du TRAIN
# pour construire les lags de la validation.

history_valid <- tail(
  train_data,
  max(lags)
)

valid_with_history <- c(
  history_valid,
  validation_data
)

valid_lag <- create_lag_data(
  valid_with_history,
  lags = lags
)

# On conserve uniquement les lignes correspondant
# à la période de validation

valid_lag <- tail(
  valid_lag,
  length(validation_data)
)

X_valid <- valid_lag[, -1]
y_valid <- valid_lag$Power


# ============================================================
# 5. ENTRAÎNEMENT XGBOOST
# ============================================================

model_xgb <- xgboost(
  data = as.matrix(X_train),
  label = y_train,
  nrounds = 500,
  max_depth = 7,
  min_child_weight = 5,
  eta = 0.01,
  subsample = 1,
  colsample_bytree = 1,
  objective = "reg:squarederror",
  verbose = 0
)


# ============================================================
# 6. PRÉDICTION VALIDATION
# ============================================================

pred_validation <- predict(
  model_xgb,
  as.matrix(X_valid)
)


# ============================================================
# 7. PERFORMANCE VALIDATION
# ============================================================

result_validation_xgb <- evaluate_forecast(
  y_valid,
  pred_validation,
  "XGBoost",
  "Validation"
)

result_validation_xgb
# ============================================================
# 8. TRAIN + VALIDATION
# ============================================================

train_valid_data <- c(
  train_data,
  validation_data
)

train_valid_lag <- create_lag_data(
  train_valid_data,
  lags = lags
)

X_train_valid <- train_valid_lag[, -1]
y_train_valid <- train_valid_lag$Power


# ============================================================
# 9. MODÈLE FINAL XGBOOST
# ============================================================

model_xgb_final <- xgboost(
  data = as.matrix(X_train_valid),
  label = y_train_valid,
  nrounds = 500,
  max_depth = 7,
  min_child_weight = 5,
  eta = 0.01,
  subsample = 1,
  colsample_bytree = 1,
  objective = "reg:squarederror",
  verbose = 0
)


# ============================================================
# 10. PRÉPARATION DU TEST
# ============================================================

history_test <- tail(
  train_valid_data,
  max(lags)
)

test_with_history <- c(
  history_test,
  test_data
)

test_lag <- create_lag_data(
  test_with_history,
  lags = lags
)

test_lag <- tail(
  test_lag,
  length(test_data)
)

X_test <- test_lag[, -1]
y_test <- test_lag$Power


# ============================================================
# 11. PRÉDICTION TEST
# ============================================================

pred_test_xgb <- predict(
  model_xgb_final,
  as.matrix(X_test)
)


# ============================================================
# 12. PERFORMANCE FINALE TEST
# ============================================================

result_test_xgb <- evaluate_forecast(
  y_test,
  pred_test_xgb,
  "XGBoost",
  "Test"
)

result_test_xgb

#'============================================================================
#'----------------------------- Final selection ------------------------------
#'============================================================================









#'============================================================================
#'============================================================================
#'---------------------------NEW ALGO ----------------------------------------
#'
#'==========================================================================
#'------------------- WNN Algo Implements in R ----------------------------
#'==========================================================================

# Step 1 : C_i = [c_{i+1}, ..., c_{i+h}]
# h  futur values associated to i

C_i <- function(c, i, h) {
  c[(i + 1):(i + h)]
}

# Step 2 : CC_i = [c_{i-w+1}, ..., c_i]
#  w are the last known values, ending at i
CC_i <- function(c, i, w) {
  c[(i - w + 1):i]
}

# Step 3 : dist(i, j) = || CC_i - CC_j 
dist_C_iC_j <- function(CC_i, CC_j) {
  sqrt(sum((CC_i - CC_j)^2))
}

# Step 4 : NS = { q_1, ..., q_k }
neighbours <- function(c, i, w, h, k) {
  candidats <- w:(i - h)
  if (length(candidats) < 1) {
    stop("There is not enough historical data for the given w, h, and i.")
  }
  
  CC_i <- CC_i(c, i, w)
  distances <- sapply(candidats, function(j) dist_C_iC_j(CC_i, CC_i(c, j, w)))
  
  k_eff <- min(k, length(candidats))
  ord <- order(distances)[seq_len(k_eff)]
  
  list(
    q    = candidats[ord],
    dist = distances[ord]
  )
}

#Step 6 : alpha_j = 1 / (dist(CC_qj, CC_i))^2

alpha_weights <- function(distances) {
  distances[distances == 0] <- .Machine$double.eps
  1 / (distances^2)
}

#Step 5 : C_i = (1 / sum(alpha_j)) * sum(alpha_j * C_qj)
predict_WNN <- function(c, i, w, h, k) {
  ns    <- neighbours(c, i, w, h, k)
  alpha <- alpha_weights(ns$dist)
  
  C_qj <- t(sapply(ns$q, function(j) C_i(c, j, h)))
  Ci  <- colSums(alpha * C_qj) / sum(alpha)
  list(prediction = Ci, NS = ns$q, dist = ns$dist, alpha = alpha)
}


#'------------------- Custom for easy calling  ----------------------
#'----------------------------------------------------------------
WNN_predict_period <- function(history, horizon, w = 96, k = 5) {
  
  history <- as.numeric(history)
  horizon <- as.numeric(horizon)
  
  h <- length(horizon)
  data <- c(history, horizon)
  
  predictions <- numeric(h)
  
  start <- length(history) + 1
  
  for (t in seq_len(h)) {
    i <- start + t - 1
    
    result <- predict_WNN(
      c = data,
      i = i - 1,
      w = w,
      h = 1,
      k = k
    )
    
    predictions[t] <- result$prediction[1]
    data[i] <- predictions[t]
  }
  
  predictions
}

# ============================================================
# WNN - TRAIN / VALIDATION / TEST
# ============================================================

# ============================================================
# PARAMÈTRES
# ============================================================

w <- 96
k_max <- 50

train_data <- as.numeric(train)
validation_data <- as.numeric(validation)
test_data <- as.numeric(test)


# ============================================================
# RECHERCHE DE K
# ============================================================

results_k <- list()

previous_mae <- Inf
best_k <- 2
best_mae <- Inf

for (k in 2:k_max) {
  
  cat("\n==============================\n")
  cat("Test de k =", k, "\n")
  cat("==============================\n")
  
  
  # Prédiction
  result_wnn <- predict_WNN(
    c = train_data,
    i = length(train_data),
    w = w,
    h = length(validation_data),
    k = k
  )
  
  pred_validation_wnn <- result_wnn$prediction
  
  
  # Évaluation
  result <- evaluate_forecast(
    validation_data,
    pred_validation_wnn,
    "WNN",
    paste0("Validation k=", k)
  )
  
  
  # Stockage
  results_k[[k]] <- result
  
  current_mae <- result$MAE
  
  cat("MAE :", current_mae, "\n")
  
  
  # ==========================================================
  # ARRÊT SI k N'AMÉLIORE PAS k-1
  # ==========================================================
  
  # Mise à jour
  if (current_mae < best_mae) {
    best_mae <- current_mae
    best_k <- k
  }
  
  previous_mae <- current_mae
}


# ============================================================
# RÉSULTATS
# ============================================================

results_k <- bind_rows(results_k)

cat("\n====================================\n")
cat("MEILLEUR MODÈLE WNN\n")
cat("====================================\n")
cat("w =", w, "\n")
cat("k =", best_k, "\n")
cat("MAE =", best_mae, "\n")
cat("====================================\n")

print(results_k)

# ============================================================
# TEST FINAL DU WNN
# k sélectionné sur la validation
# ============================================================

w <- 96
k <- 12


# ============================================================
# 1. DONNÉES
# ============================================================

train_data <- as.numeric(train)
validation_data <- as.numeric(validation)
test_data <- as.numeric(test)


# ============================================================
# 2. TRAIN + VALIDATION
# ============================================================

train_valid_data <- c(
  train_data,
  validation_data
)


# ============================================================
# 3. PRÉDICTION SUR TEST
# ============================================================

result_wnn_test <- predict_WNN(
  c = train_valid_data,
  i = length(train_valid_data),
  w = w,
  h = length(test_data),
  k = k
)

pred_test_wnn <- result_wnn_test$prediction


# ============================================================
# 4. ÉVALUATION SUR TEST
# ============================================================

result_test_wnn <- evaluate_forecast(
  test_data,
  pred_test_wnn,
  "WNN",
  "Test"
)

print(result_test_wnn)

#'____________________________________________________________________________
#'___________________PLOT________________________________________________________
#
train_data <- as.numeric(train)
validation_data <- as.numeric(validation)
test_data <- as.numeric(test)

pred_validation_wnn <- as.numeric(pred_validation_wnn)
pred_test_wnn <- as.numeric(pred_test_wnn)


# ============================================================
# 2. INDEX TEMPOREL
# ============================================================

n_train <- length(train_data)
n_validation <- length(validation_data)
n_test <- length(test_data)

time_train <- 1:n_train

time_validation <- 
  (n_train + 1):(n_train + n_validation)

time_test <- 
  (n_train + n_validation + 1):
  (n_train + n_validation + n_test)


# ============================================================
# 3. DONNÉES RÉELLES
# ============================================================

df_real <- rbind(
  
  data.frame(
    Time = time_train,
    Power = train_data,
    Type = "Train"
  ),
  
  data.frame(
    Time = time_validation,
    Power = validation_data,
    Type = "Validation"
  ),
  
  data.frame(
    Time = time_test,
    Power = test_data,
    Type = "Test"
  )
)


# ============================================================
# 4. PRÉDICTIONS WNN
# ============================================================

df_predictions <- rbind(
  
  data.frame(
    Time = time_validation,
    Power = pred_validation_wnn,
    Type = "WNN Validation"
  ),
  
  data.frame(
    Time = time_test,
    Power = pred_test_wnn,
    Type = "WNN Test"
  )
)


# ============================================================
# 5. GRAPHIQUE
# ============================================================

ggplot() +
  
  # Données réelles
  geom_line(
    data = df_real,
    aes(
      x = Time,
      y = Power,
      color = Type
    ),
    linewidth = 0.8
  ) +
  
  # Prédictions WNN
  geom_line(
    data = df_predictions,
    aes(
      x = Time,
      y = Power,
      linetype = Type
    ),
    color = "red",
    linewidth = 1.2
  ) +
  
  # Séparation Train / Validation / Test
  geom_vline(
    xintercept = n_train,
    color = "orange",
    linetype = "dashed",
    linewidth = 0.8
  ) +
  
  geom_vline(
    xintercept = n_train + n_validation,
    color = "orange",
    linetype = "dashed",
    linewidth = 0.8
  ) +
  
  # Couleurs
  scale_color_manual(
    values = c(
      "Train" = "blue",
      "Validation" = "orange",
      "Test" = "green"
    )
  ) +
  
  # Types de lignes des prédictions
  scale_linetype_manual(
    values = c(
      "WNN Validation" = "dashed",
      "WNN Test" = "solid"
    )
  ) +
  
  labs(
    title = "WNN Forecasts Based on Train, Validation, and Test Data",
    subtitle = paste(
      "Window w =", w,
      "| Nb of neighbors k =", k
    ),
    x = "Times",
    y = "Power (kW)",
    color = "Real data",
    linetype = "Predictions WNN"
  ) +
  
  theme_minimal(base_size = 13) +
  
  theme(
    plot.title = element_text(
      face = "bold",
      size = 16
    ),
    plot.subtitle = element_text(size = 12),
    legend.position = "top",
    panel.grid.minor = element_blank()
  )



















#'=====================================================================================================
#'======================================================================================================
#'=======================================================================================================
#'

# ============================================================
# 1. DONNÉES DISPONIBLES
# ============================================================
all_data <- as.numeric(df_elec_no_na$Power_kW)

h <- 96

all_lag <- create_lag_data(
  all_data,
  lags = lags
)

X_all <- all_lag[, -1, drop = FALSE]
y_all <- as.numeric(all_lag$Power)

X_all <- matrix(
  as.numeric(unlist(X_all)),
  nrow = nrow(X_all),
  ncol = ncol(X_all)
)

model_xgb_final <- xgboost(
  data = X_all,
  label = y_all,
  nrounds = 500,
  max_depth = 7,
  min_child_weight = 5,
  eta = 0.01,
  subsample = 1,
  colsample_bytree = 1,
  objective = "reg:squarederror",
  verbose = 0
)

history <- tail(
  all_data,
  max(lags)
)

pred_96 <- numeric(h)

for (i in 1:h) {
  
  lag_data <- create_lag_data(
    history,
    lags = lags
  )
  
  X_next <- lag_data[
    nrow(lag_data),
    -1,
    drop = FALSE
  ]
  
  X_next <- matrix(
    as.numeric(unlist(X_next)),
    nrow = 1,
    ncol = ncol(X_all)
  )
  
  storage.mode(X_next) <- "double"
  
  pred_96[i] <- as.numeric(
    predict(
      model_xgb_final,
      newdata = X_next
    )
  )
  
  history <- c(
    history,
    pred_96[i]
  )
}

last_timestamp <- max(
  df_elec_no_na$Timestamp,
  na.rm = TRUE
)

future_timestamp <- seq(
  from = last_timestamp + 15 * 60,
  by = 15 * 60,
  length.out = h
)

forecast_96 <- data.frame(
  Timestamp = future_timestamp,
  Predicted_Power_kW = pred_96
)

forecast_96@Predicted_Power_kW
# Write in excell
install.packages("openxlsx")
library(openxlsx)

forecast_export <- data.frame(
  Predicted_Power_kW = forecast_96$Predicted_Power_kW
)

write.xlsx(
  forecast_export,
  file = "Peyanan_TRAORE.xlsx",
  rowNames = FALSE
)


#'======================== PLOT =====================
#'
df_historique <- data.frame(
  Timestamp = df_elec_no_na$Timestamp,
  Power_kW = df_elec_no_na$Power_kW,
  Type = "Données réelles"
)

df_prediction <- data.frame(
  Timestamp = forecast_96$Timestamp,
  Power_kW = forecast_96$Predicted_Power_kW,
  Type = "Prévision XGBoost"
)

df_graph <- rbind(
  df_historique,
  df_prediction
)

ggplot(df_graph, aes(x = Timestamp, y = Power_kW, color = Type)) +
  geom_line(linewidth = 0.8) +
  labs(
    title = "Consommation électrique : données réelles et prévisions",
    x = "Date",
    y = "Power (kW)",
    color = ""
  ) +
  scale_color_manual(
    values = c(
      "Données réelles" = "blue",
      "Prévision XGBoost" = "red"
    )
  ) +
  theme_minimal() +
  theme(
    legend.position = "top"
  )