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