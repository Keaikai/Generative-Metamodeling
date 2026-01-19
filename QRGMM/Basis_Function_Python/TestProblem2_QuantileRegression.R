library(tictoc)
library(cqrReg)
library(future.apply)

m <- 300
le <- 1 / m
ue <- 1 - le
interval <- 1 / m
quantilepoints <- seq(le, ue, interval)
ntau <- length(quantilepoints)

dir.create("./data/QRGMMcoeff", recursive = TRUE)
#setwd("D:/QRGMM/Artificial_Test_Problems/Test_Problem_2_Performance")

set.seed(2024)
run <- 100

workers <- max(1, parallel::detectCores() %/% 2)
future::plan(future::multisession, workers = workers)

for (runi in 1:run) {

  path <- paste0("./data/traindata/traindata_", runi, ".csv")
  traindata <- read.csv(file = path, sep = ",", header = FALSE)
  traindata <- data.matrix(traindata)

  nc <- ncol(traindata)
  x_train <- traindata[, 2:(nc - 1)]
  y_train <- traindata[, nc]
  y_train <- matrix(y_train, length(y_train), 1)

  p <- ncol(x_train)

  coeff_list <- future_lapply(seq_len(ntau), function(i) {
    b <- QR.admm(x_train, y_train, quantilepoints[i])
    row <- numeric(p + 2)
    row[1] <- b[[2]]                 # intercept
    row[2:(p + 1)] <- b[[1]]         # slopes
    row[p + 2] <- quantilepoints[i]  # tau
    row
  })

  coeff <- do.call(rbind, coeff_list)

  out_path <- paste0("./data/QRGMMcoeff/QRGMMcoeff_", runi, ".csv")
  write.csv(coeff, out_path, row.names = FALSE)
}

future::plan(future::sequential)