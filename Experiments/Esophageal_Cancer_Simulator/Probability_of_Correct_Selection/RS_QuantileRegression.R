library(tictoc)
library(cqrReg)
library(future.apply)

m=320
le=1/m
ue=1-le
interval=1/m
quantilepoints=seq(le,ue,interval)
ntau=length(quantilepoints)
#setwd("D:/QRGMM/Esophageal_Cancer_Simulator/Probability_of_Correct_Selection")

# Create output folders
dir.create("./data/RS/GMMcoeff1", recursive = TRUE, showWarnings = FALSE)
dir.create("./data/RS/GMMcoeff2", recursive = TRUE, showWarnings = FALSE)

run=3

# Set parallel workers
workers <- max(1, parallel::detectCores() %/% 2)
future::plan(future::multisession, workers = workers)

# ============================
#   ECtraindata1 -> GMMcoeff1
# ============================
for (runi in 1:run) {

  path <- paste("./data/RS/ECtraindata1/ECtraindata1_", runi, sep = "")
  path <- paste(path, ".csv", sep = "")
  GMMtraindata <- read.csv(file = path, sep = ",", header = FALSE)
  GMMtraindata <- data.matrix(GMMtraindata)

  nc <- ncol(GMMtraindata)
  x <- GMMtraindata[, 2:(nc - 1)]
  y <- GMMtraindata[, nc]
  y <- matrix(y, length(y), 1)
  p <- ncol(x)

  coeff_list <- future_lapply(1:ntau, function(i) {
    b <- QR.admm(x, y, quantilepoints[i])
    tempb <- matrix(0, 1, p + 1)
    tempb[2:(p + 1)] <- b[[1]]
    tempb[1] <- b[[2]]
    tempb
  })

  coeff <- do.call(rbind, coeff_list)

  path <- paste("./data/RS/GMMcoeff1/GMMcoeff1_", runi, sep = "")
  path <- paste(path, ".csv", sep = "")
  write.csv(coeff, path)
}

# ============================
#   ECtraindata2 -> GMMcoeff2
# ============================
for (runi in 1:run) {

  path <- paste("./data/RS/ECtraindata2/ECtraindata2_", runi, sep = "")
  path <- paste(path, ".csv", sep = "")
  GMMtraindata <- read.csv(file = path, sep = ",", header = FALSE)
  GMMtraindata <- data.matrix(GMMtraindata)

  nc <- ncol(GMMtraindata)
  x <- GMMtraindata[, 2:(nc - 1)]
  y <- GMMtraindata[, nc]
  y <- matrix(y, length(y), 1)
  p <- ncol(x)

  coeff_list <- future_lapply(1:ntau, function(i) {
    b <- QR.admm(x, y, quantilepoints[i])
    tempb <- matrix(0, 1, p + 1)
    tempb[2:(p + 1)] <- b[[1]]
    tempb[1] <- b[[2]]
    tempb
  })

  coeff <- do.call(rbind, coeff_list)

  path <- paste("./data/RS/GMMcoeff2/GMMcoeff2_", runi, sep = "")
  path <- paste(path, ".csv", sep = "")
  write.csv(coeff, path)
}

future::plan(future::sequential)
