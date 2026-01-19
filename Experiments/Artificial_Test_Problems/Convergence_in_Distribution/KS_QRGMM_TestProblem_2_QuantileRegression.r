library(cqrReg)
library(future.apply)

run <- 100
dir.create("./data_convergence/QRGMMcoeff", recursive = TRUE)

workers <- max(1, parallel::detectCores() %/% 2)
future::plan(future::multisession, workers = workers)

for (i in 1:15) {
  m <- (i + 1) * 10
  n <- m^2

  le <- 1 / m
  ue <- 1 - le
  interval <- 1 / m
  quantilepoints <- seq(le, ue, interval)
  ntau <- length(quantilepoints)

  for (runi in 1:run) {

    path <- paste0("./data_convergence/traindata/traindata_", i, "_", runi, ".csv")
    GMMtraindata <- read.csv(file = path, sep = ",", header = FALSE)
    GMMtraindata <- data.matrix(GMMtraindata)

    nc <- ncol(GMMtraindata)
    x <- GMMtraindata[, 2:(nc - 1)]
    y <- GMMtraindata[, nc]
    y <- matrix(y, length(y), 1)

    p <- ncol(x)

    coeff_list <- future_lapply(seq_len(ntau), function(j) {
      b <- QR.admm(x, y, quantilepoints[j])

      row <- numeric(p + 2)
      row[1] <- b[[2]]              # intercept
      row[2:(p + 1)] <- b[[1]]      # slopes
      row[p + 2] <- quantilepoints[j]
      row
    })

    coeff <- do.call(rbind, coeff_list)

    out_path <- paste0("./data_convergence/QRGMMcoeff/QRGMMcoeff_", i, "_", runi, ".csv")
    write.csv(coeff, out_path, row.names = FALSE)
  }
}

future::plan(future::sequential)
