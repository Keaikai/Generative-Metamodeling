# install.packages("qrnn")

library(tictoc)
library(qrnn)
library(future.apply)


le=0.01
ue=0.99
interval=0.01
quantilepoints=seq(le,ue,interval)
ntau=length(quantilepoints)

#setwd("D:/QRGMM/Bank_Simulator")

set.seed(2024)

# -------------------------
# Parallel plan (adjust workers as you like)
# -------------------------
workers <- max(1, parallel::detectCores() %/% 2)
future::plan(future::multisession, workers = workers)

fitting <- function(x, y, tau){
  fitmodel <- future_lapply(seq_along(tau), function(i) {
    qrnn.fit(
      x = x, y = y,
      n.hidden = 5,
      tau = tau[i],
      iter.max = 100,
      n.trials = 1
    )
  })
  return(fitmodel)
}


predicting_x0 <- function(testx, fitmodel){
  ntestx <- nrow(testx)
  u <- runif(ntestx)
  low_ind <- pmax(1, u%/%interval)
  up_ind <- pmin(ntau, u%/%interval + 1)
  weight <- (u %% interval) / interval
  low_geny <- matrix(0, ntestx, 1)
  up_geny <- matrix(0, ntestx, 1)
  testx0 <- matrix(testx[1, ], 1, ncol(testx))
  quantile_curve=matrix(0,ntau,1)
  for(i in 1:ntau){
    quantile_curve[i]=qrnn.predict(testx0,fitmodel[[i]])
  }
  low_geny <- quantile_curve[low_ind]
  up_geny <- quantile_curve[up_ind]
  geny <- low_geny + weight * (up_geny - low_geny)
  return(geny)
}

predicting <- function(testx, fitmodel){
  ntestx <- nrow(testx)
  u <- runif(ntestx)
  low_ind <- pmax(1, u%/%interval)
  up_ind <- pmin(ntau, u%/%interval + 1)
  weight <- (u %% interval) / interval
  low_geny <- matrix(0, ntestx, 1)
  up_geny <- matrix(0, ntestx, 1)
  for(i in 1:ntau){
    low_geny[which(low_ind == i)] <- qrnn.predict(testx[which(low_ind == i),],fitmodel[[i]])
    up_geny[which(up_ind == i)] <- qrnn.predict(testx[which(up_ind == i),],fitmodel[[i]])
  }
  geny <- low_geny + weight * (up_geny - low_geny)
  return(geny)
}


p=10

path1 <- paste("./data/traindata/train_data_rep1.csv")
traindata <- read.csv(path1)
traindata=data.matrix(traindata)
dtraindata=ncol(traindata)

path2 <- paste("./data/onlinetest/test_data_x0.csv")
testdata <- read.csv(path2)
testdata=data.matrix(testdata)
ntestdata=nrow(testdata)
set.seed(999)

fitmodel <- vector("list", length = dtraindata - p)

for (d in (p+1):dtraindata){
  x <- traindata[,1:(d-1)]
  x<-as.matrix(x)
  y <- traindata[,d]
  y<-as.matrix(y)
  fitmodel[[d-p]]<- fitting(x, y, quantilepoints)
}

run=100
onlinetime_qrnn=matrix(0,run,dtraindata-p+1)
tic.clearlog()
for(runi in 1:run){
  set.seed(runi)
  gendata_qrnn=matrix(0,ntestdata,dtraindata)
  gendata_qrnn[,1:p]=testdata[,1:p]

  start_time <- Sys.time() 
  testx<-gendata_qrnn[,1:p]
  testx<-as.matrix(testx)
  gendata_qrnn[,p+1]<-predicting_x0(testx,fitmodel[[1]])
  end_time <- Sys.time()  
  onlinetime_qrnn[runi,1] <- end_time - start_time 

  for (d in (p+2):dtraindata){
    start_time <- Sys.time()
    testx<-gendata_qrnn[,1:(d-1)]
    testx<-as.matrix(testx)
    gendata_qrnn[,d]<-predicting(testx,fitmodel[[d-p]])
    end_time <- Sys.time()
    onlinetime_qrnn[runi,d-p] <- end_time - start_time 
  }
  
  onlinetime_qrnn[runi,dtraindata-p+1]<-sum(onlinetime_qrnn[runi,1:(dtraindata-p)])
  path3 <- paste("./data/onlinetest/testdata_QRNNGMM_online/testdata_QRNNGMM_online_rep",runi,sep="")
  path3 <- paste(path3,".csv",sep="")
  write.csv(gendata_qrnn,path3)
  
}


colMeans(onlinetime_qrnn)

write.csv(onlinetime_qrnn,"./data/onlinetest/onlinetime_QRGMM.csv")

future::plan(future::sequential)