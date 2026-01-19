# install.packages("qrnn")

library(qrnn)
library(future.apply)


le=0.01
ue=0.99
interval=0.01
quantilepoints=seq(le,ue,interval)
ntau=length(quantilepoints)

#setwd("D:/QRGMM/Bank_Simulator")

# -------------------------
# Parallel plan (adjust workers as you like)
# -------------------------
workers <- max(1, parallel::detectCores() %/% 2)
future::plan(future::multisession, workers = workers)

set.seed(2024)


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

predicting <- function(testx, fitmodel){
  geny<-matrix(0,nrow(testx),1)
  dtestx=ncol(testx)
  ntestx=nrow(testx)
  for(i in 1:ntestx){
    u = runif(1)
    low_ind<-u%/%interval
    testx1<-matrix(testx[i,],1,dtestx)
    if(low_ind<1) {
      geny[i]<-qrnn.predict(testx1, fitmodel[[1]])
    } else if(low_ind==ntau){
      geny[i]<-qrnn.predict(testx1, fitmodel[[ntau]])
    }else{
      up_ind<-low_ind+1
      weight<-(u%%interval)/interval
      low_geny<-qrnn.predict(testx1, fitmodel[[low_ind]])
      up_geny<-qrnn.predict(testx1, fitmodel[[up_ind]])
      geny[i] <- low_geny+weight*(up_geny-low_geny)
    }
  }
  return(geny)
}


p=10


run=100

for(runi in 1:run){
  set.seed(runi)
  
  path1 <- paste("./data/traindata/train_data_rep",runi,sep="")
  path1 <- paste(path1,".csv",sep="")
  traindata <- read.csv(path1)
  traindata=data.matrix(traindata)
  dtraindata=ncol(traindata)
  
  path2 <- paste("./data/testdata/test_data_rep",runi,sep="")
  path2 <- paste(path2,".csv",sep="")  
  testdata <- read.csv(path2)
  testdata=data.matrix(testdata)
  ntestdata=nrow(testdata)
  
  fitmodel <- vector("list", length = dtraindata - p)
  

  for (d in (p+1):dtraindata){
    x <- traindata[,1:d-1]
    x<-as.matrix(x)
    y <- traindata[,d]
    y<-as.matrix(y)
    fitmodel[[d-p]]<- fitting(x, y, quantilepoints)
  }

  gendata_qrnn=matrix(0,ntestdata,dtraindata)
  gendata_qrnn[,1:p]=testdata[,1:p]

  for (d in (p+1):dtraindata){
    testx <- gendata_qrnn[,1:d-1]
    testx<-as.matrix(testx)
    gendata_qrnn[,d]<-predicting(testx,fitmodel[[d-p]])

  }

  path3 <- paste("./data/testdata_QRNNGMM/testdata_QRNNGMM_rep",runi,sep="")
  path3 <- paste(path3,".csv",sep="")
  write.csv(gendata_qrnn,path3)
  
}


future::plan(future::sequential)
