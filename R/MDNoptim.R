
# weight dimensions and total number of parameters
MDN.npar <- function(X,hidden_neurons,components,static=NULL){

  ##################################
  # Dimension of weight matrices
  dim.xh <- c(ncol(as.matrix(X))+1,hidden_neurons)

  ##################################
  # Check for static parameters
  st <- c("alpha","mu","sigma") %in% static

  dim.ha <- c(ifelse(st[1],1,hidden_neurons+1),components)
  dim.hm <- c(ifelse(st[2],1,hidden_neurons+1),components)
  dim.hs <- c(ifelse(st[3],1,hidden_neurons+1),components)

  dim <- list(xh=dim.xh,
              ha=dim.ha,hm=dim.hm,hs=dim.hs,
              total=prod(dim.xh)+prod(dim.ha)+prod(dim.hm)+prod(dim.hs),
              n=nrow(as.matrix(X)))

  return(dim)
}

# Distribute weights from vector to matrices
MDN.vectomat <- function(w,dim){

  if(length(w)==1) w <- rep(w,dim$total)

  tmp <- cumsum(c(0,prod(dim$xh),prod(dim$ha),prod(dim$hm),prod(dim$hs)))+1

  ##################################
  ## weight matrices

  # input -> hidden
  W.xh <- matrix(w[tmp[1]:(tmp[2]-1)],nrow=dim$xh[1],ncol=dim$xh[2])
  # hidden -> mixture weight
  W.ha <- matrix(w[tmp[2]:(tmp[3]-1)],nrow=dim$ha[1],ncol=dim$ha[2])
  # hidden -> mean
  W.hm <- matrix(w[tmp[3]:(tmp[4]-1)],nrow=dim$hm[1],ncol=dim$hm[2])
  # hidden -> sd
  W.hs <- matrix(w[tmp[4]:(tmp[5]-1)],nrow=dim$hs[1],ncol=dim$hs[2])

  mats <- list(xh=W.xh,
               ha=W.ha,
               hm=W.hm,
               hs=W.hs)

  return(mats)
}



############################################
######## Functions for BFGS


MDN <- function(w,X,y,hidden_neurons,components,static=NULL){

  y <- as.numeric(y)

  ##################################
  # Check for static parameters
  st <- c("alpha","mu","sigma") %in% static

  ##################################
  ## weight matrices

  dim <- MDN.npar(X,hidden_neurons,components,static)
  W <- MDN.vectomat(w,dim)

  ##################################
  ## forward propagation

  # hidden
  Z.h <- cbind(1,X)%*%W$xh
  O <- tanh(Z.h)
  # mixture weights
  Z.a <- (if(st[1]) rep(1,dim$n) else cbind(1,O))%*%W$ha
  A <- softmax(Z.a)
  # mean
  Z.m <- (if(st[2]) rep(1,dim$n) else cbind(1,O))%*%W$hm
  M <- identity(Z.m)
  # standard deviation
  Z.s <- (if(st[3]) rep(1,dim$n) else cbind(1,O))%*%W$hs
  S <- exp(Z.s)

  ##################################
  ## loss function

  # negative log-likelihood
  nlL <- (-sum(log((dnorm(matrix(rep(y,ncol(A)),ncol=ncol(A)),M,S)*A)%*%rep(1,ncol(A)))))

  ##################################

  return(nlL)
}


grMDN <- function(w,X,y,hidden_neurons,components,static=NULL){

  y <- as.numeric(y)

  ##################################
  # Check for static parameters
  st <- c("alpha","mu","sigma") %in% static

  ##################################
  ## weight matrices

  dim <- MDN.npar(X,hidden_neurons,components,static)
  W <- MDN.vectomat(w,dim)

  ##################################
  ## forward propagation

  # hidden
  Z.h <- cbind(1,X)%*%W$xh
  O <- tanh(Z.h)
  # mixture weights
  Z.a <- (if(st[1]) rep(1,dim$n) else cbind(1,O))%*%W$ha
  A <- softmax(Z.a)
  # mean
  Z.m <- (if(st[2]) rep(1,dim$n) else cbind(1,O))%*%W$hm
  M <- identity(Z.m)
  # standard deviation
  Z.s <- (if(st[3]) rep(1,dim$n) else cbind(1,O))%*%W$hs
  S <- exp(Z.s)

  ##################################
  ## gradients

  n <- length(y)

  # posterior probs
  P <- softmax(log(dnorm(matrix(rep(y,ncol(A)),ncol=ncol(A)),M,S)*A))

  # mixture weight -> hidden
  dA <- A-P
  dW.ha <- (t(if(st[1]) rep(1,dim$n) else cbind(1,O))%*%dA)/n
  # mean -> hidden
  dM <- P*((M-y)/S^2)
  dW.hm <- (t(if(st[2]) rep(1,dim$n) else cbind(1,O))%*%dM)/n
  # sd -> hidden
  dS <- -P*((M-y)^2/S^3-1/S) #  dS <- -P*((M-y)^2/S^3-1/S)
  dW.hs <- (t(if(st[3]) rep(1,dim$n) else cbind(1,O))%*%dS)/n

  # hidden -> input
  D <- cbind(if(st[1]) NULL else dA,
             if(st[2]) NULL else dM,
             if(st[3]) NULL else dS)
  DW <- cbind(if(st[1]) NULL else W$ha,
              if(st[2]) NULL else W$hm,
              if(st[3]) NULL else W$hs)

  dY <- D%*%t(DW[-1,]) # remove remaining intercepts
  dO <- dY*tanh.prime(Z.h)
  dW.xh <- (t(cbind(1,X))%*%dO)/n

  ##################################
  # reconvert to 1d-vector
  grads <- c(as.numeric(dW.xh),
             as.numeric(dW.ha),
             as.numeric(dW.hm),
             as.numeric(dW.hs))

  return(grads)
}


############################################
######## Wrapper for model fit


MDN.fit <- function(X,y,hidden_neurons,components,static=NULL,w=NULL,method="BFGS"){

  dim <- MDN.npar(X,hidden_neurons=hidden_neurons,components=components,static=static)

  if(is.null(w)) w <- rnorm(dim$total,0,.2)

  mdnfit <- optim(par=w,MDN,grMDN,y=y,X=X,
                  hidden_neurons=hidden_neurons,components=components,static=static,
                  method = method,control=list(maxit=1000))

  mdnfit$str <- list(hidden_neurons=hidden_neurons,components=components,static=static)

  mdnfit$W <- MDN.vectomat(mdnfit$par,dim)
  mdnfit$par <- NULL

  return(mdnfit)
}

MDN.predict <- function(mdnfit,X,parameter=NULL,component=NULL){

  ##################################
  # Check for static parameters

  st <-  c("alpha","mu","sigma") %in% mdnfit$str$static

  ##################################
  ## forward propagation

  X <- as.matrix(X)

  # hidden
  Z.h <- cbind(1,X)%*%mdnfit$W$xh
  O <- tanh(Z.h)
  # mixture weights
  Z.a <- (if(st[1]) rep(1,nrow(X)) else cbind(1,O))%*%mdnfit$W$ha
  A <- softmax(Z.a)
  # mean
  Z.m <- (if(st[2]) rep(1,nrow(X)) else cbind(1,O))%*%mdnfit$W$hm
  M <- identity(Z.m)
  # standard deviation
  Z.s <- (if(st[3]) rep(1,nrow(X)) else cbind(1,O))%*%mdnfit$W$hs
  S <- exp(Z.s)

  ##################################

  if(is.null(component)) component <- 1:mdnfit$str$components

  if(is.null(parameter)){
    return(cbind(A[,component],M[,component],S[,component]))
  }

  if(parameter=="alpha"){
    return(A[,component])
  }
  if(parameter=="mu"){
    return(M[,component])
  }
  if(parameter=="sigma"){
    return(S[,component])
  }

}
