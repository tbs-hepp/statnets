
# log-logistic function embedded
rho_ab <- function(z, rob=rob){
  exp(rob+z)/(exp(rob+z)+1)}
rho <- function(z, rob=rob){
  log((1+exp(z+rob))/(1+exp(rob)))}


# weight dimensions and total number of parameters
DNN.npar <- function(X,hidden_neurons,static=NULL){

  ##################################
  # Dimension of weight matrices
  dim.xh <- c(ncol(as.matrix(X))+1,hidden_neurons)

  ##################################
  # Check for static parameters
  st <- c("mu","sigma") %in% static

  dim.hm <- c(ifelse(st[1],1,hidden_neurons+1),1)
  dim.hs <- c(ifelse(st[2],1,hidden_neurons+1),1)

  dim <- list(xh=dim.xh,
              hm=dim.hm,hs=dim.hs,
              total=prod(dim.xh)+prod(dim.hm)+prod(dim.hs),
              n=nrow(as.matrix(X)))

  return(dim)
}

DNN.npar(rnorm(100),5)

# Distribute weights from vector to matrices
DNN.vectomat <- function(w,dim){

  if(length(w)==1) w <- rep(w,dim$total)

  tmp <- cumsum(c(0,prod(dim$xh),prod(dim$hm),prod(dim$hs)))+1

  ##################################
  ## weight matrices

  # input -> hidden
  W.xh <- matrix(w[tmp[1]:(tmp[2]-1)],nrow=dim$xh[1],ncol=dim$xh[2])
  # hidden -> mean
  W.hm <- matrix(w[tmp[2]:(tmp[3]-1)],nrow=dim$hm[1],ncol=dim$hm[2])
  # hidden -> sd
  W.hs <- matrix(w[tmp[3]:(tmp[4]-1)],nrow=dim$hs[1],ncol=dim$hs[2])

  mats <- list(xh=W.xh,
               hm=W.hm,
               hs=W.hs)

  return(mats)
}



############################################
######## Functions for BFGS


DNN <- function(w,X,y,hidden_neurons,rob=0,static=NULL){

  ##################################
  # Check for static parameters
  st <- c("mu","sigma") %in% static

  ##################################
  ## weight matrices

  dim <- DNN.npar(X,hidden_neurons,static)
  W <- DNN.vectomat(w,dim)

  ##################################
  ## forward propagation

  # hidden
  Z.h <- cbind(1,X)%*%W$xh
  O <- tanh(Z.h)
  # mean
  Z.m <- (if(st[1]) rep(1,dim$n) else cbind(1,O))%*%W$hm
  M <- identity(Z.m)
  # standard deviation
  Z.s <- (if(st[2]) rep(1,dim$n) else cbind(1,O))%*%W$hs
  S <- exp(Z.s)

  ##################################
  ## loss function

  # negative log-likelihood
  # nlL <- (-sum(log((dnorm(y,M,S)))))
  nlL <- sum(-rho(z=dnorm(y, M, S, log = TRUE),rob=rob))
  ##################################

  return(nlL)
}


grDNN <- function(w,X,y,hidden_neurons,rob=0,static=NULL){

  ##################################
  # Check for static parameters
  st <- c("mu","sigma") %in% static

  ##################################
  ## weight matrices

  dim <- DNN.npar(X,hidden_neurons,static)
  W <- DNN.vectomat(w,dim)

  ##################################
  ## forward propagation

  # hidden
  Z.h <- cbind(1,X)%*%W$xh
  O <- tanh(Z.h)
  # mean
  Z.m <- (if(st[1]) rep(1,dim$n) else cbind(1,O))%*%W$hm
  M <- identity(Z.m)
  # standard deviation
  Z.s <- (if(st[2]) rep(1,dim$n) else cbind(1,O))%*%W$hs
  S <- exp(Z.s)

  ##################################
  ## gradients

  n <- length(y)

  # mean -> hidden
  # dM <- ((M-y)/S^2)
  dM <- -rho_ab(z=-0.5*log(2*pi)-log(S)-0.5*(y-M)^2/S^2 ,rob=rob)*(1/S^2) * (y - M)
  dW.hm <- (t(if(st[1]) rep(1,dim$n) else cbind(1,O))%*%dM)/n
  # sd -> hidden
  # dS <- -((M-y)^2/S^3-1/S) #  dS <- -P*((M-y)^2/S^3-1/S)
  dS <-  -rho_ab(z=-0.5*log(2*pi)-log(S)-0.5*(y-M)^2/exp(2*log(S)) ,rob=rob)*(-1 + exp(-2 * log(S)) * ((y - M)^2) )
  dW.hs <- (t(if(st[2]) rep(1,dim$n) else cbind(1,O))%*%dS)/n

  # hidden -> input
  D <- cbind(if(st[1]) NULL else dM,
             if(st[2]) NULL else dS)
  DW <- cbind(if(st[1]) NULL else W$hm,
              if(st[2]) NULL else W$hs)

  dY <- D%*%t(DW[-1,]) # remove remaining intercepts
  dO <- dY*tanh.prime(Z.h)
  dW.xh <- (t(cbind(1,X))%*%dO)/n

  ##################################
  # reconvert to 1d-vector
  grads <- c(as.numeric(dW.xh),
             as.numeric(dW.hm),
             as.numeric(dW.hs))

  return(grads)
}


############################################
######## Wrapper for model fit


DNN.fit <- function(X,y,hidden_neurons,rob=0,static=NULL,w=NULL,method="BFGS"){

  dim <- DNN.npar(X,hidden_neurons=hidden_neurons,static=static)

  if(is.null(w)) w <- rnorm(dim$total,0,.2)

  dnnfit <- optim(par=w,DNN,grDNN,y=y,X=X,
                  hidden_neurons=hidden_neurons,
                  rob=rob,static=static,
                  method = method,control=list(maxit=1000))

  dnnfit$str <- list(hidden_neurons=hidden_neurons,static=static)

  dnnfit$W <- DNN.vectomat(dnnfit$par,dim)
  dnnfit$par <- NULL

  return(dnnfit)
}

DNN.predict <- function(dnnfit,X,parameter=NULL){

  ##################################
  # Check for static parameters

  st <-  c("mu","sigma") %in% dnnfit$str$static

  ##################################
  ## forward propagation

  X <- as.matrix(X)

  # hidden
  Z.h <- cbind(1,X)%*%dnnfit$W$xh
  O <- tanh(Z.h)
  # mean
  Z.m <- (if(st[1]) rep(1,nrow(X)) else cbind(1,O))%*%dnnfit$W$hm
  M <- identity(Z.m)
  # standard deviation
  Z.s <- (if(st[2]) rep(1,nrow(X)) else cbind(1,O))%*%dnnfit$W$hs
  S <- exp(Z.s)

  ##################################

  if(is.null(parameter)){
    return(cbind(M,S))
  }

  if(parameter=="mu"){
    return(M)
  }
  if(parameter=="sigma"){
    return(S)
  }

}



#################

x <- seq(0,1,l=5000)

matplot(x,cbind(qnorm(0.025,2*x+sin(x*2.5*pi),.5+2*x-1.5*x^2),
                qnorm(0.975,2*x+sin(x*2.5*pi),.5+2*x-1.5*x^2)),t="l",lty=1,col=1)


y <- rnorm(5000,2*x+sin(x*2.5*pi),.5+2*x-1.5*x^2)


plot(x,y)

xout <- sort(runif(1500))
yout <- rnorm(1500,2+2*xout+sin(xout*2.5*pi),2)


plot(scale(c(x,xout)),c(y,yout),col=rep(c(1,2),times=c(5000,1500)))

dnnfit <- DNN.fit(scale(c(x,xout)),c(y,yout),5,rob=0)

# plot(scale(c(x,xout)),c(y,yout),col=rep(c(1,2),times=c(1000,500)))

xpred <- (seq(0,1,l=50)-mean(c(x,xout)))/sd(c(x,xout))

lines(xpred,
      qnorm(0.025,DNN.predict(dnnfit,X=xpred,"mu"),(DNN.predict(dnnfit,X=xpred,"sigma"))),
      col="orange",lwd=2)
lines(xpred,
      qnorm(0.975,DNN.predict(dnnfit,X=xpred,"mu"),(DNN.predict(dnnfit,X=xpred,"sigma"))),
      col="orange",lwd=2)

matlines(xpred,cbind(qnorm(0.025,2*seq(0,1,l=50)+sin(seq(0,1,l=50)*2.5*pi),.5+2*seq(0,1,l=50)-1.5*seq(0,1,l=50)^2),
                qnorm(0.975,2*seq(0,1,l=50)+sin(seq(0,1,l=50)*2.5*pi),.5+2*seq(0,1,l=50)-1.5*seq(0,1,l=50)^2)),
         t="l",col=3,lty=2,lwd=2)














#####################################
#######################################
robust_GaussianMu <- function (mu = NULL, sigma = NULL, stabilization,rob=rob)
{
  loss <- function(sigma, y, f) -rho(z=dnorm(x = y, mean = f, sd = sigma, log = TRUE),rob=rob)  # changes within the loss function
  risk <- function(y, f, w = 1) {
    sum(w * loss(y = y, f = f, sigma = sigma))
  }
  ngradient <- function(y, f, w = 1) {
    ngr <-     rho_ab(z=-0.5*log(2*pi)-log(sigma)-0.5*(y-f)^2/sigma^2 ,rob=rob)*(1/sigma^2) * (y - f)
    ngr <- stabilize_ngradient(ngr, w = w, stabilization)
    return(ngr)
  }
  offset <- function(y, w) {
    if (!is.null(mu)) {
      RET <- mu
    }
    else {
      RET <- weighted.mean(y, w = w, na.rm = TRUE)
    }
    return(RET)
  }
  rho_ab <- function(z, rob=rob){          # log-logistic function embedded
    exp(rob+z)/(exp(rob+z)+1)}             # log-logistic function embedded
  rho <- function(z, rob=rob){             # log-logistic function embedded
    log((1+exp(z+rob))/(1+exp(rob)))}      # log-logistic function embedded

  mboost::Family(ngradient = ngradient, risk = risk, loss = loss, response = function(f) f,
                 offset = offset, name = "robust Normal distribution: mu(id link)") # "robust"
}

############################################################################################################

robust_GaussianSigma <- function (mu = NULL, sigma = NULL, stabilization,rob=rob)
{
  loss <- function(y, f, mu) -rho(z=dnorm(x = y, mean = mu, sd = exp(f), log = TRUE),rob=rob)   # changes within the loss function
  risk <- function(y, f, w = 1) {
    sum(w * loss(y = y, f = f, mu = mu))
  }
  ngradient <- function(y, f, w = 1) {
    ngr <-  rho_ab(z=-0.5*log(2*pi)-f-0.5*(y-mu)^2/exp(2*f) ,rob=rob)*(-1 + exp(-2 * f) * ((y - mu)^2) )
    ngr <- stabilize_ngradient(ngr, w = w, stabilization)
    return(ngr)
  }
  offset <- function(y, w) {
    if (!is.null(sigma)) {
      RET <- log(sigma)
    }
    else {
      RET <- log(weighted.sd(y,w=w, na.rm = TRUE)) #log(sd(y, na.rm = TRUE)) # eigentlich:
    }
    return(RET)
  }
  rho_ab <- function(z, rob=rob){          # log-logistic function embedded
    exp(rob+z)/(exp(rob+z)+1)}             # log-logistic function embedded
  rho <- function(z, rob=rob){             # log-logistic function embedded
    log((1+exp(z+rob))/(1+exp(rob)))}      # log-logistic function embedded

  mboost::Family(ngradient = ngradient, risk = risk, loss = loss, response = function(f) exp(f),
                 offset = offset, name = "robust Normal distribution: sigma (log link)") ## "robust"
}

