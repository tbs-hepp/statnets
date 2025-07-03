# Examples

## Mixture Density Network

set.seed(12345)

x <- runif(5000)

alpha <- exp(.75+.75*sin(1.5+x*pi*1.5))/(1+exp(.75+.75*sin(1.5+x*pi*1.5)))
ALP <- cbind(alpha,1-alpha)

MU <- cbind(x+10*sin((x-.5)*sqrt(12)*pi/2),20+(20+1)*x+10*sin((x-.5)*sqrt(12)*pi/2))
SIG <- cbind(8+5*x,11+9*x)

z <- sapply(1:length(x),function(i){
  sample(1:2,1,prob = ALP[i,])
})

y <- sapply(1:length(x),function(i){
  rnorm(1,MU[i,z[i]],SIG[i,z[i]])
})

mdnfit <- MDN.fit(scale(x),scale(y),5,2)

xpred <- seq(min(scale(x)),max(scale(x)),l=50)

par(mfrow=c(2,2))

plot(scale(x),scale(y),col=z,pch=20)
lines(xpred,qnorm(.025,MDN.predict(mdnfit,xpred,"mu",1),(MDN.predict(mdnfit,xpred,"sigma",1))),lwd=2)
lines(xpred,qnorm(.975,MDN.predict(mdnfit,xpred,"mu",1),(MDN.predict(mdnfit,xpred,"sigma",1))),lwd=2)
lines(xpred,qnorm(.025,MDN.predict(mdnfit,xpred,"mu",2),(MDN.predict(mdnfit,xpred,"sigma",2))),lwd=2,col=2)
lines(xpred,qnorm(.975,MDN.predict(mdnfit,xpred,"mu",2),(MDN.predict(mdnfit,xpred,"sigma",2))),lwd=2,col=2)

matplot(x,ALP,ylim=c(0,1))
matlines(xpred*sd(x)+mean(x),MDN.predict(mdnfit,xpred,"alpha"),lwd=2)

matplot(x,MU)
matlines(xpred*sd(x)+mean(x),MDN.predict(mdnfit,xpred,"mu")*sd(y)+mean(y),lwd=2)

matplot(x,SIG)
matlines(xpred*sd(x)+mean(x),MDN.predict(mdnfit,xpred,"sigma")*sd(y),lwd=2)


