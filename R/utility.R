
# used for backprop
tanh.prime <- function(x){
  4/(exp(-x) + exp(x))^2
}

# link function mixture weights
softmax <- function(x){
  apply(x,2,function(z) exp(z)/rowSums(exp(x)))
}
