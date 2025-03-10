// Stan model for pupil-baseline estimation
//
functions{
    // asymmetric laplace function with the mu, sigma, tau parametrization
    real my_skew_double_exponential_lpdf(real y, real mu, real sigma, real tau) {
      return log(tau) + log1m(tau)
        - log(sigma)
        - 2 * ((y < mu) ? (1 - tau) * (mu - y) : tau * (y - mu)) / sigma;
    }
    
    // zero-centered asymmetric laplace function with the mu, lambda, kappa parametrization
    real skew_double_exponential2_lpdf(real y, real lam, real kappa) {
      return log(lam) - log(kappa+1/kappa)
        + ((y<0) ? (lam/kappa) : (-lam*kappa))*(y);
    }
}
data{ 
  int<lower=1> n; // number of timepoints in the signal
  vector[n] sy;   // the pupil signal
    
  int<lower=1> ncol; // number of basis functions (columns in B)
  matrix[n,ncol] B;  // spline basis functions
    
  int<lower=1> npeaks;          // number of lower peaks in the signal
  array[npeaks] int<lower=1> peakix;  // index of the lower peaks in sy
  vector<lower=0>[npeaks] lam_prominences; // lambda-converted prominence values
    
  real<lower=0> lam_sig;    // lambda for the signal where there is no peak
  real<lower=0,upper=1> pa; // proportion of allowed distribution below 0
}

transformed data{
    vector[n] lam;       // lambda at each timepoint
    real<lower=0> kappa; // calculated kappa from pa
    kappa=sqrt(pa)/sqrt(1-pa);
    
    lam=rep_vector(lam_sig, n); 
    for(i in 1:npeaks){
        lam[peakix[i]]=lam_prominences[i];
    }
}
parameters {
    vector[ncol] coef; // coefficients for the basis-functions
}

transformed parameters{
    
}

model {
    {
    vector[n] d;
        
    coef ~ normal(0,5);
    d=sy-(B*coef); // center at estimated baseline
    for( i in 1:n ){
        d[i] ~ skew_double_exponential2(lam[i], kappa);
    }
    }
}