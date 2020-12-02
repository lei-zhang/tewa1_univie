data {
  int<lower=1> nTrials;
  int<lower=1, upper=2> choice[nTrials];
  int<lower=-1, upper=1> reward[nTrials];
}

transformed data{
  vector[2] initV;
  initV = rep_vector(0, 2);
}

parameters {
  real<lower=0, upper=1> alpha;
  real<lower=0, upper=10> tau;
}

model {
  vector[2] V;
  vector[2] p;
  real pe;
  
  V = initV;
  
  for (t in 1:nTrials) {
    p = softmax(V);
    choice[t] ~ categorical(p);
    pe = reward[t] - V[choice[t]];
    V[choice[t]] = V[choice[t]] + alpha * pe; 
  }
}

