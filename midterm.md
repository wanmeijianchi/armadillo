# Presentation for midterm code 
## Strategy for accelerating running speed of code

- Accelerating by formulars
- Accelerating by avoiding repetitive computation
- Accelerating by changing orders of matrix computation
### Accelerating by formulars
$$ (A+UCV)^{-1}=A^{-1}-A^{-1}U(C^{-1}+VA^{-1}U)^{-1}VA^{-1}$$
$$ det(A+UWV^T) = det(W^{-1}+V^TA^{-1}U)det(W)det(A)$$


### Accelerating by avoiding repetitive computation
```C++
void fastCompute(mat U,mat W){
      mat AInverse,B_U,WInverse;
      vector<mat> InverseList_(N);
      vector<double>DetList_(N);
      double det_w = det(W);
      WInverse = inv(W);
      for(int i=0;i<N;i++){
        AInverse = eye(BList.at(i).n_rows,BList.at(i).n_rows)/sigmaSq;
        B_U = BList.at(i)*U;
        InverseList_.at(i) = AInverse -B_U*solve((WInverse+B_U.t()*B_U/sigmaSq),eye(size(W)))*B_U.t()/(sigmaSq*sigmaSq);
        DetList_.at(i) = pow(sigmaSq,BList.at(i).n_rows)*det(WInverse+B_U.t()*B_U/sigmaSq)*det_w;
        //DetList_.at(i) = det(B_U*W*B_U.t()+sigmaSq*eye(BList.at(i).n_rows,BList.at(i).n_rows));
      }
      InverseList = InverseList_;
      DetList = DetList_;
    }
```
$B_nU$ and $\Sigma^{-1}$ is used several times in the code.It is good strategy to compute $B_nU$ and $\Sigma^{-1}$  first.  
A function is built for computing  $\Sigma^{-1}$ and determinants.     $\Sigma^{-1}$  and determinants are stored in vectors as private numbers.  
Besides,the diagonal matrixs are computed as scalar manually.
### Accelerating by changing orders of matrix computation
```C++
B_U = BList.at(i)*U;
graSigma = InverseList.at(i) - (InverseList.at(i)*yList.at(i))*(yList.at(i).t()*InverseList.at(i));
result.at(0) += 2*BList.at(i).t()*graSigma*B_U*W;
result.at(1) += (B_U.t()*graSigma*B_U);
```
$$S_n=y_ny_n^T$$
$$\frac {\partial L_n(\Sigma_n)} {\partial \Sigma_n}=\Sigma_n^{-1}S_n\Sigma_n^{-1}
=\Sigma_n^{-1}y_ny_n^T\Sigma_n^{-1}
=(\Sigma_n^{-1}y_n)(y_n^T\Sigma_n^{-1})$$
Accelerating by changing orders of matrix computation is also a great strategy to reducing compute time. This strategy makes compute time reduce from 14s to 10.5s in my computer.

## Source code
```C++
#include <RcppArmadillo.h>
#include <vector>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
using namespace std;


class MLEMethod{
public:
    // Initialize
    MLEMethod(vector<vec> yList_, vector<mat> BList_): yList(yList_), BList(BList_), sigmaSq(0.5) {}
    
    // Take in a pair of U and W,
    // and output the value of the objective function L. Eqn~(1)
    //公式加速，不重复计算加速
    void fastCompute(mat U,mat W){
      mat AInverse,B_U,WInverse;
      vector<mat> InverseList_(N);
      vector<double>DetList_(N);
      double det_w = det(W);
      WInverse = inv(W);
      for(int i=0;i<N;i++){
        AInverse = eye(BList.at(i).n_rows,BList.at(i).n_rows)/sigmaSq;
        B_U = BList.at(i)*U;
        InverseList_.at(i) = AInverse - B_U*solve((WInverse+B_U.t()*B_U/sigmaSq),eye(size(W)))*B_U.t()/(sigmaSq*sigmaSq);
        DetList_.at(i) = pow(sigmaSq,BList.at(i).n_rows)*det(WInverse+B_U.t()*B_U/sigmaSq)*det_w;
        //DetList_.at(i) = det(B_U*W*B_U.t()+sigmaSq*eye(BList.at(i).n_rows,BList.at(i).n_rows));
      }
      InverseList = InverseList_;
      DetList = DetList_;
    }
    double objectiveL(mat U, mat W){
        double result = 0;
        double logDetVal;
        for(int i=0;i<N;i++)
        {

          logDetVal = log(DetList.at(i));
          result += (logDetVal+arma::as_scalar(yList.at(i).t()*InverseList.at(i)*yList.at(i)));
        }
        
        result = result/N;
        
        return result;
    }
    
    // Takes in a pair of U and W,
    // and output the gradient function with respect to U and W in Eqn~(2) and (3).
        vector<mat> gradientL(mat U, mat W){
        fastCompute(U,W);
        vector<mat> result(2);
        result.at(0) = mat(15,3,fill::zeros);
        result.at(1) = mat(3,3,fill::zeros);
        mat graSigma,B_U;
        for(int i=0;i<N;i++)
        {
          B_U = BList.at(i)*U;
          graSigma = InverseList.at(i) - (InverseList.at(i)*yList.at(i))*(yList.at(i).t()*InverseList.at(i));
          result.at(0) += 2*BList.at(i).t()*graSigma*B_U*W;
          //改变计算顺序加速
          result.at(1) += (B_U.t()*graSigma*B_U);
        }
        for(int i=0;i<2;i++)
          result.at(i) = result.at(i)/N;
        return result;
    }
    
private:
    vector<vec> yList;
    vector<mat> BList;
    double sigmaSq;
    const int N =2000;
    vector<mat> InverseList;
    vector<double> DetList;

};



// Do not modify the function below!
// [[Rcpp::export]]
Rcpp::List evaluate(vector<vec> yList_, vector<mat> BList_, mat U_, mat W_, int K, int R){
  double initObj;
  vector<mat> initGrad;
  MLEMethod testFun(yList_, BList_);
  
  // classical gradient algorithms evaluate the objective function and the gradient function repeated. 
  // In each round of iteration, the gradient function is evaluated once,
  //      and the the objective function is evaluated multiple times.
  // One target of your code is to improve the running speed of the code below.
  mat U1, W1;
  int i,j;
  for(i=0; i<200;i++){
    U1 = mat(K,R,fill::randn);
    W1 = mat(R,R,fill::randn);
    W1 = W1.t()*W1;
    testFun.gradientL(U_, W_);
    testFun.objectiveL(U_, W_);
    
    for(j=0; j<10; j++){
      U1 = mat(K,R,fill::randn);
      W1 = mat(R,R,fill::randn);
      W1 = W1.t()*W1;
      testFun.objectiveL(U_, W_);
    }
  }
  
  
  // The objective function and gradient function will be evaluated at the input values (U_, W_).
  // The values below will be returned to test the accuracy of your code.
  initObj = testFun.objectiveL(U_, W_); // Eqn (1)
  initGrad = testFun.gradientL(U_, W_); // Eqn (2)
  
  // return the values
  return Rcpp::List::create(Rcpp::Named("obj") = initObj, // the value of the objective function. 
                            Rcpp::Named("gradU") = initGrad.at(0),// the partial derivative with respect to U.
                            Rcpp::Named("gradU") = initGrad.at(1)); // the partial derivative with respect to W.
}
```
