function [X_hat, X, Y, A] = problem1(n,m,k,max_iters,s,lambda)
    % Initialize the X that we will recreate...
    X = zeros(n,1);
    for i = 1:k
        index = randi(n);
        X(index) = 1;
    end
   
    % generating the matrix A s.t. A~N(0, 1/m)
    A = sqrt((1/m)).*randn(m,n);
    
    % setting up the compressed vector Y
    Y = A*X;
    
    % initialize the new vector X_hat randomly
    X_hat = rand(n,1);
    
    % preform gradient decsent %according 
    % to the partial derivative w.r.t X
    for i = 1:max_iters
        % partial derivative
        dX = sign(X_hat) + (((lambda * -1 * A).')*sign(Y - A*X_hat));
        % update step
        X_hat = X_hat - (s * dX);
    end
end