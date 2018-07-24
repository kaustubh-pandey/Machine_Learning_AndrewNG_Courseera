function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%display(size(X));
h=sigmoid((X*theta)');
h=h';

h1=log(h);
h2=log(1-h);
h1=y.*h1;
h2=(1-y).*h2;
J=sum(-h1-h2)/m;
s=sum(theta.^2)-(theta(1)^2);
s=s*lambda/(2*m);
J=J+s;
l=h-y;
%for i=2:28
%grad(i)= (sum(l.*X(:,i))/m+(lambda/m*theta(i)));
%end;
grad=((h-y)'*X)'/m+(lambda/m)*theta;
grad(1)=sum(l)/m;

% =============================================================

end
