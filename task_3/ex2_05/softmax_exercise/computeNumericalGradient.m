function numgrad = computeNumericalGradient(J,theta)
% numgrad = computerNumerical Gradient(J, theta)
% theta: a vector of paramters
% J: a function tha outputs a real-number.Calling y = J(theta) will return
% the function value at theta.

% Initialize numgrad with zeros
numgrad = zeros(size(theta));

%% ----------YOUR CODE HERE-----------------------------------
% Instructions:
% Implement numerical gradient checking, and return the result in numgrad.
% (See Section 2.3 of the lecture notes.)
% You shoul write code so that numgrad(i) is (the numerical approximation
% to) the pratical dervative of J with respect to the i-th input argument,
% evaluated at theta.
% I.e.,numgrad(i) should be the (approximately) the partial derivative of J
% with respect to theta(i).
%
% Hint: You will probably want to compute the elements of numgrad one at a
% time

EPSILON = 10^-4;
for i=1:size(theta)
    theta_temp1=theta;
    theta_temp2=theta;
    theta_temp1(i)=theta_temp1(i)+EPSILON;
    theta_temp2(i)=theta_temp2(i)+EPSILON;
    numgrad(i)=(J(theta_temp1)-J(theta_temp2))./(exp(1)*EPSILON);
end

%% ------------------------------------------------------------------
end

