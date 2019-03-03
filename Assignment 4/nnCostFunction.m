function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%adding bias node to a1
a1 = [ones(m,1) X];
z2 = a1*Theta1';

a2 = sigmoid(z2);

%adding bias node to a2
a2=[ones(size(z2,1),1) a2];

z3=a2*Theta2';
a3=sigmoid(z3);

%Hypothesis h(x)
h=a3;

%Re shape of the y matrix so that i have a dimensional
Mat_y = repmat(1:num_labels, m, 1) == repmat(y, 1, num_labels); 



%Cost Function
J = (1/m)* sum(sum(-Mat_y.*log(h)-(1-Mat_y).*log(1-h)));

%Regularization Term
Regularization = (lambda/(2*m))*( (sum(sum(Theta1(:,2:end).^2))) + (sum(sum(Theta2(:,2:end).^2))) );
J = J + Regularization;


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.

%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

%Back Propagation 
D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));

for t = 1:m 
   act1 = a1(t,:)';
   act2 = a2(t,:)';
   act3 = a3(t,:)';
   Mat_Y= Mat_y(t,:)';
   
   %Error happens at the last layer (L) which is Layer 3
   d3t = act3 - Mat_Y;
    
   z1new = [1; Theta1 * act1];
   
   %Error happens at the layer (L-1) which is Layer 2
   d2t = Theta2' * d3t .* sigmoidGradient(z1new);
   
   %Error isn't computed @Layer 1
   
   %NormLIZED the Error (i remove the bias node as there no weight as an
   %input for it)
   D1 = D1 + d2t(2:end).*act1';
   D2 = D2 + d3t .* act2';
    
end


%Obtaining the unregularized gradient for the neural network cost function
Theta1_grad = (1/m) * D1;
Theta2_grad = (1/m) * D2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%REGULARIZATION OF THE GRADIENT
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*(Theta1(:,2:end));
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*(Theta2(:,2:end));

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
