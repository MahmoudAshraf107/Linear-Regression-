%% Adjustment the Work Environment
clc
clear 
close all
%% load data and set the parameters
load data1.txt            % load the dataset
alpha=0.01;               % Learning rate coefficient 
iterations=1000;          % Number of training iterations
%% Main code
[m,n]=size(data1);                                    % get the size of the dataset
NewData=[ones(m,1)  data1];                           % add new column to the matrix (X0)
x=NewData(:,1:n);                                     % Export the input matrix form the dataset
Population=NewData(:,2);                              % Export the feature matrix form the dataset
Profit=NewData(:,3);                                  % Export the label matrix form the dataset
y=NewData(:,n+1);                                     % Export the output matrix form the dataset
theta=zeros(1,n);                                     % Assume Theta vector with zeros  
cost0=J(x,y,theta);                                   % Compute the cost with the assuemed theta
[finaltheta,cost]=GD(x,y,theta,alpha,iterations,n);   % Call the Gradient Descent function 
finalcost=cost(1,iterations);                         % Call the cost function 

%% find the best fit line 
X=linspace (min(Population),max (Population),100);    % take 100 values with equal intervals from the input 
Y=finaltheta(1,1)+(finaltheta(1,2)*X);                % find the expected vaules of the output at X
 
%% Plotting the results 
subplot(1,2,1)                                 %create subplot 1
hold on
grid on                                        % Create grid in the figure 
scatter(Population,Profit,'filled');           % Scatter plot between input in x-axis and output in y-axis 
ylabel({'Profit'});                            % x-axis label
xlabel({'Population'});                        % y-axis label
plot(X,Y,'r','linewidth',2);                   % Plot the best fit line with redline and width 2 on the same figure 
hold off
subplot(1,2,2)                                 %create subplot 2
plot(1:iterations,cost,'r','linewidth',2);     %Plot the rerlation between number of iterations and cost 
grid on                                        % Create grid in the figure 
axis([-50 1050 4 7])                           %Change the axes scale 
ylabel({'Cost'});                              % y-axis label
xlabel({'Iterations'});                        % x-axis label


%% Ending the Editor
fprintf(' \t\t\t\t\t\t\t\t\t\t\t Machine Learning \n')
fprintf(' \t\t\t\t\t\t\t\t\t\t\t Linear Regression \n')
fprintf(' \t\t\t\t\t\t\t\t\t\t  Mahmoud Ashraf Mahmoud \n')
%% Functions
function J=J(input,output,theta)                                 % create function with output J called J and three inputs 
         J=sum(((input*theta')-output).^2)/(2*length(input));    % Function equation (cost equation) 
          return                                              
end                                                                


function [theta,cost]=GD(input,output,theta,alpha,iterations,n)            % Gradient Descent function with outputs (theta, cost) called GD with six inputs 
         temp=zeros(size(theta));                                          % create zeros matrix with thenat size 
         cost=zeros(iterations);                                           % create cost as zeros matrix with iterations size

         for i=1:1:iterations                                              % for loop from i=1 to interations 
             error=(input*theta')-output;                                  % compute the error 

             for j=1:n                                                     % for loop fron j=1 to n 
                 term=error.*input(:,j);                                   % Compute the term of h(x)-
                 temp(1,j)=theta(1,j)-((alpha/length(input))*sum(term));   % compute the new theta (Gardient Desent)
             end

         theta=temp;                                                       % get the new vales of theta 
         cost(1,i)=sum(((input*theta')-output).^2)/(2*length(input));      %compute the cost according to the new values of theta
         cost=cost(1,:);                                                   %store the values of cost in matrix
         end 
         return
end
          