# Weak or direct parameter references are used
Directly checking request parameters without following a strong params pattern can lead to unintentional avenues for injection attacks.


## Recommendation
Instead of manually checking parameters from the \`param\` object, it is recommended that you follow the strong parameters pattern established in Rails: https://api.rubyonrails.org/classes/ActionController/StrongParameters.html

In the strong parameters pattern, you are able to specify required and allowed parameters for each action called by your controller methods. This acts as an additional layer of data validation before being passed along to other areas of your application, such as the model.


## References
