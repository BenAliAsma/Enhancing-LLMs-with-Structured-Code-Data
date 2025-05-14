# Manually checking http verb instead of using built in rails routes and protections
Manually checking the HTTP request verb inside of a controller method can lead to CSRF bypass if GET or HEAD requests are handled improperly.


## Recommendation
It is better to use different controller methods for each resource/http verb combination and configure the Rails routes in your application to call them accordingly.


## References
* See https://guides.rubyonrails.org/routing.html for more information.
