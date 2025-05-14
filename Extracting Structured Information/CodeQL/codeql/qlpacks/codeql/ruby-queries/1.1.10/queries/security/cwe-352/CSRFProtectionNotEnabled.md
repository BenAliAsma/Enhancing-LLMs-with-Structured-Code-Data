# CSRF protection not enabled
Cross-site request forgery (CSRF) is a type of vulnerability in which an attacker is able to force a user to carry out an action that the user did not intend.

The attacker tricks an authenticated user into submitting a request to the web application. Typically this request will result in a state change on the server, such as changing the user's password. The request can be initiated when the user visits a site controlled by the attacker. If the web application relies only on cookies for authentication, or on other credentials that are automatically included in the request, then this request will appear as legitimate to the server.

A common countermeasure for CSRF is to generate a unique token to be included in the HTML sent from the server to a user. This token can be used as a hidden field to be sent back with requests to the server, where the server can then check that the token is valid and associated with the relevant user session.


## Recommendation
In the Rails web framework, CSRF protection is enabled by the adding a call to the `protect_from_forgery` method inside an `ActionController` class. Typically this is done in the `ApplicationController` class, or an equivalent class from which other controller classes are subclassed. The default behaviour of this method is to null the session when an invalid CSRF token is provided. This may not be sufficient to avoid a CSRF vulnerability - for example if parts of the session are memoized. Calling `protect_from_forgery with: :exception` can help to avoid this by raising an exception on an invalid CSRF token instead.


## Example
The following example shows a case where CSRF protection is enabled with a secure request handling strategy of `:exception`.


```ruby
class ApplicationController < ActionController::Base
  protect_from_forgery with: :exception
end
  
```

## References
* Wikipedia: [Cross-site request forgery](https://en.wikipedia.org/wiki/Cross-site_request_forgery)
* OWASP: [Cross-site request forgery](https://owasp.org/www-community/attacks/csrf)
* Securing Rails Applications: [Cross-Site Request Forgery (CSRF)](https://guides.rubyonrails.org/security.html#cross-site-request-forgery-csrf)
* Veracode: [When Rails' protect_from_forgery Fails](https://www.veracode.com/blog/managing-appsec/when-rails-protectfromforgery-fails).
* Common Weakness Enumeration: [CWE-352](https://cwe.mitre.org/data/definitions/352.html).
