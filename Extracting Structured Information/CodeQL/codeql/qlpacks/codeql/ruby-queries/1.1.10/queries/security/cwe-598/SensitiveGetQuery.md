# Sensitive data read from GET request
Sensitive information such as passwords should not be transmitted within the query string of the requested URL. Sensitive information within URLs may be logged in various locations, including the user's browser, the web server, and any proxy servers between the two endpoints. URLs may also be displayed on-screen, bookmarked or emailed around by users. They may be disclosed to third parties via the Referer header when any off-site links are followed. Placing sensitive information into the URL therefore increases the risk that it will be captured by an attacker.


## Recommendation
Use HTTP POST to send sensitive information as part of the request body; for example, as form data.


## Example
The following example shows two route handlers that both receive a username and a password. The first receives this sensitive information from the query parameters of a GET request, which is transmitted in the URL. The second receives this sensitive information from the request body of a POST request.


```ruby
Rails.application.routes.draw do
  get "users/login", to: "#login_get" # BAD: sensitive data transmitted through query parameters
  post "users/login", to: "users#login_post" # GOOD: sensitive data transmitted in the request body
end

```

```ruby
class UsersController < ActionController::Base
  def login_get
    password = params[:password]
    authenticate_user(params[:username], password)
  end

  def login_post
    password = params[:password]
    authenticate_user(params[:username], password)
  end

  private
  def authenticate_user(username, password)
    # ... authenticate the user here
  end
end

```

## References
* CWE: [CWE-598: Use of GET Request Method with Sensitive Query Strings](https://cwe.mitre.org/data/definitions/598.html)
* PortSwigger (Burp): [Password Submitted using GET Method](https://portswigger.net/kb/issues/00400300_password-submitted-using-get-method)
* OWASP: [Information Exposure through Query Strings in URL](https://owasp.org/www-community/vulnerabilities/Information_exposure_through_query_strings_in_url)
* Common Weakness Enumeration: [CWE-598](https://cwe.mitre.org/data/definitions/598.html).
