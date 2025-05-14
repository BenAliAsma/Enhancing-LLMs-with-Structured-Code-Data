# Server-side request forgery
Directly incorporating user input into an HTTP request without validating the input can facilitate server-side request forgery (SSRF) attacks. In these attacks, the request may be changed, directed at a different server, or via a different protocol. This can allow the attacker to obtain sensitive information or perform actions with escalated privilege.


## Recommendation
To guard against SSRF attacks you should avoid putting user-provided input directly into a request URL. Instead, maintain a list of authorized URLs on the server; then choose from that list based on the input provided. Alternatively, ensure requests constructed from user input are limited to a particular host or more restrictive URL prefix.


## Example
The following example shows an HTTP request parameter being used directly to form a new request without validating the input, which facilitates SSRF attacks. It also shows how to remedy the problem by validating the user input against a known fixed string.


```ruby
require "excon"
require "json"

class PostsController < ActionController::Base
  def create
    user = params[:user_id]

    # BAD - user can control the entire URL of the request
    users_service_domain = params[:users_service_domain]
    response = Excon.post("#{users_service_domain}/logins", body: {user_id: user}).body
    token = JSON.parse(response)["token"]

    # GOOD - path is validated against a known fixed string
    path = if params[:users_service_path] == "v1/users"
             "v1/users"
           else 
             "v2/users"
           end
    response = Excon.post("users-service/#{path}", body: {user_id: user}).body
    token = JSON.parse(response)["token"]

    @post = Post.create(params[:post].merge(user_token: token))
    render @post
  end
end

```

## References
* [OWASP SSRF](https://owasp.org/www-community/attacks/Server_Side_Request_Forgery)
* Common Weakness Enumeration: [CWE-918](https://cwe.mitre.org/data/definitions/918.html).
