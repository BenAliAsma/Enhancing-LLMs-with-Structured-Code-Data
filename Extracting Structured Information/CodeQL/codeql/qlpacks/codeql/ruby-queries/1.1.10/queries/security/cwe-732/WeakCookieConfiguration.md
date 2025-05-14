# Weak cookie configuration
Cookies can be used for security measures, such as authenticating a user based on cookies sent with a request. Misconfiguration of cookie settings in a web application can expose users to attacks that compromise these security measures.


## Recommendation
Modern web frameworks typically have good default configuration for cookie settings. If an application overrides these settings, then take care to ensure that these changes are necessary and that they don't weaken the cookie configuration.


## Example
In the first example, the value of `config.action_dispatch.cookies_same_site_protection` is set to `:none`. This has the effect of setting the default `SameSite` attribute sent by the server when setting a cookie to `None` rather than the default of `Lax`. This may make the application more vulnerable to cross-site request forgery attacks.

In the second example, this option is instead set to `:strict`. This is a stronger restriction than the default of `:lax`, and doesn't compromise on cookie security.


```ruby
module App
  class Application < Rails::Application
    # Sets default `Set-Cookie` `SameSite` attribute to `None`
    config.action_dispatch.cookies_same_site_protection = :none

    # Sets default `Set-Cookie` `SameSite` attribute to `Strict`
    config.action_dispatch.cookies_same_site_protection = :strict
  end
end

```

## References
* OWASP: [SameSite](https://owasp.org/www-community/SameSite).
* Rails: [Configuring Action Dispatch](https://guides.rubyonrails.org/configuring.html#configuring-action-dispatch).
* Common Weakness Enumeration: [CWE-732](https://cwe.mitre.org/data/definitions/732.html).
* Common Weakness Enumeration: [CWE-1275](https://cwe.mitre.org/data/definitions/1275.html).
