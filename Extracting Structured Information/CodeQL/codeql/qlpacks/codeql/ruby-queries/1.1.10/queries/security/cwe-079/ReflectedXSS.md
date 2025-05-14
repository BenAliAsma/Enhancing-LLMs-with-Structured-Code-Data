# Reflected server-side cross-site scripting
Directly writing user input (for example, an HTTP request parameter) to a webpage, without properly sanitizing the input first, allows for a cross-site scripting vulnerability.


## Recommendation
To guard against cross-site scripting, escape user input before writing it to the page. Some frameworks, such as Rails, perform this escaping implicitly and by default.

Take care when using methods such as `html_safe` or `raw`. They can be used to emit a string without escaping it, and should only be used when the string has already been manually escaped (for example, with the Rails `html_escape` method), or when the content is otherwise guaranteed to be safe (such as a hard-coded string).


## Example
The following example is safe because the `params[:user_name]` content within the output tags will be HTML-escaped automatically before being emitted.


```none
<p>Hello <%= params[:user_name] %>!</p>

```
However, the following example is unsafe because user-controlled input is emitted without escaping, since it is marked as `html_safe`.


```none
<p>Hello <%= params[:user_name].html_safe %>!</p>

```

## References
* OWASP: [XSS Ruby on Rails Cheatsheet](https://cheatsheetseries.owasp.org/cheatsheets/Ruby_on_Rails_Cheat_Sheet.html#cross-site-scripting-xss).
* Wikipedia: [Cross-site scripting](http://en.wikipedia.org/wiki/Cross-site_scripting).
* Common Weakness Enumeration: [CWE-79](https://cwe.mitre.org/data/definitions/79.html).
* Common Weakness Enumeration: [CWE-116](https://cwe.mitre.org/data/definitions/116.html).
