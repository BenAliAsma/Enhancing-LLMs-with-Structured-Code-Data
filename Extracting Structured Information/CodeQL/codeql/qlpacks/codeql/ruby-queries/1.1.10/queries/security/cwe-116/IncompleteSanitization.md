# Incomplete string escaping or encoding
Sanitizing untrusted input is a common technique for preventing injection attacks such as SQL injection or cross-site scripting. Usually, this is done by escaping meta-characters such as quotes in a domain-specific way so that they are treated as normal characters.

However, directly using the `String#sub` method to perform escaping is notoriously error-prone. Common mistakes include only replacing the first occurrence of a meta-character, or backslash-escaping various meta-characters but not the backslash itself.

In the former case, later meta-characters are left undisturbed and can be used to subvert the sanitization. In the latter case, preceding a meta-character with a backslash leads to the backslash being escaped, but the meta-character appearing un-escaped, which again makes the sanitization ineffective.

Even if the escaped string is not used in a security-critical context, incomplete escaping may still have undesirable effects, such as badly rendered or confusing output.


## Recommendation
Use a (well-tested) sanitization library if at all possible. These libraries are much more likely to handle corner cases correctly than a custom implementation.

An even safer alternative is to design the application so that sanitization is not needed. Otherwise, make sure to use `String#gsub` rather than `String#sub`, to ensure that all occurrences are replaced, and remember to escape backslashes if applicable.


## Example
As an example, assume that we want to embed a user-controlled string `account_number` into a SQL query as part of a string literal. To avoid SQL injection, we need to ensure that the string does not contain un-escaped single-quote characters. The following method attempts to ensure this by doubling single quotes, and thereby escaping them:


```ruby
def escape_quotes(s)
  s.sub "'", "''"
end
```
As written, this sanitizer is ineffective: `String#sub` will replace only the *first* occurrence of that string.

As mentioned above, the method `escape_quotes` should be replaced with a purpose-built sanitizer, such as `ActiveRecord::Base::sanitize_sql` in Rails, or by using ORM methods that automatically sanitize parameters.

If this is not an option, `escape_quotes` should be rewritten to use the `String#gsub` method instead:


```ruby
def escape_quotes(s)
  s.gsub "'", "''"
end
```

## References
* OWASP Top 10: [A1 Injection](https://www.owasp.org/index.php/Top_10-2017_A1-Injection).
* Rails: [ActiveRecord::Base::sanitize_sql](https://api.rubyonrails.org/classes/ActiveRecord/Sanitization/ClassMethods.html).
* Common Weakness Enumeration: [CWE-20](https://cwe.mitre.org/data/definitions/20.html).
* Common Weakness Enumeration: [CWE-80](https://cwe.mitre.org/data/definitions/80.html).
* Common Weakness Enumeration: [CWE-116](https://cwe.mitre.org/data/definitions/116.html).
