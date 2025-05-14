# Badly anchored regular expression
Regular expressions in Ruby can use anchors to match the beginning and end of a string. However, if the `^` and `$` anchors are used, the regular expression can match a single line of a multi-line string. This allows bad actors to bypass your regular expression checks and inject malicious input.


## Recommendation
Use the `\A` and `\z` anchors since these anchors will always match the beginning and end of the string, even if the string contains newlines.


## Example
The following (bad) example code uses a regular expression to check that a string contains only digits.


```ruby
def bad(input) 
    raise "Bad input" unless input =~ /^[0-9]+$/

    # ....
end
```
The regular expression `/^[0-9]+$/` will match a single line of a multi-line string, which may not be the intended behavior. The following (good) example code uses the regular expression `\A[0-9]+\z` to match the entire input string.


```ruby
def good(input)
    raise "Bad input" unless input =~ /\A[0-9]+\z/

    # ....
end
```

## References
* Ruby documentation: [Anchors](https://ruby-doc.org/3.2.0/Regexp.html#class-Regexp-label-Anchors)
* Common Weakness Enumeration: [CWE-20](https://cwe.mitre.org/data/definitions/20.html).
