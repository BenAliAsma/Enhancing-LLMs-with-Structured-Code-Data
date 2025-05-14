# Incomplete multi-character sanitization
Sanitizing untrusted input is a common technique for preventing injection attacks and other security vulnerabilities. Regular expressions are often used to perform this sanitization. However, when the regular expression matches multiple consecutive characters, replacing it just once can result in the unsafe text re-appearing in the sanitized input.

Attackers can exploit this issue by crafting inputs that, when sanitized with an ineffective regular expression, still contain malicious code or content. This can lead to code execution, data exposure, or other vulnerabilities.


## Recommendation
To prevent this issue, it is highly recommended to use a well-tested sanitization library whenever possible. These libraries are more likely to handle corner cases and ensure effective sanitization.

If a library is not an option, you can consider alternative strategies to fix the issue. For example, applying the regular expression replacement repeatedly until no more replacements can be performed, or rewriting the regular expression to match single characters instead of the entire unsafe text.


## Example
Consider the following Ruby code that aims to remove all HTML comment start and end tags:

```ruby

str.gsub(/<!--|--!?>/, "")  

```
Given the input string "&lt;!&lt;!--- comment ---&gt;&gt;", the output will be "&lt;!-- comment --&gt;", which still contains an HTML comment.

One possible fix for this issue is to apply the regular expression replacement repeatedly until no more replacements can be performed. This ensures that the unsafe text does not re-appear in the sanitized input, effectively removing all instances of the targeted pattern:

```ruby

def remove_html_comments(input)  
  previous = nil  
  while input != previous  
    previous = input  
    input = input.gsub(/<!--|--!?>/, "")  
  end  
  input  
end  

```

## Example
Another example is the following regular expression intended to remove script tags:

```ruby

str.gsub(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/, "")  

```
If the input string is "&lt;scrip&lt;script&gt;is removed&lt;/script&gt;t&gt;alert(123)&lt;/script&gt;", the output will be "&lt;script&gt;alert(123)&lt;/script&gt;", which still contains a script tag.

A fix for this issue is to rewrite the regular expression to match single characters ("&lt;" and "&gt;") instead of the entire unsafe text. This simplifies the sanitization process and ensures that all potentially unsafe characters are removed:

```ruby

def remove_all_html_tags(input)  
  input.gsub(/<|>/, "")  
end  

```
Another potential fix is to use the popular `sanitize` gem. It keeps most of the safe HTML tags while removing all unsafe tags and attributes.

```ruby

require 'sanitize'

def sanitize_html(input)
  Sanitize.fragment(input)
end

```

## Example
Lastly, consider a path sanitizer using the regular expression `/\.\.\//`:

```ruby

str.gsub(/\.\.\//, "")  

```
The regular expression attempts to strip out all occurences of `/../` from `str`. This will not work as expected: for the string `/./.././`, for example, it will remove the single occurrence of `/../` in the middle, but the remainder of the string then becomes `/../`, which is another instance of the substring we were trying to remove.

A possible fix for this issue is to use the `File.sanitize` function from the Ruby Facets gem. This library is specifically designed to handle path sanitization, and should handle all corner cases and ensure effective sanitization:

```ruby

require 'facets'  
  
def sanitize_path(input)  
  File.sanitize(input)  
end  

```

## References
* OWASP Top 10: [A1 Injection](https://www.owasp.org/index.php/Top_10-2017_A1-Injection).
* Stack Overflow: [Removing all script tags from HTML with JS regular expression](https://stackoverflow.com/questions/6659351/removing-all-script-tags-from-html-with-js-regular-expression).
* Common Weakness Enumeration: [CWE-20](https://cwe.mitre.org/data/definitions/20.html).
* Common Weakness Enumeration: [CWE-80](https://cwe.mitre.org/data/definitions/80.html).
* Common Weakness Enumeration: [CWE-116](https://cwe.mitre.org/data/definitions/116.html).
