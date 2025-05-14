# Stored cross-site scripting
Directly writing an uncontrolled stored value (for example, a database field) to a webpage, without properly sanitizing the value first, allows for a cross-site scripting vulnerability.

This kind of vulnerability is also called *stored* cross-site scripting, to distinguish it from other types of cross-site scripting.


## Recommendation
To guard against stored cross-site scripting, consider escaping before using uncontrolled stored values to create HTML content. Some frameworks, such as Rails, perform this escaping implicitly and by default.

Take care when using methods such as `html_safe` or `raw`. They can be used to emit a string without escaping it, and should only be used when the string has already been manually escaped (for example, with the Rails `html_escape` method), or when the content is otherwise guaranteed to be safe (such as a hard-coded string).


## Example
The following example is safe because the `user.name` content within the output tags will be HTML-escaped automatically before being emitted.


```none
<% user = User.find(1) %>
<p>Hello <%= user.name %>!</p>

```
However, the following example may be unsafe because `user.name` is emitted without escaping, since it is marked as `html_safe`. If the `name` is not sanitized before being written to the database, then an attacker could use this to insert arbitrary content into the HTML output, including scripts.


```none
<% user = User.find(1) %>
<p>Hello <%= user.name.html_safe %>!</p>

```
In the next example, content from a file on disk is inserted literally into HTML content. This approach is sometimes used to load script content, such as extensions for a web application, from files on disk. Care should taken in these cases to ensure both that the loaded files are trusted, and that the file cannot be modified by untrusted users.


```none
<script>
  <%= File.read(File.join(SCRIPT_DIR, "script.js")).html_safe %>
</script>

```

## References
* OWASP: [XSS Ruby on Rails Cheatsheet](https://cheatsheetseries.owasp.org/cheatsheets/Ruby_on_Rails_Cheat_Sheet.html#cross-site-scripting-xss).
* Wikipedia: [Cross-site scripting](http://en.wikipedia.org/wiki/Cross-site_scripting).
* Common Weakness Enumeration: [CWE-79](https://cwe.mitre.org/data/definitions/79.html).
* Common Weakness Enumeration: [CWE-116](https://cwe.mitre.org/data/definitions/116.html).
