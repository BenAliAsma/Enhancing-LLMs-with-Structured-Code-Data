# Uncontrolled data used in path expression
Accessing files using paths constructed from user-controlled data can allow an attacker to access unexpected resources. This can result in sensitive information being revealed or deleted, or an attacker being able to influence behavior by modifying unexpected files.


## Recommendation
Validate user input before using it to construct a file path, either using an off-the-shelf library like `ActiveStorage::Filename#sanitized` in Rails, or by performing custom validation.

Ideally, follow these rules:

* Do not allow more than a single "." character.
* Do not allow directory separators such as "/" or "\\" (depending on the file system).
* Do not rely on simply replacing problematic sequences such as "../". For example, after applying this filter to ".../...//", the resulting string would still be "../".
* Use a whitelist of known good patterns.

## Example
In the first example, a file name is read from an HTTP request and then used to access a file. However, a malicious user could enter a file name which is an absolute path, such as `"/etc/passwd"`.

In the second example, it appears that the user is restricted to opening a file within the `"user"` home directory. However, a malicious user could enter a file name containing special characters. For example, the string `"../../etc/passwd"` will result in the code reading the file located at `"/home/user/../../etc/passwd"`, which is the system's password file. This file would then be sent back to the user, giving them access to all the system's passwords.


```ruby
class FilesController < ActionController::Base
  def first_example
    # BAD: This could read any file on the file system
    @content = File.read params[:path]
  end

  def second_example
    # BAD: This could still read any file on the file system
    @content = File.read "/home/user/#{ params[:path] }"
  end
end

```

## References
* OWASP: [Path Traversal](https://owasp.org/www-community/attacks/Path_Traversal).
* Rails: [ActiveStorage::Filename\#sanitized](https://api.rubyonrails.org/classes/ActiveStorage/Filename.html#method-i-sanitized).
* Common Weakness Enumeration: [CWE-22](https://cwe.mitre.org/data/definitions/22.html).
* Common Weakness Enumeration: [CWE-23](https://cwe.mitre.org/data/definitions/23.html).
* Common Weakness Enumeration: [CWE-36](https://cwe.mitre.org/data/definitions/36.html).
* Common Weakness Enumeration: [CWE-73](https://cwe.mitre.org/data/definitions/73.html).
* Common Weakness Enumeration: [CWE-99](https://cwe.mitre.org/data/definitions/99.html).
