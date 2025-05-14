# Deserialization of user-controlled data
Deserializing untrusted data using any method that allows the construction of arbitrary objects is easily exploitable and, in many cases, allows an attacker to execute arbitrary code.


## Recommendation
Avoid deserialization of untrusted data if possible. If the architecture permits it, use serialization formats that cannot represent arbitrary objects. For libraries that support it, such as the Ruby standard library's `JSON` module, ensure that the parser is configured to disable deserialization of arbitrary objects.

If deserializing an untrusted YAML document using the `psych` gem, prefer the `safe_load` and `safe_load_file` methods over `load` and `load_file`, as the former will safely handle untrusted data. Avoid passing untrusted data to the `load_stream` method. In `psych` version 4.0.0 and above, the `load` method can safely be used.

If deserializing an untrusted XML document using the `ox` gem, do not use `parse_obj` and `load` using the non-default :object mode. Instead use the `load` method in the default mode or better explicitly set a safe mode such as :hash.

To safely deserialize [Property List](https://en.wikipedia.org/wiki/Property_list) files using the `plist` gem, ensure that you pass `marshal: false` when calling `Plist.parse_xml`.


## Example
The following example calls the `Marshal.load`, `JSON.load`, `YAML.load`, `Oj.load` and `Ox.parse_obj` methods on data from an HTTP request. Since these methods are capable of deserializing to arbitrary objects, this is inherently unsafe.


```ruby
require 'json'
require 'yaml'
require 'oj'

class UserController < ActionController::Base
  def marshal_example
    data = Base64.decode64 params[:data]
    object = Marshal.load data
    # ...
  end

  def json_example
    object = JSON.load params[:json]
    # ...
  end

  def yaml_example
    object = YAML.load params[:yaml]
    # ...
  end

  def oj_example
    object = Oj.load params[:json]
    # ...
  end

  def ox_example
    object = Ox.parse_obj params[:xml]
    # ...
  end
end
```
Using `JSON.parse` and `YAML.safe_load` instead, as in the following example, removes the vulnerability. Similarly, calling `Oj.load` with any mode other than `:object` is safe, as is calling `Oj.safe_load`. Note that there is no safe way to deserialize untrusted data using `Marshal`.


```ruby
require 'json'

class UserController < ActionController::Base
  def safe_json_example
    object = JSON.parse params[:json]
    # ...
  end

  def safe_yaml_example
    object = YAML.safe_load params[:yaml]
    # ...
  end

  def safe_oj_example
    object = Oj.load params[:yaml], { mode: :strict }
    # or
    object = Oj.safe_load params[:yaml]
    # ...
  end
end
```

## References
* OWASP vulnerability description: [deserialization of untrusted data](https://www.owasp.org/index.php/Deserialization_of_untrusted_data).
* Ruby documentation: [guidance on deserializing objects safely](https://docs.ruby-lang.org/en/3.0.0/doc/security_rdoc.html).
* Ruby documentation: [security guidance on the Marshal library](https://ruby-doc.org/core-3.0.2/Marshal.html#module-Marshal-label-Security+considerations).
* Ruby documentation: [security guidance on JSON.load](https://ruby-doc.org/stdlib-3.0.2/libdoc/json/rdoc/JSON.html#method-i-load).
* Ruby documentation: [security guidance on the YAML library](https://ruby-doc.org/stdlib-3.0.2/libdoc/yaml/rdoc/YAML.html#module-YAML-label-Security).
* You can read that how unsafe yaml load methods can lead to code executions: [Universal Deserialisation Gadget for Ruby 2.x-3.x ](https://devcraft.io/2021/01/07/universal-deserialisation-gadget-for-ruby-2-x-3-x.html).
* Common Weakness Enumeration: [CWE-502](https://cwe.mitre.org/data/definitions/502.html).
