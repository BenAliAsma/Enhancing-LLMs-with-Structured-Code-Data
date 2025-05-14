# Deserialization of user-controlled data
Deserializing untrusted data using any deserialization framework that allows the construction of arbitrary serializable objects is easily exploitable and in many cases allows an attacker to execute arbitrary code. Even before a deserialized object is returned to the caller of a deserialization method a lot of code may have been executed, including static initializers, constructors, and finalizers. Automatic deserialization of fields means that an attacker may craft a nested combination of objects on which the executed initialization code may have unforeseen effects, such as the execution of arbitrary code.

There are many different serialization frameworks. This query currently supports Pickle, Marshal and Yaml.


## Recommendation
Avoid deserialization of untrusted data if at all possible. If the architecture permits it then use other formats instead of serialized objects, for example JSON.

If you need to use YAML, use the `yaml.safe_load` function.


## Example
The following example calls `pickle.loads` directly on a value provided by an incoming HTTP request. Pickle then creates a new value from untrusted data, and is therefore inherently unsafe.


```python

from django.conf.urls import url
import pickle

def unsafe(pickled):
    return pickle.loads(pickled)

urlpatterns = [
    url(r'^(?P<object>.*)$', unsafe)
]
```
Changing the code to use `json.loads` instead of `pickle.loads` removes the vulnerability.


```python

from django.conf.urls import url
import json

def safe(pickled):
    return json.loads(pickled)

urlpatterns = [
    url(r'^(?P<object>.*)$', safe)
]

```

## References
* OWASP vulnerability description: [Deserialization of untrusted data](https://www.owasp.org/index.php/Deserialization_of_untrusted_data).
* OWASP guidance on deserializing objects: [Deserialization Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Deserialization_Cheat_Sheet.html).
* Talks by Chris Frohoff &amp; Gabriel Lawrence: [ AppSecCali 2015: Marshalling Pickles - how deserializing objects will ruin your day](http://frohoff.github.io/appseccali-marshalling-pickles/)
* Common Weakness Enumeration: [CWE-502](https://cwe.mitre.org/data/definitions/502.html).
