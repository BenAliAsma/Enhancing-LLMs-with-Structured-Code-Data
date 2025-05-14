# Deserialization of user-controlled data
Deserializing untrusted data using any deserialization framework that allows the construction of arbitrary serializable objects is easily exploitable and in many cases allows an attacker to execute arbitrary code. Even before a deserialized object is returned to the caller of a deserialization method a lot of code may have been executed, including static initializers, constructors, and finalizers. Automatic deserialization of fields means that an attacker may craft a nested combination of objects on which the executed initialization code may have unforeseen effects, such as the execution of arbitrary code.

There are many different serialization frameworks. This query currently supports Kryo, XmlDecoder, XStream, SnakeYaml, JYaml, JsonIO, YAMLBeans, HessianBurlap, Castor, Burlap, Jackson, Jabsorb, Jodd JSON, Flexjson, Gson, JMS, and Java IO serialization through `ObjectInputStream`/`ObjectOutputStream`.


## Recommendation
Avoid deserialization of untrusted data if at all possible. If the architecture permits it then use other formats instead of serialized objects, for example JSON or XML. However, these formats should not be deserialized into complex objects because this provides further opportunities for attack. For example, XML-based deserialization attacks are possible through libraries such as XStream and XmlDecoder.

Alternatively, a tightly controlled whitelist can limit the vulnerability of code, but be aware of the existence of so-called Bypass Gadgets, which can circumvent such protection measures.

Recommendations specific to particular frameworks supported by this query:

**FastJson** - `com.alibaba:fastjson`

* **Secure by Default**: Partially
* **Recommendation**: Call `com.alibaba.fastjson.parser.ParserConfig#setSafeMode` with the argument `true` before deserializing untrusted data.


**FasterXML** - `com.fasterxml.jackson.core:jackson-databind`

* **Secure by Default**: Yes
* **Recommendation**: Don't call `com.fasterxml.jackson.databind.ObjectMapper#enableDefaultTyping` and don't annotate any object fields with `com.fasterxml.jackson.annotation.JsonTypeInfo` passing either the `CLASS` or `MINIMAL_CLASS` values to the annotation. Read [this guide](https://cowtowncoder.medium.com/jackson-2-10-safe-default-typing-2d018f0ce2ba).


**Kryo** - `com.esotericsoftware:kryo` and `com.esotericsoftware:kryo5`

* **Secure by Default**: Yes for `com.esotericsoftware:kryo5` and for `com.esotericsoftware:kryo` &gt;= v5.0.0
* **Recommendation**: Don't call `com.esotericsoftware.kryo(5).Kryo#setRegistrationRequired` with the argument `false` on any `Kryo` instance that may deserialize untrusted data.


**ObjectInputStream** - `Java Standard Library`

* **Secure by Default**: No
* **Recommendation**: Use a validating input stream, such as `org.apache.commons.io.serialization.ValidatingObjectInputStream`.


**SnakeYAML** - `org.yaml:snakeyaml`

* **Secure by Default**: No
* **Recommendation**: Pass an instance of `org.yaml.snakeyaml.constructor.SafeConstructor` to `org.yaml.snakeyaml.Yaml`'s constructor before using it to deserialize untrusted data.


**XML Decoder** - `Standard Java Library`

* **Secure by Default**: No
* **Recommendation**: Do not use with untrusted user input.


**ObjectMesssage** - `Java EE/Jakarta EE`

* **Secure by Default**: Depends on the JMS implementation.
* **Recommendation**: Do not use with untrusted user input.



## Example
The following example calls `readObject` directly on an `ObjectInputStream` that is constructed from untrusted data, and is therefore inherently unsafe.


```java
public MyObject {
  public int field;
  MyObject(int field) {
    this.field = field;
  }
}

public MyObject deserialize(Socket sock) {
  try(ObjectInputStream in = new ObjectInputStream(sock.getInputStream())) {
    return (MyObject)in.readObject(); // unsafe
  }
}

```
Rewriting the communication protocol to only rely on reading primitive types from the input stream removes the vulnerability.


```java
public MyObject deserialize(Socket sock) {
  try(DataInputStream in = new DataInputStream(sock.getInputStream())) {
    return new MyObject(in.readInt());
  }
}

```

## References
* OWASP vulnerability description: [Deserialization of untrusted data](https://www.owasp.org/index.php/Deserialization_of_untrusted_data).
* OWASP guidance on deserializing objects: [Deserialization Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Deserialization_Cheat_Sheet.html).
* Talks by Chris Frohoff &amp; Gabriel Lawrence: [ AppSecCali 2015: Marshalling Pickles - how deserializing objects will ruin your day](http://frohoff.github.io/appseccali-marshalling-pickles/), [OWASP SD: Deserialize My Shorts: Or How I Learned to Start Worrying and Hate Java Object Deserialization](http://frohoff.github.io/owaspsd-deserialize-my-shorts/).
* Alvaro Muñoz &amp; Christian Schneider, RSAConference 2016: [Serial Killer: Silently Pwning Your Java Endpoints](https://speakerdeck.com/pwntester/serial-killer-silently-pwning-your-java-endpoints).
* SnakeYaml documentation on deserialization: [SnakeYaml deserialization](https://bitbucket.org/snakeyaml/snakeyaml/wiki/Documentation#markdown-header-loading-yaml).
* Hessian deserialization and related gadget chains: [Hessian deserialization](https://paper.seebug.org/1137/).
* Castor and Hessian java deserialization vulnerabilities: [Castor and Hessian deserialization](https://securitylab.github.com/research/hessian-java-deserialization-castor-vulnerabilities/).
* Remote code execution in JYaml library: [JYaml deserialization](https://www.cybersecurity-help.cz/vdb/SB2020022512).
* JsonIO deserialization vulnerabilities: [JsonIO deserialization](https://klezvirus.github.io/Advanced-Web-Hacking/Serialisation/).
* Research by Moritz Bechler: [Java Unmarshaller Security - Turning your data into code execution](https://www.github.com/mbechler/marshalsec/blob/master/marshalsec.pdf?raw=true)
* Blog posts by the developer of Jackson libraries: [On Jackson CVEs: Don’t Panic — Here is what you need to know](https://cowtowncoder.medium.com/on-jackson-cves-dont-panic-here-is-what-you-need-to-know-54cd0d6e8062) [Jackson 2.10: Safe Default Typing](https://cowtowncoder.medium.com/jackson-2-10-safe-default-typing-2d018f0ce2ba)
* Jabsorb documentation on deserialization: [Jabsorb JSON Serializer](https://github.com/Servoy/jabsorb/blob/master/src/org/jabsorb/).
* Jodd JSON documentation on deserialization: [JoddJson Parser](https://json.jodd.org/parser).
* RCE in Flexjson: [Flexjson deserialization](https://codewhitesec.blogspot.com/2020/03/liferay-portal-json-vulns.html).
* Android Intent deserialization vulnerabilities with GSON parser: [Insecure use of JSON parsers](https://blog.oversecured.com/Exploiting-memory-corruption-vulnerabilities-on-Android/#insecure-use-of-json-parsers).
* Research by Matthias Kaiser: [Pwning Your Java Messaging With Deserialization Vulnerabilities](https://www.blackhat.com/docs/us-16/materials/us-16-Kaiser-Pwning-Your-Java-Messaging-With-Deserialization-Vulnerabilities.pdf).
* Common Weakness Enumeration: [CWE-502](https://cwe.mitre.org/data/definitions/502.html).
