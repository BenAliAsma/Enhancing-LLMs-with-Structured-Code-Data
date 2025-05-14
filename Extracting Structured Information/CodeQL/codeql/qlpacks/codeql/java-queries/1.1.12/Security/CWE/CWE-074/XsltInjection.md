# XSLT transformation with user-controlled stylesheet
XSLT (Extensible Stylesheet Language Transformations) is a language for transforming XML documents into other XML documents or other formats. Processing unvalidated XSLT stylesheets can allow attackers to read arbitrary files from the filesystem or to execute arbitrary code.


## Recommendation
The general recommendation is to not process untrusted XSLT stylesheets. If user-provided stylesheets must be processed, enable the secure processing mode.


## Example
In the following examples, the code accepts an XSLT stylesheet from the user and processes it.

In the first example, the user-provided XSLT stylesheet is parsed and processed.

In the second example, secure processing mode is enabled.


```java
import javax.xml.XMLConstants;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.stream.StreamResult;
import javax.xml.transform.stream.StreamSource;

public void transform(Socket socket, String inputXml) throws Exception {
  StreamSource xslt = new StreamSource(socket.getInputStream());
  StreamSource xml = new StreamSource(new StringReader(inputXml));
  StringWriter result = new StringWriter();
  TransformerFactory factory = TransformerFactory.newInstance();

  // BAD: User provided XSLT stylesheet is processed
  factory.newTransformer(xslt).transform(xml, new StreamResult(result));

  // GOOD: The secure processing mode is enabled
  factory.setFeature(XMLConstants.FEATURE_SECURE_PROCESSING, true);
  factory.newTransformer(xslt).transform(xml, new StreamResult(result));
}  
```

## References
* Wikipedia: [XSLT](https://en.wikipedia.org/wiki/XSLT).
* The Java Tutorials: [Transforming XML Data with XSLT](https://docs.oracle.com/javase/tutorial/jaxp/xslt/transformingXML.html).
* [XSLT Injection Basics](https://blog.hunniccyber.com/ektron-cms-remote-code-execution-xslt-transform-injection-java/).
* Common Weakness Enumeration: [CWE-74](https://cwe.mitre.org/data/definitions/74.html).
