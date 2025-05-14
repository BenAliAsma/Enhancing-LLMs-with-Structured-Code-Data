# XPath injection
If an XPath expression is built using string concatenation, and the components of the concatenation include user input, it makes it very easy for a user to create a malicious XPath expression.


## Recommendation
If user input must be included in an XPath expression, either sanitize the data or pre-compile the query and use variable references to include the user input.

XPath injection can also be prevented by using XQuery.


## Example
In the first three examples, the code accepts a name and password specified by the user, and uses this unvalidated and unsanitized value in an XPath expression. This is vulnerable to the user providing special characters or string sequences that change the meaning of the XPath expression to search for different values.

In the fourth example, the code uses `setXPathVariableResolver` which prevents XPath injection.

The final two examples are for dom4j. They show an example of XPath injection and one method of preventing it.


```java
final String xmlStr = "<users>" + 
                        "   <user name=\"aaa\" pass=\"pass1\"></user>" + 
                        "   <user name=\"bbb\" pass=\"pass2\"></user>" + 
                        "</users>";
try {
    DocumentBuilderFactory domFactory = DocumentBuilderFactory.newInstance();
    domFactory.setNamespaceAware(true);
    DocumentBuilder builder = domFactory.newDocumentBuilder();
    //Document doc = builder.parse("user.xml");
    Document doc = builder.parse(new InputSource(new StringReader(xmlStr)));

    XPathFactory factory = XPathFactory.newInstance();
    XPath xpath = factory.newXPath();

    // Injectable data
    String user = request.getParameter("user");
    String pass = request.getParameter("pass");
    if (user != null && pass != null) {
        boolean isExist = false;

        // Bad expression
        String expression1 = "/users/user[@name='" + user + "' and @pass='" + pass + "']";
        isExist = (boolean)xpath.evaluate(expression1, doc, XPathConstants.BOOLEAN);
        System.out.println(isExist);

        // Bad expression
        XPathExpression expression2 = xpath.compile("/users/user[@name='" + user + "' and @pass='" + pass + "']");
        isExist = (boolean)expression2.evaluate(doc, XPathConstants.BOOLEAN);
        System.out.println(isExist);

        // Bad expression
        StringBuffer sb = new StringBuffer("/users/user[@name=");
        sb.append(user);
        sb.append("' and @pass='");
        sb.append(pass);
        sb.append("']");
        String query = sb.toString();
        XPathExpression expression3 = xpath.compile(query);
        isExist = (boolean)expression3.evaluate(doc, XPathConstants.BOOLEAN);
        System.out.println(isExist);

        // Good expression
        String expression4 = "/users/user[@name=$user and @pass=$pass]";
        xpath.setXPathVariableResolver(v -> {
        switch (v.getLocalPart()) {
            case "user":
                return user;
            case "pass":
                return pass;
            default:
                throw new IllegalArgumentException();
            }
        });
        isExist = (boolean)xpath.evaluate(expression4, doc, XPathConstants.BOOLEAN);
        System.out.println(isExist);


        // Bad Dom4j 
        org.dom4j.io.SAXReader reader = new org.dom4j.io.SAXReader();
        org.dom4j.Document document = reader.read(new InputSource(new StringReader(xmlStr)));
        isExist = document.selectSingleNode("/users/user[@name='" + user + "' and @pass='" + pass + "']") != null;
        // or document.selectNodes
        System.out.println(isExist);

        // Good Dom4j
        org.jaxen.SimpleVariableContext svc = new org.jaxen.SimpleVariableContext();
        svc.setVariableValue("user", user);
        svc.setVariableValue("pass", pass);
        String xpathString = "/users/user[@name=$user and @pass=$pass]";
        org.dom4j.XPath safeXPath = document.createXPath(xpathString);
        safeXPath.setVariableContext(svc);
        isExist = safeXPath.selectSingleNode(document) != null;
        System.out.println(isExist);
    }
} catch (ParserConfigurationException e) {

} catch (SAXException e) {

} catch (XPathExpressionException e) {

} catch (org.dom4j.DocumentException e) {

}
```

## References
* OWASP: [Testing for XPath Injection](https://owasp.org/www-project-web-security-testing-guide/latest/4-Web_Application_Security_Testing/07-Input_Validation_Testing/09-Testing_for_XPath_Injection).
* OWASP: [XPath Injection](https://owasp.org/www-community/attacks/XPATH_Injection).
* Common Weakness Enumeration: [CWE-643](https://cwe.mitre.org/data/definitions/643.html).
