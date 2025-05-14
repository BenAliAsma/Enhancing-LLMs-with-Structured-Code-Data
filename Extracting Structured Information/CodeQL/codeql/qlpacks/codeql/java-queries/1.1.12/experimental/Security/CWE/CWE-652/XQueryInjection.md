# XQuery query built from user-controlled sources
The software uses external input to dynamically construct an XQuery expression which is then used to retrieve data from an XML database. However, the input is not neutralized, or is incorrectly neutralized, which allows an attacker to control the structure of the query.


## Recommendation
Use parameterized queries. This will help ensure the program retains control of the query structure.


## Example
The following example compares building a query by string concatenation (bad) vs. using `bindString` to parameterize the query (good).


```java
import javax.servlet.http.HttpServletRequest;
import javax.xml.namespace.QName;
import javax.xml.xquery.XQConnection;
import javax.xml.xquery.XQDataSource;
import javax.xml.xquery.XQException;
import javax.xml.xquery.XQItemType;
import javax.xml.xquery.XQPreparedExpression;
import javax.xml.xquery.XQResultSequence;
import net.sf.saxon.xqj.SaxonXQDataSource;

public void bad(HttpServletRequest request) throws XQException {
    String name = request.getParameter("name");
    XQDataSource ds = new SaxonXQDataSource();
    XQConnection conn = ds.getConnection();
    String query = "for $user in doc(\"users.xml\")/Users/User[name='" + name + "'] return $user/password";
    XQPreparedExpression xqpe = conn.prepareExpression(query);
    XQResultSequence result = xqpe.executeQuery();
    while (result.next()){
        System.out.println(result.getItemAsString(null));
    }
}

public void bad1(HttpServletRequest request) throws XQException {
    String name = request.getParameter("name");
    XQDataSource xqds = new SaxonXQDataSource();
    String query = "for $user in doc(\"users.xml\")/Users/User[name='" + name + "'] return $user/password";
    XQConnection conn = xqds.getConnection();
    XQExpression expr = conn.createExpression();
    XQResultSequence result = expr.executeQuery(query);
    while (result.next()){
        System.out.println(result.getItemAsString(null));
    }
}

public void bad2(HttpServletRequest request) throws XQException {
    String name = request.getParameter("name");
    XQDataSource xqds = new SaxonXQDataSource();
    XQConnection conn = xqds.getConnection();
    XQExpression expr = conn.createExpression();
    //bad code
    expr.executeCommand(name);
}

public void good(HttpServletRequest request) throws XQException {
    String name = request.getParameter("name");
    XQDataSource ds = new SaxonXQDataSource();
    XQConnection conn = ds.getConnection();
    String query = "declare variable $name as xs:string external;"
            + " for $user in doc(\"users.xml\")/Users/User[name=$name] return $user/password";
    XQPreparedExpression xqpe = conn.prepareExpression(query);
    xqpe.bindString(new QName("name"), name, conn.createAtomicType(XQItemType.XQBASETYPE_STRING));
    XQResultSequence result = xqpe.executeQuery();
    while (result.next()){
        System.out.println(result.getItemAsString(null));
    }
}

public void good1(HttpServletRequest request) throws XQException {
    String name = request.getParameter("name");
    String query = "declare variable $name as xs:string external;"
            + " for $user in doc(\"users.xml\")/Users/User[name=$name] return $user/password";
    XQDataSource xqds = new SaxonXQDataSource();
    XQConnection conn = xqds.getConnection();
    XQExpression expr = conn.createExpression();
    expr.bindString(new QName("name"), name, conn.createAtomicType(XQItemType.XQBASETYPE_STRING));
    XQResultSequence result = expr.executeQuery(query);
    while (result.next()){
        System.out.println(result.getItemAsString(null));
    }
}
```

## References
* Balisage: [XQuery Injection](https://www.balisage.net/Proceedings/vol7/html/Vlist02/BalisageVol7-Vlist02.html).
* Common Weakness Enumeration: [CWE-652](https://cwe.mitre.org/data/definitions/652.html).
