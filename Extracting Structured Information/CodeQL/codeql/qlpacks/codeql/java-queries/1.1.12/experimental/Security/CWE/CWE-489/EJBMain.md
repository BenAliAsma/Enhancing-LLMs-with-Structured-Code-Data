# Main Method in Enterprise Java Bean
Debug code can create unintended entry points in a deployed Java EE web application therefore should never make into production. There is no reason to have a main method in a Java EE web application. Having a main method in the Java EE application increases the attack surface that an attacker can exploit to attack the application logic.


## Recommendation
Remove the main method from enterprise beans.


## Example
The following example shows two ways of implementing enterprise beans. In the 'BAD' case, a main method is implemented. In the 'GOOD' case, no main method is implemented.


```java
public class EJBMain implements SessionBean {
    /**
     * Create the session bean (empty implementation)
     */
    public void ejbCreate() throws javax.ejb.CreateException {
        System.out.println("EJBMain:ejbCreate()");
    }

    public void ejbActivate() throws javax.ejb.EJBException, java.rmi.RemoteException {
    }

    public void ejbPassivate() throws javax.ejb.EJBException, java.rmi.RemoteException {
    }

    public void ejbRemove() throws javax.ejb.EJBException, java.rmi.RemoteException {
    }

    public void setSessionContext(SessionContext parm1) throws javax.ejb.EJBException, java.rmi.RemoteException {
    }

    public String doService() {
        return null;
    }

    // BAD - Implement a main method in session bean.
    public static void main(String[] args) throws Exception {
        EJBMain b = new EJBMain();
        b.doService();
    }

    // GOOD - Not to have a main method in session bean.
}

```

## References
* SonarSource: [Web applications should not have a "main" method](https://rules.sonarsource.com/java/tag/owasp/RSPEC-2653)
* Carnegie Mellon University: [ENV06-J. Production code must not contain debugging entry points](https://wiki.sei.cmu.edu/confluence/display/java/ENV06-J.+Production+code+must+not+contain+debugging+entry+points)
* Common Weakness Enumeration: [CWE-489](https://cwe.mitre.org/data/definitions/489.html).
