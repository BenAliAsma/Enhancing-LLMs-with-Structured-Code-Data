# Insecure JavaMail SSL Configuration
JavaMail is commonly used in Java applications to send emails. There are popular third-party libraries like Apache Commons Email which are built on JavaMail and facilitate integration. Authenticated mail sessions require user credentials and mail sessions can require SSL/TLS authentication. It is a common security vulnerability that host-specific certificate data is not validated or is incorrectly validated. Failing to validate the certificate makes the SSL session susceptible to a man-in-the-middle attack.

This query checks whether the SSL certificate is validated when credentials are used and SSL is enabled in email communications.

The query has code for both plain JavaMail invocation and mailing through Apache SimpleMail to make it more comprehensive.


## Recommendation
Validate SSL certificate when sensitive information is sent in email communications.


## Example
The following two examples show two ways of configuring secure emails through JavaMail or Apache SimpleMail. In the 'BAD' case, credentials are sent in an SSL session without certificate validation. In the 'GOOD' case, the certificate is validated.


```java
import java.util.Properties;

import javax.activation.DataSource;
import javax.mail.Authenticator;
import javax.mail.Message;
import javax.mail.MessagingException;
import javax.mail.PasswordAuthentication;
import javax.mail.Session;

import org.apache.logging.log4j.util.PropertiesUtil;

class JavaMail {
    public static void main(String[] args) {
      // BAD: Don't have server certificate check
      {
		final Properties properties = PropertiesUtil.getSystemProperties();
		properties.put("mail.transport.protocol", "protocol");
		properties.put("mail.smtp.host", "hostname");
		properties.put("mail.smtp.socketFactory.class", "classname");

		final Authenticator authenticator = buildAuthenticator("username", "password");
		if (null != authenticator) {
			properties.put("mail.smtp.auth", "true");
		}
		final Session session = Session.getInstance(properties, authenticator);
      }

      // GOOD: Have server certificate check
      {
		final Properties properties = PropertiesUtil.getSystemProperties();
		properties.put("mail.transport.protocol", "protocol");
		properties.put("mail.smtp.host", "hostname");
		properties.put("mail.smtp.socketFactory.class", "classname");

		final Authenticator authenticator = buildAuthenticator("username", "password");
		if (null != authenticator) {
			properties.put("mail.smtp.auth", "true");
			properties.put("mail.smtp.ssl.checkserveridentity", "true");
		}
		final Session session = Session.getInstance(properties, authenticator);
      }
    }
}
```

```java
import org.apache.commons.mail.DefaultAuthenticator;
import org.apache.commons.mail.Email;
import org.apache.commons.mail.EmailException;
import org.apache.commons.mail.SimpleEmail;

class SimpleMail {
    public static void main(String[] args) throws EmailException {
      // BAD: Don't have setSSLCheckServerIdentity set or set as false    
      {
        Email email = new SimpleEmail();
        email.setHostName("hostName");
        email.setSmtpPort(25);
        email.setAuthenticator(new DefaultAuthenticator("username", "password"));
        email.setSSLOnConnect(true);
        
        //email.setSSLCheckServerIdentity(false);
        email.setFrom("fromAddress");
        email.setSubject("subject");
        email.setMsg("body");
        email.addTo("toAddress");
        email.send();
      }

      // GOOD: Have setSSLCheckServerIdentity set to true
      {
        Email email = new SimpleEmail();
        email.setHostName("hostName");
        email.setSmtpPort(25);
        email.setAuthenticator(new DefaultAuthenticator("username", "password"));
        email.setSSLOnConnect(true);

        email.setSSLCheckServerIdentity(true);
        email.setFrom("fromAddress");
        email.setSubject("subject");
        email.setMsg("body");
        email.addTo("toAddress");
        email.send();
      }
    }
}
```

## References
* Jakarta Mail: [SSL Notes](https://eclipse-ee4j.github.io/mail/docs/SSLNOTES.txt).
* Apache Commons: [Email security](https://commons.apache.org/proper/commons-email/userguide.html#Security).
* Log4j2: [Add support for specifying an SSL configuration for SmtpAppender (CVE-2020-9488)](https://issues.apache.org/jira/browse/LOG4J2-2819).
* Common Weakness Enumeration: [CWE-297](https://cwe.mitre.org/data/definitions/297.html).
