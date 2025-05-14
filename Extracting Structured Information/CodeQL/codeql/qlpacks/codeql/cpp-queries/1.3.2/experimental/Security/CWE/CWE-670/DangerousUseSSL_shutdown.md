# Dangerous use SSL_shutdown.
Incorrect closing of the connection leads to the creation of different states for the server and client, which can be exploited by an attacker.


## Example
The following example shows the incorrect and correct usage of function SSL_shutdown.


```cpp
...
SSL_shutdown(ssl); 
SSL_shutdown(ssl); // BAD
...
    switch ((ret = SSL_shutdown(ssl))) {
    case 1:
      break;
    case 0:
      ERR_clear_error();
      if (-1 != (ret = SSL_shutdown(ssl))) break; // GOOD
...

```

## References
* CERT Coding Standard: [EXP12-C. Do not ignore values returned by functions - SEI CERT C Coding Standard - Confluence](https://wiki.sei.cmu.edu/confluence/display/c/EXP12-C.+Do+not+ignore+values+returned+by+functions).
* Openssl.org: [SSL_shutdown - shut down a TLS/SSL connection](https://www.openssl.org/docs/man3.0/man3/SSL_shutdown.html).
* Common Weakness Enumeration: [CWE-670](https://cwe.mitre.org/data/definitions/670.html).
