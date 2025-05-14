# Leaking sensitive information through a ResultReceiver
If a `ResultReceiver` is obtained from an untrusted source, such as an `Intent` received by an exported component, do not send it sensitive data. Otherwise, the information may be leaked to a malicious application.


## Recommendation
Do not send sensitive data to an untrusted `ResultReceiver`.


## Example
In the following (bad) example, sensitive data is sent to an untrusted `ResultReceiver`.


```java
// BAD: Sensitive data is sent to an untrusted result receiver 
void bad(String password) {
    Intent intent = getIntent();
    ResultReceiver rec = intent.getParcelableExtra("Receiver");
    Bundle b = new Bundle();
    b.putCharSequence("pass", password);
    rec.send(0, b); 
}
```

## References
* Common Weakness Enumeration: [CWE-927](https://cwe.mitre.org/data/definitions/927.html).
