# Exposure of sensitive information to notifications
Sensitive information such as passwords or two-factor authentication (2FA) codes should not be exposed in a system notification. Notifications should not be considered secure, as other untrusted applications may be able to use a `NotificationListenerService` to read the contents of notifications.


## Recommendation
Do not expose sensitive data in notifications.


## Example
In the following sample, the `password` is sent as part of a notification. This can allow another application to read this password.


```java
// BAD: `password` is exposed in a notification.
void confirmPassword(String password) {
    NotificationManager manager = NotificationManager.from(this);
    manager.send(
        new Notification.Builder(this, CHANNEL_ID)
        .setContentText("Your password is: " + password)
        .build());
}
```

## References
* OWASP Mobile Application Security: [Android Data Storage - Application Notifications](https://mas.owasp.org/MASTG/Android/0x05d-Testing-Data-Storage/#app-notifications)
* Common Weakness Enumeration: [CWE-200](https://cwe.mitre.org/data/definitions/200.html).
