# Android Intent redirection
An exported Android component that obtains a user-provided Intent and uses it to launch another component can be exploited to obtain access to private, unexported components of the same app or to launch other apps' components on behalf of the victim app.


## Recommendation
Do not export components that start other components from a user-provided Intent. They can be made private by setting the `android:exported` property to `false` in the app's Android Manifest.

If this is not possible, restrict either which apps can send Intents to the affected component, or which components can be started from it.


## Example
The following snippet contains three examples. In the first example, an arbitrary component can be started from the externally provided `forward_intent` Intent. In the second example, the destination component of the Intent is first checked to make sure it is safe. In the third example, the component that created the Intent is first checked to make sure it comes from a trusted origin.


```java
// BAD: A user-provided Intent is used to launch an arbitrary component
Intent forwardIntent = (Intent) getIntent().getParcelableExtra("forward_intent");
startActivity(forwardIntent);

// GOOD: The destination component is checked before launching it
Intent forwardIntent = (Intent) getIntent().getParcelableExtra("forward_intent");
ComponentName destinationComponent = forwardIntent.resolveActivity(getPackageManager());
if (destinationComponent.getPackageName().equals("safe.package") && 
    destinationComponent.getClassName().equals("SafeClass")) {
    startActivity(forwardIntent);
}

// GOOD: The component that sent the Intent is checked before launching the destination component
Intent forwardIntent = (Intent) getIntent().getParcelableExtra("forward_intent");
ComponentName originComponent = getCallingActivity();
if (originComponent.getPackageName().equals("trusted.package") && originComponent.getClassName().equals("TrustedClass")) {
    startActivity(forwardIntent);
}

```

## References
* Google: [Remediation for Intent Redirection Vulnerability](https://support.google.com/faqs/answer/9267555?hl=en).
* OWASP Mobile Security Testing Guide: [Intents](https://mobile-security.gitbook.io/mobile-security-testing-guide/android-testing-guide/0x05a-platform-overview#intents).
* Android Developers: [The android:exported attribute](https://developer.android.com/guide/topics/manifest/activity-element#exported).
* Common Weakness Enumeration: [CWE-926](https://cwe.mitre.org/data/definitions/926.html).
* Common Weakness Enumeration: [CWE-940](https://cwe.mitre.org/data/definitions/940.html).
