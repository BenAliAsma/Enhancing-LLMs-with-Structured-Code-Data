# Android debuggable attribute enabled
The Android manifest file defines configuration settings for Android applications. In this file, the `android:debuggable` attribute of the `application` element can be used to define whether or not the application can be debugged. When set to `true`, this attribute will allow the application to be debugged even when running on a device in user mode.

When a debugger is enabled, it could allow for entry points in the application or reveal sensitive information. As a result, `android:debuggable` should only be enabled during development and should be disabled in production builds.


## Recommendation
In Android applications, either set the `android:debuggable` attribute to `false`, or do not include it in the manifest. The default value, when not included, is `false`.


## Example
In the example below, the `android:debuggable` attribute is set to `true`.


```xml
<manifest ... >
    <!-- BAD: 'android:debuggable' set to 'true' -->
    <application
        android:debuggable="true">
        <activity ... >
        </activity>
    </application>
</manifest>

```
The corrected version sets the `android:debuggable` attribute to `false`.


```xml
<manifest ... >
    <!-- GOOD: 'android:debuggable' set to 'false' -->
    <application
        android:debuggable="false">
        <activity ... >
        </activity>
    </application>
</manifest>

```

## References
* Android Developers: [App Manifest Overview](https://developer.android.com/guide/topics/manifest/manifest-intro).
* Android Developers: [The android:debuggable attribute](https://developer.android.com/guide/topics/manifest/application-element#debug).
* Android Developers: [Enable debugging](https://developer.android.com/studio/debug#enable-debug).
* Common Weakness Enumeration: [CWE-489](https://cwe.mitre.org/data/definitions/489.html).
