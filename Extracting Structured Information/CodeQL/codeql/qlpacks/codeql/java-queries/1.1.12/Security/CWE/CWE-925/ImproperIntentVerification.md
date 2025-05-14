# Improper verification of intent by broadcast receiver
When an Android application uses a `BroadcastReceiver` to receive intents, it is also able to receive explicit intents that are sent directly to it, regardless of its filter. Certain intent actions are only able to be sent by the operating system, not third-party applications. However, a `BroadcastReceiver` that is registered to receive system intents is still able to receive intents from a third-party application, so it should check that the intent received has the expected action. Otherwise, a third-party application could impersonate the system this way to cause unintended behavior, such as a denial of service.


## Example
In the following code, the `ShutdownReceiver` initiates a shutdown procedure upon receiving an intent, without checking that the received action is indeed `ACTION_SHUTDOWN`. This allows third-party applications to send explicit intents to this receiver to cause a denial of service.


```java
public class ShutdownReceiver extends BroadcastReceiver {
    @Override
    public void onReceive(final Context context, final Intent intent) {
        mainActivity.saveLocalData();
        mainActivity.stopActivity();
    }
}
```

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android" package="test">
    <application>
        <receiver android:name=".BootReceiverXml">
            <intent-filter>
                <action android:name="android.intent.action.BOOT_COMPLETED" />
            </intent-filter>
        </receiver>
    </application>
</manifest>
```

## Recommendation
In the `onReceive` method of a `BroadcastReceiver`, the action of the received Intent should be checked. The following code demonstrates this.


```java
public class ShutdownReceiver extends BroadcastReceiver {
    @Override
    public void onReceive(final Context context, final Intent intent) {
        if (!intent.getAction().equals(Intent.ACTION_SHUTDOWN)) {
            return;
        }
        mainActivity.saveLocalData();
        mainActivity.stopActivity();
    }
}
```

## References
* Common Weakness Enumeration: [CWE-925](https://cwe.mitre.org/data/definitions/925.html).
