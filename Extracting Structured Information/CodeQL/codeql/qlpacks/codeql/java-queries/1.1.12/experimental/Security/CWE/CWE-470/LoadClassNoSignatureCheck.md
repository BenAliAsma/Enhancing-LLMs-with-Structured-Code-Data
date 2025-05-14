# Load 3rd party classes or code ('unsafe reflection') without signature check
If an application loads classes or code from another app based solely on its package name without first checking its package signature, this could allow a malicious app with the same package name to be loaded through "package namespace squatting". If the victim user install such malicious app in the same device as the vulnerable app, the vulnerable app would load classes or code from the malicious app, potentially leading to arbitrary code execution.


## Recommendation
Verify the package signature in addition to the package name before loading any classes or code from another application.


## Example
The `BadClassLoader` class illustrates class loading with the `android.content.pm.PackageInfo.packageName.startsWith()` method without any check on the package signature.


```java
package poc.sample.classloader;

import android.app.Application;
import android.content.pm.PackageInfo;
import android.content.Context;
import android.util.Log;

public class BadClassLoader extends Application {
    @Override
    public void onCreate() {
        super.onCreate();
        for (PackageInfo p : getPackageManager().getInstalledPackages(0)) {
            try {
                if (p.packageName.startsWith("some.package.")) {
                    Context appContext = createPackageContext(p.packageName,
                            CONTEXT_INCLUDE_CODE | CONTEXT_IGNORE_SECURITY);
                    ClassLoader classLoader = appContext.getClassLoader();
                    Object result = classLoader.loadClass("some.package.SomeClass")
                            .getMethod("someMethod")
                            .invoke(null);
                }
            } catch (Exception e) {
                Log.e("Class loading failed", e.toString());
            }
        }
    }
}

```
The `GoodClassLoader` class illustrates class loading with correct package signature check using the `android.content.pm.PackageManager.checkSignatures()` method.


```java
package poc.sample.classloader;

import android.app.Application;
import android.content.pm.PackageInfo;
import android.content.Context;
import android.content.pm.PackageManager;
import android.util.Log;

public class GoodClassLoader extends Application {
    @Override
    public void onCreate() {
        super.onCreate();
        PackageManager pm = getPackageManager();
        for (PackageInfo p : pm.getInstalledPackages(0)) {
            try {
                if (p.packageName.startsWith("some.package.") &&
                        (pm.checkSignatures(p.packageName, getApplicationContext().getPackageName()) == PackageManager.SIGNATURE_MATCH)
                ) {
                    Context appContext = createPackageContext(p.packageName,
                            CONTEXT_INCLUDE_CODE | CONTEXT_IGNORE_SECURITY);
                    ClassLoader classLoader = appContext.getClassLoader();
                    Object result = classLoader.loadClass("some.package.SomeClass")
                            .getMethod("someMethod")
                            .invoke(null);
                }
            } catch (Exception e) {
                Log.e("Class loading failed", e.toString());
            }
        }
    }
}

```

## References
* [ Oversecured (Android: arbitrary code execution via third-party package contexts) ](https://blog.oversecured.com/Android-arbitrary-code-execution-via-third-party-package-contexts/)
* Common Weakness Enumeration: [CWE-470](https://cwe.mitre.org/data/definitions/470.html).
